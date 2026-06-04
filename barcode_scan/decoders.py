from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
from pyzbar import pyzbar

from .models import ScanResult


class BaseDecoder(ABC):
    """Barcode decoder interface."""

    def __init__(self):
        self.min_size = 30
        self.scan_mode = "모두 인식"
        self.debug_mode = False
        self.enhanced_scan = True
        self.yolo_scan = True
        self.yolo_model_path = "models/YOLOV8s_Barcode_Detection.pt"
        self.yolo_confidence = 0.35
        self.yolo_min_edge_density = 0.003
        self.last_yolo_detections = []
        self.last_zbar_input_image = None
        self._yolo_detector = None

    @abstractmethod
    def decode(self, frame) -> List[ScanResult]:
        pass


class PyzbarDecoder(BaseDecoder):
    """Barcode decoder backed by pyzbar."""

    def decode(self, frame) -> List[ScanResult]:
        self.last_yolo_detections = []
        self.last_zbar_input_image = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self._decode_candidate(gray, gray, self._identity_rect)
        if results:
            return results

        if self.yolo_scan:
            results = self._decode_yolo_rois(frame, gray)
            if results:
                return results

        if not self.enhanced_scan:
            return results

        seen = {(result.barcode_type, result.data) for result in results}
        for candidate, map_rect in self._build_enhanced_candidates(gray):
            candidate_results = self._decode_candidate(candidate, gray, map_rect)
            for result in candidate_results:
                key = (result.barcode_type, result.data)
                if key not in seen:
                    results.append(result)
                    seen.add(key)

            if results:
                break

        return results

    def _decode_yolo_rois(self, frame, gray) -> List[ScanResult]:
        detector = self._get_yolo_detector()
        if detector is None:
            return []

        results = []
        seen = set()
        detections = detector.detect(frame)
        self.last_yolo_detections = detections

        for x, y, w, h in detections:
            if w < self.min_size or h < self.min_size:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            if roi_gray.size == 0:
                continue

            for candidate, candidate_mapper in self._build_roi_candidates(roi_gray):
                self.last_zbar_input_image = candidate.copy()
                roi_results = self._decode_candidate(
                    candidate,
                    gray,
                    self._offset_rect_mapper(x, y, candidate_mapper),
                )
                for result in roi_results:
                    key = (result.barcode_type, result.data)
                    if key not in seen:
                        results.append(result)
                        seen.add(key)

                if results:
                    break

        return results

    def _get_yolo_detector(self):
        if self._yolo_detector is False:
            return None
        if self._yolo_detector is not None:
            return self._yolo_detector

        model_path = Path(self.yolo_model_path)
        if not model_path.exists():
            self._yolo_detector = False
            return None

        try:
            self._yolo_detector = YoloBarcodeDetector(
                model_path, self.yolo_confidence, self.yolo_min_edge_density
            )
        except (ImportError, OSError, RuntimeError):
            self._yolo_detector = False
            return None

        return self._yolo_detector

    def _decode_candidate(
        self,
        image,
        original_gray,
        map_rect: Callable[[Tuple[int, int, int, int]], Tuple[int, int, int, int]],
    ) -> List[ScanResult]:
        barcodes = pyzbar.decode(image)
        results = []

        for barcode in barcodes:
            barcode_type = barcode.type
            x, y, w, h = map_rect(tuple(barcode.rect))

            if w < self.min_size or h < self.min_size:
                continue

            is_qr = barcode_type == "QRCODE"
            if self.scan_mode == "1D 바코드만" and is_qr:
                continue
            if self.scan_mode == "QR 코드만" and not is_qr:
                continue

            decoded_text = barcode.data.decode("utf-8", errors="ignore")
            physical_pattern = None

            if self.debug_mode and not is_qr:
                physical_pattern = self._extract_physical_pattern(
                    original_gray, x, y, w, h
                )

            results.append(
                ScanResult(decoded_text, barcode_type, (x, y, w, h), physical_pattern)
            )

        return results

    def _build_enhanced_candidates(self, gray):
        for candidate in self._preprocess_candidates(gray):
            yield candidate, self._identity_rect

        for scale in (1.5, 2.0, 0.75):
            resized = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            yield resized, self._scale_rect_mapper(scale)

        for angle in (-15, 15, -30, 30, -45, 45, 90):
            rotated, matrix = self._rotate_bound(gray, angle)
            yield rotated, self._affine_rect_mapper(matrix, gray.shape)

    def _build_roi_candidates(self, roi_gray):
        yield roi_gray, self._identity_rect

        if self.enhanced_scan:
            yield from self._build_enhanced_candidates(roi_gray)

    def _preprocess_candidates(self, gray):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        yield clahe

        yield cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yield cv2.morphologyEx(clahe, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _rotate_bound(self, image, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, matrix

    def _identity_rect(self, rect):
        return rect

    def _scale_rect_mapper(self, scale):
        def map_rect(rect):
            x, y, w, h = rect
            return (
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
            )

        return map_rect

    def _offset_rect_mapper(self, offset_x, offset_y, inner_mapper):
        def map_rect(rect):
            x, y, w, h = inner_mapper(rect)
            return (x + offset_x, y + offset_y, w, h)

        return map_rect

    def _affine_rect_mapper(self, matrix, original_shape):
        inverse = cv2.invertAffineTransform(matrix)
        original_h, original_w = original_shape[:2]

        def map_rect(rect):
            x, y, w, h = rect
            corners = np.array(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ],
                dtype=np.float32,
            )
            restored = cv2.transform(np.array([corners]), inverse)[0]
            x1, y1 = restored.min(axis=0)
            x2, y2 = restored.max(axis=0)

            x1 = max(0, min(original_w - 1, int(round(x1))))
            y1 = max(0, min(original_h - 1, int(round(y1))))
            x2 = max(0, min(original_w, int(round(x2))))
            y2 = max(0, min(original_h, int(round(y2))))
            return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

        return map_rect

    def _extract_physical_pattern(self, gray, x: int, y: int, w: int, h: int):
        img_h, img_w = gray.shape
        center_y = min(max(y + (h // 2), 0), img_h - 1)
        safe_x1 = max(x, 0)
        safe_x2 = min(x + w, img_w)

        scanline = gray[center_y : center_y + 1, safe_x1:safe_x2]
        if scanline.size == 0:
            return None

        _, binary_line = cv2.threshold(
            scanline, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        binary_line = binary_line.flatten()
        raw_bits = ["1" if p == 0 else "0" for p in binary_line]
        pattern_str = "".join(raw_bits)
        return pattern_str[:100] + ("..." if len(pattern_str) > 100 else "")


class YoloBarcodeDetector:
    def __init__(self, model_path: Path, confidence: float, min_edge_density: float):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.min_edge_density = min_edge_density

    def detect(self, frame):
        predictions = self.model.predict(
            frame, imgsz=640, conf=self.confidence, max_det=5, verbose=False
        )
        if not predictions:
            return []

        boxes = predictions[0].boxes
        if boxes is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        detections = []
        for xyxy in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(frame_w - 1, int(round(x1))))
            y1 = max(0, min(frame_h - 1, int(round(y1))))
            x2 = max(0, min(frame_w, int(round(x2))))
            y2 = max(0, min(frame_h, int(round(y2))))

            x, y, w, h = self._pad_rect(x1, y1, x2 - x1, y2 - y1, frame_w, frame_h)
            if (
                w > 0
                and h > 0
                and self._has_barcode_texture(frame[y : y + h, x : x + w])
            ):
                detections.append((x, y, w, h))

        detections.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
        return detections

    def _pad_rect(self, x, y, w, h, frame_w, frame_h):
        pad_x = max(8, int(w * 0.15))
        pad_y = max(8, int(h * 0.20))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)
        return (x1, y1, x2 - x1, y2 - y1)

    def _has_barcode_texture(self, roi):
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = cv2.countNonZero(edges) / edges.size
        return edge_density >= self.min_edge_density
