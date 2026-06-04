import cv2
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from .decoders import BaseDecoder


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    roi_pixmap_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def __init__(self, decoder: BaseDecoder):
        super().__init__()
        self._run_flag = True
        self.decoder = decoder
        self._camera_index = 5
        self._pending_camera_index = 5
        self._camera_mutex = QMutex()

    def run(self):
        cap = self._open_capture(self._camera_index)

        while self._run_flag:
            next_camera_index = self._take_pending_camera_index()
            if next_camera_index != self._camera_index:
                cap.release()
                self._camera_index = next_camera_index
                cap = self._open_capture(self._camera_index)

            ret, frame = cap.read()
            if not ret:
                self.msleep(50)
                continue

            results = self.decoder.decode(frame)
            self._draw_yolo_detections(frame)
            for result in results:
                self._draw_result(frame, result)
                self.log_signal.emit(self._build_log_message(result))

            qt_image = self._convert_frame_to_qimage(frame)
            self.change_pixmap_signal.emit(qt_image)
            self.roi_pixmap_signal.emit(
                self._convert_debug_image_to_qimage(self.decoder.last_zbar_input_image)
            )

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_camera_index(self, camera_index):
        with QMutexLocker(self._camera_mutex):
            self._pending_camera_index = camera_index

    def camera_index(self):
        with QMutexLocker(self._camera_mutex):
            return self._pending_camera_index

    def _take_pending_camera_index(self):
        with QMutexLocker(self._camera_mutex):
            return self._pending_camera_index

    def _open_capture(self, camera_index):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.log_signal.emit(f"--- 카메라 {camera_index} 열기 ---")
        return cap

    def _draw_result(self, frame, result):
        x, y, w, h = result.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{result.data} ({result.barcode_type})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def _draw_yolo_detections(self, frame):
        for x, y, w, h in self.decoder.last_yolo_detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(
                frame,
                "YOLO",
                (x, max(y - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 128, 0),
                2,
            )

    def _build_log_message(self, result):
        log_msg = f"[{result.barcode_type}] {result.data}"
        if self.decoder.debug_mode:
            if result.physical_pattern:
                log_msg += f"\n └─ [물리 패턴] {result.physical_pattern}"
            elif result.barcode_type == "QRCODE":
                log_msg += "\n └─ [DEBUG] QR코드는 2D 패턴이라 1D 스캔라인을 지원하지 않습니다."

        return log_msg

    def _convert_frame_to_qimage(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb_image.shape
        bytes_per_line = ch * w_img
        return QImage(
            rgb_image.data,
            w_img,
            h_img,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

    def _convert_debug_image_to_qimage(self, image):
        if image is None:
            return QImage()

        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h_img, w_img, ch = rgb_image.shape
        bytes_per_line = ch * w_img
        return QImage(
            rgb_image.data,
            w_img,
            h_img,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
