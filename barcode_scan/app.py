import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .decoders import PyzbarDecoder
from .video_thread import VideoThread


class BarcodeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt 스마트 바코드 스캐너 (아키텍처 개선판)")
        self.resize(800, 600)
        self.last_scanned = ""
        self.decoder = PyzbarDecoder()

        self.init_ui()
        self.init_thread()

    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 480)

        self.roi_label = QLabel(self)
        self.roi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_label.setStyleSheet("background-color: #111; color: #aaa;")
        self.roi_label.setMinimumSize(260, 180)
        self.roi_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        roi_title = QLabel("YOLO ROI → zbar 입력")
        roi_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(roi_title)
        preview_layout.addWidget(self.roi_label)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.video_label, 3)
        camera_layout.addLayout(preview_layout, 1)

        control_layout = QHBoxLayout()

        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setRange(0, 20)
        self.camera_spinbox.setValue(5)
        self.camera_spinbox.valueChanged.connect(self.change_camera_index)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["모두 인식", "1D 바코드만", "QR 코드만"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)

        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(0, 300)
        self.size_spinbox.setValue(30)
        self.size_spinbox.setSuffix(" px")
        self.size_spinbox.valueChanged.connect(self.change_min_size)

        self.debug_checkbox = QCheckBox("디버그 모드 (물리패턴)")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug)

        self.enhanced_checkbox = QCheckBox("강화 스캔")
        self.enhanced_checkbox.setChecked(True)
        self.enhanced_checkbox.stateChanged.connect(self.toggle_enhanced_scan)

        self.yolo_checkbox = QCheckBox("YOLO ROI")
        self.yolo_checkbox.setChecked(True)
        self.yolo_checkbox.stateChanged.connect(self.toggle_yolo_scan)

        control_layout.addWidget(QLabel("카메라:"))
        control_layout.addWidget(self.camera_spinbox)
        control_layout.addWidget(QLabel(" | "))
        control_layout.addWidget(QLabel("모드:"))
        control_layout.addWidget(self.mode_combo)
        control_layout.addWidget(QLabel(" | 최소 크기:"))
        control_layout.addWidget(self.size_spinbox)
        control_layout.addStretch()
        control_layout.addWidget(self.yolo_checkbox)
        control_layout.addWidget(self.enhanced_checkbox)
        control_layout.addWidget(self.debug_checkbox)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)

        main_layout = QVBoxLayout()
        main_layout.addLayout(camera_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

    def init_thread(self):
        self.thread = VideoThread(self.decoder)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.roi_pixmap_signal.connect(self.update_roi_image)
        self.thread.log_signal.connect(self.update_log)
        self.thread.start()

    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_roi_image(self, qt_image):
        if qt_image.isNull():
            self.roi_label.clear()
            return

        pixmap = QPixmap.fromImage(qt_image)
        self.roi_label.setPixmap(
            pixmap.scaled(
                self.roi_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def update_log(self, msg):
        if msg != self.last_scanned:
            self.log_box.append(msg)
            self.last_scanned = msg
            scrollbar = self.log_box.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def change_mode(self, text):
        self.decoder.scan_mode = text
        self.log_box.append(f"--- ⚙️ 모드 변경: {text} ---")
        self.last_scanned = ""

    def change_min_size(self, value):
        self.decoder.min_size = value

    def change_camera_index(self, value):
        if hasattr(self, "thread"):
            self.thread.set_camera_index(value)
        self.log_box.append(f"--- 카메라 번호 변경: {value} ---")
        self.last_scanned = ""

    def toggle_debug(self, state):
        self.decoder.debug_mode = state == 2
        status = "활성화" if self.decoder.debug_mode else "비활성화"
        self.log_box.append(f"--- 🐛 디버그 모드 {status} ---")
        self.last_scanned = ""

    def toggle_enhanced_scan(self, state):
        self.decoder.enhanced_scan = state == 2
        status = "활성화" if self.decoder.enhanced_scan else "비활성화"
        self.log_box.append(f"--- 강화 스캔 {status} ---")
        self.last_scanned = ""

    def toggle_yolo_scan(self, state):
        self.decoder.yolo_scan = state == 2
        status = "활성화" if self.decoder.yolo_scan else "비활성화"
        self.log_box.append(f"--- YOLO ROI {status} ---")
        self.last_scanned = ""

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = BarcodeApp()
    window.show()
    sys.exit(app.exec())
