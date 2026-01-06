from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .settings import UiSettings
from .frame_provider import FrameProvider
from .recorder import FrameRecorder, RecordConfig
from .system_stats import get_system_stats


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pi5 Stereo Vision UI (PyQt6)")

        self.s = UiSettings()
        self.provider = FrameProvider(self.s, use_webcam=False)  # set True if you want /dev/video0
        self.recorder = FrameRecorder()

        self._frame_idx = 0
        self._last_fps_t = time.time()
        self._fps_counter = 0
        self._fps_value = 0.0

        # --- UI widgets ---
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background:#111; color:#ddd;")
        self.preview_label.setMinimumSize(800, 450)

        self.btn_preview = QPushButton("Start Preview")
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setEnabled(False)

        self.chk_show_right = QCheckBox("Show Right")
        self.chk_show_right.setChecked(self.s.show_right)
        self.chk_show_disp = QCheckBox("Show Disparity")
        self.chk_show_disp.setChecked(self.s.show_disparity)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(5, 120)
        self.spin_fps.setValue(self.s.target_fps)

        self.res_combo = QComboBox()
        self.resolutions = [
            (640, 480),
            (848, 480),
            (960, 540),
            (1280, 720),
        ]
        for (w, h) in self.resolutions:
            self.res_combo.addItem(f"{w} x {h}", (w, h))
        self._set_combo_to_current_resolution()

        # status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("FPS: -- | CPU: -- | Free: -- GB")
        self.status.addPermanentWidget(self.status_label)

        # layout
        right_panel = self._build_controls_panel()

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.preview_label, stretch=3)
        layout.addWidget(right_panel, stretch=1)
        self.setCentralWidget(root)

        # menu
        self._build_menu()

        # timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_preview)

        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._tick_stats)
        self.stats_timer.start(1000)

        # connections
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_record.clicked.connect(self._toggle_recording)
        self.chk_show_right.stateChanged.connect(self._update_view_flags)
        self.chk_show_disp.stateChanged.connect(self._update_view_flags)
        self.spin_fps.valueChanged.connect(self._update_fps)
        self.res_combo.currentIndexChanged.connect(self._update_resolution)

    def closeEvent(self, event) -> None:
        try:
            self.timer.stop()
            self.stats_timer.stop()
            if self.recorder.is_recording:
                self.recorder.stop()
            self.provider.close()
        finally:
            event.accept()

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        act_export = QAction("Export UI Screenshot", self)
        act_export.triggered.connect(self._export_screenshot)
        file_menu.addAction(act_export)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        # Actions group
        g_actions = QGroupBox("Actions")
        a = QVBoxLayout(g_actions)
        a.addWidget(self.btn_preview)
        a.addWidget(self.btn_record)
        v.addWidget(g_actions)

        # View group
        g_view = QGroupBox("View")
        f = QFormLayout(g_view)
        f.addRow(self.chk_show_right)
        f.addRow(self.chk_show_disp)
        v.addWidget(g_view)

        # Settings group
        g_settings = QGroupBox("Settings")
        s = QFormLayout(g_settings)
        s.addRow("Target FPS", self.spin_fps)
        s.addRow("Resolution", self.res_combo)
        v.addWidget(g_settings)

        # Info group
        g_info = QGroupBox("Info")
        info = QVBoxLayout(g_info)
        info.addWidget(QLabel("Integration tip: replace FrameProvider.get_frame()\nwith pipeline output.\n\nThis UI is designed to stay\nportable + operator-friendly."))
        v.addWidget(g_info)

        v.addStretch(1)
        return panel

    def _toggle_preview(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.btn_preview.setText("Start Preview")
            self.btn_record.setEnabled(False)
        else:
            interval_ms = int(1000 / max(1, self.s.target_fps))
            self.timer.start(interval_ms)
            self.btn_preview.setText("Stop Preview")
            self.btn_record.setEnabled(True)

    def _toggle_recording(self) -> None:
        if not self.timer.isActive():
            QMessageBox.information(self, "Info", "Start preview before recording.")
            return

        if self.recorder.is_recording:
            self.recorder.stop()
            self.btn_record.setText("Start Recording")
        else:
            out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
            if not out_dir:
                return
            # frame size is based on current preview image label content; safer: use next frame size
            frame = self.provider.get_frame()
            h, w = frame.shape[:2]
            cfg = RecordConfig(out_dir=Path(out_dir), fps=self.s.record_fps)
            self.recorder.start(cfg, frame_size=(w, h))
            self.btn_record.setText("Stop Recording")

    def _update_view_flags(self) -> None:
        self.s.show_right = self.chk_show_right.isChecked()
        self.s.show_disparity = self.chk_show_disp.isChecked()

    def _update_fps(self) -> None:
        self.s.target_fps = int(self.spin_fps.value())
        if self.timer.isActive():
            self.timer.setInterval(int(1000 / max(1, self.s.target_fps)))

    def _update_resolution(self) -> None:
        w, h = self.res_combo.currentData()
        self.s.width = int(w)
        self.s.height = int(h)

    def _set_combo_to_current_resolution(self) -> None:
        for i in range(self.res_combo.count()):
            w, h = self.res_combo.itemData(i)
            if (w, h) == (self.s.width, self.s.height):
                self.res_combo.setCurrentIndex(i)
                return

    def _tick_preview(self) -> None:
        frame = self.provider.get_frame()
        qimg = bgr_to_qimage(frame)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

        if self.recorder.is_recording:
            self.recorder.write(frame, self._frame_idx)

        self._frame_idx += 1
        self._fps_counter += 1
        now = time.time()
        if now - self._last_fps_t >= 1.0:
            self._fps_value = self._fps_counter / (now - self._last_fps_t)
            self._fps_counter = 0
            self._last_fps_t = now

    def _tick_stats(self) -> None:
        stats = get_system_stats()
        cpu = f"{stats.cpu_temp_c:.1f}°C" if stats.cpu_temp_c is not None else "--"
        free = f"{stats.disk_free_gb:.1f}" if stats.disk_free_gb is not None else "--"
        self.status_label.setText(f"FPS: {self._fps_value:.1f} | CPU: {cpu} | Free: {free} GB")

    def _export_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "ui_screenshot.png", "PNG Files (*.png)")
        if not path:
            return
        pix = self.centralWidget().grab()
        pix.save(path, "PNG")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
