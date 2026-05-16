import sys
import cv2
import numpy as np
import time
import threading
import os
import shutil
from collections import deque
from picamera2 import Picamera2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QWidget, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# 导入 SGBM 文件中的所有内容
from sgbm_640_30fps_depth_cap import *

# --- Hardware Telemetry Helpers ---
def get_storage_stats(path="/"):
    """
    获取指定路径的存储空间使用率。
    默认路径 “/” 代表树莓派内部的 SD 卡。
    """
    try:
        total, used, free = shutil.disk_usage(path)
        return (used / total) * 100
    except Exception:
        return 0.0



class StereoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.trigger_save = False
        self.is_saving = False
        self.last_touch_coords = None
        self.current_mode = 0  
        self.mode_names = ["OVERLAY", "PAIR", "DISP"]
        self._stop_ref = None # 用于保存内部 stop 变量的引用，以便安全退出

    def run(self):
        # 1. 初始化参数 (与原代码完全一致)
        params = np.load(CALIB_NPZ)
        image_size = tuple(params["image_size"])
        full_w, full_h = int(image_size[0]), int(image_size[1])

        map1_x, map1_y = params["map1_x"].astype(np.float32), params["map1_y"].astype(np.float32)
        map2_x, map2_y = params["map2_x"].astype(np.float32), params["map2_y"].astype(np.float32)
        fx, baseline_m = get_fx_baseline_from_params(params)
        crop_xywh = compute_crop_from_valid_rois(params, full_w, full_h)

        dataset_path = "dataset_logs" 
        dirs = ensure_dataset_dirs(dataset_path)

        camL = Picamera2(1)
        camR = Picamera2(0)
        cfgL = camL.create_video_configuration(main={"size": (full_w, full_h), "format": "RGB888"}, controls={"FrameRate": FRAME_RATE}, buffer_count=LOW_LATENCY_BUFFER_COUNT, queue=LOW_LATENCY_QUEUE)
        cfgR = camR.create_video_configuration(main={"size": (full_w, full_h), "format": "RGB888"}, controls={"FrameRate": FRAME_RATE}, buffer_count=LOW_LATENCY_BUFFER_COUNT, queue=LOW_LATENCY_QUEUE)
        camL.configure(cfgL); camR.configure(cfgR)
        camL.start(); camR.start()

        if USE_AE_AWB_CONVERGE_THEN_LOCK:
            converge_then_lock(camL, camR, settle_s=SETTLE_S, throw_n=THROW_N)
        else:
            throw_frames(camL, camR, n=30)


        lock = threading.Lock()
        stop = threading.Event()        # <-- 恢复为 stop，不再用 self.stop_event
        self._stop_ref = stop           # 保存引用，方便外部调用 stop() 时停止线程
        
        rawL = deque(maxlen=BUF_LEN)
        rawR = deque(maxlen=BUF_LEN)
        ids = {"L": 0, "R": 0}

        shared = {
            "rectL": None, "rectR": None, "disp": None, "valid": None,
            "disp_color": None, "dt_ms": None, "comp_fps": 0.0,
            "comp_ok": False, "pair_key": None
        }


        # =========================================================================
        
        def cap_thread(cam, side):
            fps = 0.0
            t_prev = None
            while not stop.is_set():
                rgb, ts = capture_one_with_ts(cam)
                now = time.perf_counter()
                if t_prev is None:
                    t_prev = now
                else:
                    dt = now - t_prev
                    t_prev = now
                    if dt > 1e-6:
                        inst = 1.0 / dt
                        fps = (1.0 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst

                with lock:
                    ids[side] += 1
                    if side == "L":
                        rawL.append((ids[side], rgb, ts))
                        shared["capL_fps"] = fps
                    else:
                        rawR.append((ids[side], rgb, ts))
                        shared["capR_fps"] = fps



        def comp_thread():
            comp_fps = 0.0
            t_prev_new = None
            last_pair_key = None
            last_post = None

            left_matcher = None
            right_matcher = None
            wls = None
            use_xi = has_ximgproc()
        
        
    
        threading.Thread(target=cap_thread, args=(camL, "L"), daemon=True).start()
        threading.Thread(target=cap_thread, args=(camR, "R"), daemon=True).start()
        threading.Thread(target=comp_thread, daemon=True).start()
        # =========================================================================

        t_next_ui = time.perf_counter()
        last_telemetry_check = 0
        storage_pct = 0.0

        # UI 循环也必须监听局部的 stop 变量
        while self.is_running and not stop.is_set():
            now = time.perf_counter()
            sleep_s = t_next_ui - now
            if sleep_s > 0: time.sleep(sleep_s)
            t_next_ui += UI_PERIOD_S

            if now - last_telemetry_check > 2.0:
                storage_pct = get_storage_stats("/") 
                last_telemetry_check = now

            with lock:
                rectL = shared.get("rectL")
                rectR = shared.get("rectR")
                disp_full = shared.get("disp")
                disp_color = shared.get("disp_color")
                valid = shared.get("valid")
                dt_ms = shared.get("dt_ms")
                comp_fps = shared.get("comp_fps", 0.0)
                pair_key = shared.get("pair_key")
                m = self.current_mode

            # 拍照保存
            if self.trigger_save and rectL is not None and disp_full is not None:
                self.is_saving = True 
                self.frame_ready.emit(np.zeros((10,10,3), dtype=np.uint8), {"is_saving": True, "mode": "", "comp_fps": 0, "dt_ms": 0, "storage_pct": 0, "touch_depth": "Saving..."})
                print(f"[INFO] Saving to {dataset_path}")
                save_capture_bundle(
                    dirs, rectL, rectR, disp_color, disp_full, valid,
                    fx, baseline_m, dt_ms, pair_key, POSTPROC_DEFAULT, 
                    DISP_SCALE, MIN_DISPARITIES_FULL_TARGET, NUM_DISPARITIES_FULL_TARGET
                )
                self.trigger_save = False 
                self.is_saving = False

            # 触摸测距
            touch_depth_str = "Touch to measure"
            if self.last_touch_coords is not None and disp_full is not None:
                tx, ty = self.last_touch_coords
                if 0 <= ty < disp_full.shape[0] and 0 <= tx < disp_full.shape[1]:
                    disp_val = disp_full[ty, tx]
                    if disp_val > 0:
                        z_mm = (fx * baseline_m) / disp_val * 1000
                        touch_depth_str = f"Target: {z_mm:.0f} mm"
                    else:
                        touch_depth_str = "Target: Invalid Depth"

            # 画面合成
            if rectL is None:
                frame = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            elif m == 0:  
                frame = overlay_depth_on_left(rectL, disp_color)
            elif m == 1: 
                if rectR is None:
                    frame = cv2.cvtColor(rectL, cv2.COLOR_RGB2BGR) # 注意：如果是普通模式拼接可能需要BGR转换，具体看你们组员逻辑
                else:
                    pair = np.hstack([rectL, rectR])
                    frame = pair # 直接传 RGB 即可
            elif m == 2:  
                frame = disp_color.copy() if disp_color is not None else np.zeros((rectL.shape[0], rectL.shape[1], 3), dtype=np.uint8)

            telemetry = {
                "comp_fps": comp_fps,
                "dt_ms": dt_ms if dt_ms is not None else 0.0,
                "mode": self.mode_names[m],
                "storage_pct": storage_pct,
                "is_saving": self.is_saving,
                "touch_depth": touch_depth_str
            }
            
            self.frame_ready.emit(frame, telemetry)

        # 退出清理
        if self._stop_ref:
            self._stop_ref.set()
        time.sleep(0.2)
        camL.stop(); camR.stop()

    def stop(self):
        self.is_running = False
        if self._stop_ref:
            self._stop_ref.set() # 触发组员原生的 stop 事件
        self.wait()


# 计算屏幕触摸坐标并映射回图像坐标的 QLabel 子类
class VideoLabel(QLabel):
    """
    自定义的视频显示标签类（继承自 QLabel）。
    它的作用是让屏幕上的视频画面变成“可触摸/可点击”的。
    当你在屏幕上点击某个物体时，它会计算出该点在 640x480 原始图像中的真实像素坐标。
    """
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        # 设置画面居中，背景全黑，修复边框自适应导致的无限放大Bug
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def mousePressEvent(self, event):
        """
        PyQt 的内置鼠标/触摸按下事件。在这里进行坐标映射的数学计算。
        """
        if self.pixmap() is None:
            return
            
        
        label_w, label_h = self.width(), self.height()
        pixmap_w, pixmap_h = self.pixmap().width(), self.pixmap().height()

        #  计算画面上下或左右的黑边宽度
        offset_x = (label_w - pixmap_w) / 2.0
        offset_y = (label_h - pixmap_h) / 2.0

        # 计算触点在实际图像（去除黑边后）上的相对坐标
        touch_x = event.pos().x() - offset_x
        touch_y = event.pos().y() - offset_y

        # 如果触点确实在图像范围内，则将其映射回设定分辨率（例如 640x480）
        if 0 <= touch_x <= pixmap_w and 0 <= touch_y <= pixmap_h:
            native_x = int((touch_x / pixmap_w) * 640)
            native_y = int((touch_y / pixmap_h) * 480)
            self.clicked.emit(native_x, native_y)





# UI 组件和主窗口
class PiStereoApp(QMainWindow):
    """
    主用户界面类（运行在主线程）。
    负责所有的按钮绘制、触摸响应以及接收后台发来的视频画面。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Vision Pi 5")
        self.showFullScreen()

        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #121212; color: white;")
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 添加我们自定义的支持触摸测距的视频画面组件
        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self.handle_video_click)
        layout.addWidget(self.video_label, stretch=1)
        
        # 顶部的状态信息栏 (FPS, 延迟, 距离等)
        self.telemetry_label = QLabel("Initializing Hardware Pipeline...")
        self.telemetry_label.setStyleSheet("font-size: 20px; padding: 10px; font-weight: bold;")
        layout.addWidget(self.telemetry_label)

        control_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("Capture Dataset Snapshot")
        self.btn_capture.setMinimumHeight(80)
        self.btn_capture.setStyleSheet("font-size: 20px; background-color: #2d2d2d; border-radius: 10px;")
        self.btn_capture.clicked.connect(self.trigger_capture)
        control_layout.addWidget(self.btn_capture)

        self.btn_mode = QPushButton("Cycle View Mode")
        self.btn_mode.setMinimumHeight(80)
        self.btn_mode.setStyleSheet("font-size: 20px; background-color: #2d2d2d; border-radius: 10px;")
        self.btn_mode.clicked.connect(self.cycle_mode)
        control_layout.addWidget(self.btn_mode)
        
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setMinimumHeight(80)
        self.btn_exit.setStyleSheet("font-size: 20px; background-color: #8b0000; border-radius: 10px;")
        self.btn_exit.clicked.connect(self.close)
        control_layout.addWidget(self.btn_exit)

        layout.addLayout(control_layout)
        # 实例化并启动后台硬件线程
        self.worker = StereoWorker()
        self.worker.frame_ready.connect(self.update_ui)
        self.worker.start()

    def update_ui(self, frame, telemetry):
        """
        此函数由后台线程的 frame_ready 信号自动触发。
        """
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 将画面缩放以适应屏幕，保持原始比例
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

        # 如果后台正在向硬盘写入庞大的数据集，UI变红警告用户不要继续狂按
        if telemetry.get("is_saving", False):
            self.telemetry_label.setStyleSheet("font-size: 20px; padding: 10px; font-weight: bold; color: red; background-color: #330000;")
            status_text = "SAVING DATASET TO DISK... PLEASE WAIT"
        else:
            self.telemetry_label.setStyleSheet("font-size: 20px; padding: 10px; font-weight: bold; color: white; background-color: transparent;")
            status_text = (
                f"FPS: {telemetry['comp_fps']:.1f} | "
                f"Lat: {telemetry['dt_ms']:.1f}ms | "
                f"SD: {telemetry['storage_pct']:.1f}% | "
                f"Mode: {telemetry['mode']} | "
                f"<span style='color: #00ff00;'>{telemetry['touch_depth']}</span>" # Highlights the depth in green!
            )

        self.telemetry_label.setText(status_text)

    def trigger_capture(self):
        """
        拍照按钮被按下的逻辑。通知后台保存图片，并让按钮闪烁反馈。
        """
        self.worker.trigger_save = True
        
        # 快门
        self.btn_capture.setStyleSheet("background-color: white; color: black; font-size: 20px; border-radius: 10px;")
        
        QTimer.singleShot(200, lambda: self.btn_capture.setStyleSheet("background-color: #2d2d2d; color: white; font-size: 20px; border-radius: 10px;"))

    def cycle_mode(self):
        """
        切换视图模式
        """
        self.worker.current_mode = (self.worker.current_mode + 1) % 3

    def closeEvent(self, event):
        """
        当点击 Exit 或关闭窗口时触发
        """
        self.worker.stop()
        event.accept()

    def handle_video_click(self, x, y):
        """
        接收 VideoLabel 发来的触摸坐标，并传给后台线程计算深度。
        """
        # Send the coordinates to the background worker to look up the depth
        self.worker.last_touch_coords = (x, y)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PiStereoApp()
    window.show()
    sys.exit(app.exec())