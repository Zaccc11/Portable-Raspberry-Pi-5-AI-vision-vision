import cv2
import numpy as np
import time
import threading
from collections import deque
from picamera2 import Picamera2

# =========================
# Tunables (只改这里)
# =========================
FRAME_RATE = 60
LOW_LATENCY_BUFFER_COUNT = 2
LOW_LATENCY_QUEUE = True

TARGET_UI_FPS = 30.0
TARGET_COMP_FPS = 30.0
UI_PERIOD_S = 1.0 / TARGET_UI_FPS
COMP_PERIOD_S = 1.0 / TARGET_COMP_FPS

CALIB_NPZ = "stereo_params_640.npz"
ROI_BORDER_RATIO = 0.00  # 仍保留；推荐保持0，靠 validPixROI 裁剪

BUF_LEN = 8
#DT_OK_FOR_COMP_MS = 35.0
DT_OK_FOR_COMP_MS = 20.0

# SGBM（为30fps优化：先轻后重）
MIN_DISPARITIES = 0
NUM_DISPARITIES = 128
BLOCK_SIZE = 5
UNIQUENESS = 5
SPECKLE_WINSIZE = 0
SPECKLE_RANGE = 0
DISP12_MAXDIFF = 1

OVERLAY_ALPHA = 0.45
DISP_VIS_MAX = None

DISPLAY_SCALE = 0.75
DRAW_DEBUG_LINES = True
LINES_Y_FRAC = [0.25, 0.50, 0.75]

MEASURE_WIN = 11
MIN_VALID_PIXELS = 20
MIN_DISP_FOR_DEPTH = 0.5

USE_AE_AWB_CONVERGE_THEN_LOCK = True
SETTLE_S = 1.2
THROW_N = 25

FPS_EMA_ALPHA = 0.15

# --- Buttons ---
BTN_W = 140
BTN_H = 40
BTN_PAD = 10
BTN_GAP = 8
BTN_ALPHA_BG = 0.55
# =========================

# 稳定性：避免OpenCV内部线程/ocl抖动
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ========== SGBM ==========
def create_sgbm():
    P1 = 8 * BLOCK_SIZE * BLOCK_SIZE
    P2 = 32 * BLOCK_SIZE * BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITIES,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1, P2=P2,
        disp12MaxDiff=DISP12_MAXDIFF,
        uniquenessRatio=UNIQUENESS,
        speckleWindowSize=SPECKLE_WINSIZE,
        speckleRange=SPECKLE_RANGE,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ========== Camera helpers ==========
def capture_one_with_ts(cam):
    frame = cam.capture_array()
    meta = cam.capture_metadata()
    ts = meta.get("SensorTimestamp", time.time_ns())
    return frame, ts


def throw_frames(camL, camR, n=20):
    for _ in range(n):
        _ = camL.capture_array()
        _ = camR.capture_array()


def converge_then_lock(camL, camR, settle_s=1.2, throw_n=25):
    camL.set_controls({"AeEnable": True, "AwbEnable": True})
    camR.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(settle_s)
    throw_frames(camL, camR, n=throw_n)

    mL = camL.capture_metadata()
    exp = int(mL.get("ExposureTime", 8000))
    gain = float(mL.get("AnalogueGain", 2.0))
    cg = mL.get("ColourGains", None)

    controls = {"AeEnable": False, "AwbEnable": False, "ExposureTime": exp, "AnalogueGain": gain}
    if cg is not None:
        controls["ColourGains"] = cg
    camL.set_controls(controls)
    camR.set_controls(controls)
    throw_frames(camL, camR, n=20)

    print("[LOCK] ExposureTime:", exp, "AnalogueGain:", gain, "ColourGains:", cg)


# ========== Rectify ROI intersection ==========
def roi_intersection(roi1, roi2):
    """
    roi format: [x, y, w, h]
    return (x,y,w,h) for intersection; if no overlap -> None
    """
    x1, y1, w1, h1 = [int(v) for v in roi1]
    x2, y2, w2, h2 = [int(v) for v in roi2]
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    iw = xb - xa
    ih = yb - ya
    if iw <= 0 or ih <= 0:
        return None
    return (xa, ya, iw, ih)


def compute_crop_from_valid_rois(params, full_w, full_h):
    """
    Prefer validPixROI1/2 in npz. Fallback to full image if absent.
    Then apply optional ROI_BORDER_RATIO shrink.
    """
    have = ("validPixROI1" in params.files) and ("validPixROI2" in params.files)
    if have:
        roi1 = params["validPixROI1"].reshape(-1)
        roi2 = params["validPixROI2"].reshape(-1)
        inter = roi_intersection(roi1, roi2)
        if inter is None:
            print("[WARN] validPixROI1/2 have no intersection -> fallback to full frame crop")
            x, y, w, h = 0, 0, full_w, full_h
        else:
            x, y, w, h = inter
            print("[INFO] validPixROI1:", [int(v) for v in roi1])
            print("[INFO] validPixROI2:", [int(v) for v in roi2])
            print("[INFO] intersection ROI:", (x, y, w, h))
    else:
        print("[WARN] npz missing validPixROI1/2 -> no rectified ROI crop (full frame)")
        x, y, w, h = 0, 0, full_w, full_h

    # apply ROI_BORDER_RATIO on top (shrink inside the chosen ROI)
    if ROI_BORDER_RATIO > 0:
        dx = int(w * ROI_BORDER_RATIO)
        dy = int(h * ROI_BORDER_RATIO)
        x = x + dx
        y = y + dy
        w = max(1, w - 2 * dx)
        h = max(1, h - 2 * dy)
        print("[INFO] ROI_BORDER_RATIO shrink ->", (x, y, w, h))

    # clamp
    x = max(0, min(x, full_w - 1))
    y = max(0, min(y, full_h - 1))
    w = max(1, min(w, full_w - x))
    h = max(1, min(h, full_h - y))
    return (x, y, w, h)


def rectify_and_crop_common(rgbL, rgbR, map1_x, map1_y, map2_x, map2_y, crop_xywh):
    """
    先全图remap得到 rectL_full/rectR_full
    再用共同有效ROI crop，保证 L/R/disp/overlay 坐标一致
    """
    rectL = cv2.remap(rgbL, map1_x, map1_y, cv2.INTER_LINEAR)
    rectR = cv2.remap(rgbR, map2_x, map2_y, cv2.INTER_LINEAR)

    x, y, w, h = crop_xywh
    rectL = rectL[y:y + h, x:x + w]
    rectR = rectR[y:y + h, x:x + w]
    return rectL, rectR


# ========== Disparity / overlay ==========
def disp_q4_to_float_and_valid(disp_q4):
    invalid_q4 = MIN_DISPARITIES * 16 - 16
    valid = disp_q4 > invalid_q4
    disp = disp_q4.astype(np.float32) / 16.0
    disp[~valid] = 0.0
    disp = np.clip(disp, 0.0, float(NUM_DISPARITIES - 1))
    return disp, valid


def disp_to_colormap_fast(disp, valid):
    if valid is None or np.count_nonzero(valid) == 0:
        h, w = disp.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)
    maxd = (NUM_DISPARITIES - 1) if DISP_VIS_MAX is None else float(DISP_VIS_MAX)
    maxd = max(1.0, float(maxd))
    vis = np.clip(disp / maxd * 255.0, 0, 255).astype(np.uint8)
    vis[~valid] = 0
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def overlay_depth_on_left(rectL_rgb, disp_color_bgr):
    base_bgr = cv2.cvtColor(rectL_rgb, cv2.COLOR_RGB2BGR)
    if disp_color_bgr is None:
        return base_bgr
    return cv2.addWeighted(base_bgr, 1.0 - OVERLAY_ALPHA, disp_color_bgr, OVERLAY_ALPHA, 0.0)


def resize_for_display(img_bgr, scale):
    if scale >= 0.999:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ========== Pairing ==========
def find_best_pair_O_N(rawL, rawR):
    if not rawL or not rawR:
        return None, None
    idL, rgbL, tsL = rawL[-1]
    best = None
    best_dt = None
    for (idR, rgbR, tsR) in rawR:
        dt_ms = abs(int(tsL) - int(tsR)) / 1e6
        if best is None or dt_ms < best_dt:
            best = (idL, rgbL, tsL, idR, rgbR, tsR)
            best_dt = dt_ms
    return best, best_dt


# ========== Depth measure ==========
def measure_depth_at(disp, valid, x, y, fx, baseline_m, win=11, min_valid=20):
    h, w = disp.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, {"reason": "out_of_range"}

    r = win // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    d_patch = disp[y0:y1, x0:x1]
    v_patch = valid[y0:y1, x0:x1]
    vals = d_patch[v_patch]

    if vals.size < min_valid:
        return None, {"reason": "too_few_valid", "valid": int(vals.size)}

    d_med = float(np.median(vals))
    if d_med < MIN_DISP_FOR_DEPTH:
        return None, {"reason": "disp_too_small", "d_med": d_med, "valid": int(vals.size)}

    z_m = (fx * baseline_m) / d_med
    return float(z_m), {"d_med": d_med, "valid": int(vals.size), "win": win}


def get_fx_baseline_from_params(params):
    fx = float(params["P1"][0, 0]) if "P1" in params.files else float(params["cameraMatrix1"][0, 0])

    baseline_m = None
    if "Q" in params.files:
        Q = params["Q"].astype(np.float64)
        if abs(Q[3, 2]) > 1e-12:
            baseline_m = abs(1.0 / Q[3, 2])
    if baseline_m is None and "T" in params.files:
        T = params["T"].reshape(-1).astype(np.float64)
        baseline_m = float(abs(T[0]))
    if baseline_m is None:
        baseline_m = 0.06
    return fx, baseline_m


# ========== Buttons ==========
def draw_button(img, rect, label, active=False):
    x0, y0, x1, y1 = rect
    overlay = img.copy()
    bg = (70, 70, 70) if not active else (120, 120, 120)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, thickness=-1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), thickness=2)
    img[:] = cv2.addWeighted(overlay, BTN_ALPHA_BG, img, 1.0 - BTN_ALPHA_BG, 0.0)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 + th) // 2
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def point_in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def compute_button_rects(frame_w):
    x1 = frame_w - BTN_PAD
    x0 = x1 - BTN_W
    y = BTN_PAD
    rects = {}
    rects["OVERLAY"] = (x0, y, x1, y + BTN_H); y += BTN_H + BTN_GAP
    rects["PAIR"]    = (x0, y, x1, y + BTN_H); y += BTN_H + BTN_GAP
    rects["DISP"]    = (x0, y, x1, y + BTN_H); y += BTN_H + BTN_GAP
    rects["CLEAR"]   = (x0, y, x1, y + BTN_H)
    return rects


# ========== MAIN ==========
def main():
    params = np.load(CALIB_NPZ)
    image_size = tuple(params["image_size"])
    full_w, full_h = int(image_size[0]), int(image_size[1])
    print("[INFO] calib:", CALIB_NPZ, "image_size:", (full_w, full_h))

    map1_x, map1_y = params["map1_x"].astype(np.float32), params["map1_y"].astype(np.float32)
    map2_x, map2_y = params["map2_x"].astype(np.float32), params["map2_y"].astype(np.float32)

    fx, baseline_m = get_fx_baseline_from_params(params)
    print(f"[INFO] fx={fx:.2f}px baseline={baseline_m*1000:.1f}mm")

    # ---- compute common crop ROI from validPixROI intersection ----
    crop_xywh = compute_crop_from_valid_rois(params, full_w, full_h)

    cams = Picamera2.global_camera_info()
    print("[INFO] Detected cameras:", len(cams))
    for i, c in enumerate(cams):
        print("  cam", i, c)
    if len(cams) < 2:
        raise RuntimeError("Less than 2 cameras detected!")

    camL = Picamera2(1)
    camR = Picamera2(0)

    cfgL = camL.create_video_configuration(
        main={"size": (full_w, full_h), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE},
        buffer_count=LOW_LATENCY_BUFFER_COUNT,
        queue=LOW_LATENCY_QUEUE,
    )
    cfgR = camR.create_video_configuration(
        main={"size": (full_w, full_h), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE},
        buffer_count=LOW_LATENCY_BUFFER_COUNT,
        queue=LOW_LATENCY_QUEUE,
    )
    camL.configure(cfgL); camR.configure(cfgR)
    camL.start(); camR.start()

    if USE_AE_AWB_CONVERGE_THEN_LOCK:
        converge_then_lock(camL, camR, settle_s=SETTLE_S, throw_n=THROW_N)
    else:
        throw_frames(camL, camR, n=30)

    lock = threading.Lock()
    stop = threading.Event()

    rawL = deque(maxlen=BUF_LEN)
    rawR = deque(maxlen=BUF_LEN)
    ids = {"L": 0, "R": 0}

    # 0=OVERLAY 1=PAIR 2=DISP
    mode = {"val": 0}
    mode_names = ["OVERLAY", "PAIR", "DISP"]

    shared = {
        "rectL": None, "rectR": None,
        "disp": None, "valid": None,
        "disp_color": None,
        "dt_ms": None,
        "comp_fps": 0.0,
        "comp_ok": False,
        "last_click": None,
        "pair_key": None,
        "crop_xywh": crop_xywh,
    }

    def cap_thread(cam, side):
        while not stop.is_set():
            rgb, ts = capture_one_with_ts(cam)
            with lock:
                ids[side] += 1
                if side == "L":
                    rawL.append((ids[side], rgb, ts))
                else:
                    rawR.append((ids[side], rgb, ts))

    threading.Thread(target=cap_thread, args=(camL, "L"), daemon=True).start()
    threading.Thread(target=cap_thread, args=(camR, "R"), daemon=True).start()

    def comp_thread():
        sgbm = create_sgbm()
        comp_fps = 0.0
        t_prev_new = None
        last_pair_key = None
        t_next = time.perf_counter()

        while not stop.is_set():
            now = time.perf_counter()
            sleep_s = t_next - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += COMP_PERIOD_S

            with lock:
                Ls = list(rawL)
                Rs = list(rawR)

            if not Ls or not Rs:
                continue

            best, dt_ms = find_best_pair_O_N(Ls, Rs)
            if best is None or dt_ms is None or dt_ms > DT_OK_FOR_COMP_MS:
                with lock:
                    shared["dt_ms"] = dt_ms
                    shared["comp_ok"] = False
                continue

            idL, rgbL, tsL, idR, rgbR, tsR = best
            pair_key = (idL, idR)
            if pair_key == last_pair_key:
                with lock:
                    shared["dt_ms"] = dt_ms
                    shared["comp_ok"] = False
                continue

            rectL, rectR = rectify_and_crop_common(rgbL, rgbR, map1_x, map1_y, map2_x, map2_y, crop_xywh)

            gL = cv2.cvtColor(rectL, cv2.COLOR_RGB2GRAY)
            gR = cv2.cvtColor(rectR, cv2.COLOR_RGB2GRAY)
            disp_q4 = sgbm.compute(gL, gR)
            disp, valid = disp_q4_to_float_and_valid(disp_q4)
            disp_color = disp_to_colormap_fast(disp, valid)

            nowc = time.perf_counter()
            if t_prev_new is None:
                t_prev_new = nowc
            else:
                dtc = nowc - t_prev_new
                t_prev_new = nowc
                if dtc > 1e-6:
                    inst = 1.0 / dtc
                    comp_fps = (1.0 - FPS_EMA_ALPHA) * comp_fps + FPS_EMA_ALPHA * inst

            last_pair_key = pair_key

            with lock:
                shared["rectL"] = rectL
                shared["rectR"] = rectR
                shared["disp"] = disp
                shared["valid"] = valid
                shared["disp_color"] = disp_color
                shared["dt_ms"] = dt_ms
                shared["comp_fps"] = comp_fps
                shared["comp_ok"] = True
                shared["pair_key"] = pair_key

    threading.Thread(target=comp_thread, daemon=True).start()

    # ---- Mouse: click buttons or measure ----
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # display -> original coordinates
        if DISPLAY_SCALE < 0.999:
            x0 = int(x / DISPLAY_SCALE)
            y0 = int(y / DISPLAY_SCALE)
        else:
            x0, y0 = x, y

        with lock:
            m = mode["val"]
            rectL = shared["rectL"]
            rectR = shared["rectR"]
            disp = shared["disp"]
            valid = shared["valid"]

        if rectL is None:
            return

        # current frame width for button placement
        if m == 1 and rectR is not None:
            frame_w = rectL.shape[1] * 2
        else:
            frame_w = rectL.shape[1]

        rects = compute_button_rects(frame_w)

        # buttons first
        for name, r in rects.items():
            if point_in_rect(x0, y0, r):
                with lock:
                    if name == "OVERLAY":
                        mode["val"] = 0
                        shared["last_click"] = None
                    elif name == "PAIR":
                        mode["val"] = 1
                        shared["last_click"] = None
                    elif name == "DISP":
                        mode["val"] = 2
                        shared["last_click"] = None
                    elif name == "CLEAR":
                        shared["last_click"] = None
                return

        # not in buttons -> measure
        if disp is None or valid is None:
            return

        h, w = rectL.shape[:2]
        if m == 1:
            # PAIR mode: only left half corresponds to disp
            if x0 >= w:
                return
            x_use, y_use = x0, y0
        else:
            x_use, y_use = x0, y0

        z_m, info = measure_depth_at(disp, valid, x_use, y_use, fx, baseline_m,
                                     win=MEASURE_WIN, min_valid=MIN_VALID_PIXELS)
        with lock:
            shared["last_click"] = (x_use, y_use, z_m, info)

    cv2.namedWindow("stereo_rt_preview", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("stereo_rt_preview", on_mouse)

    ui_fps = 0.0
    t_ui_prev = None
    t_next_ui = time.perf_counter()

    try:
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k in [27, ord('q')]:
                break

            # UI paced (VNC friendly)
            now = time.perf_counter()
            sleep_s = t_next_ui - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next_ui += UI_PERIOD_S

            with lock:
                m = mode["val"]
                rectL = shared["rectL"]
                rectR = shared["rectR"]
                disp_color = shared["disp_color"]
                dt_ms = shared["dt_ms"]
                comp_fps = shared["comp_fps"]
                comp_ok = shared["comp_ok"]
                last_click = shared["last_click"]
                pair_key = shared["pair_key"]

            # compose frame
            if rectL is None:
                frame = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            else:
                if m == 0:  # OVERLAY
                    frame = overlay_depth_on_left(rectL, disp_color)
                elif m == 1:  # PAIR
                    if rectR is None:
                        frame = cv2.cvtColor(rectL, cv2.COLOR_RGB2BGR)
                    else:
                        pair = np.hstack([rectL, rectR])
                        frame = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
                        if DRAW_DEBUG_LINES:
                            hh, ww = frame.shape[:2]
                            for f in LINES_Y_FRAC:
                                y = int(hh * f)
                                cv2.line(frame, (0, y), (ww - 1, y), (0, 255, 0), 1)
                else:  # DISP
                    frame = disp_color.copy() if disp_color is not None else np.zeros((rectL.shape[0], rectL.shape[1], 3), dtype=np.uint8)

            # UI fps
            nowu = time.perf_counter()
            if t_ui_prev is None:
                t_ui_prev = nowu
            else:
                dtu = nowu - t_ui_prev
                t_ui_prev = nowu
                if dtu > 1e-6:
                    inst = 1.0 / dtu
                    ui_fps = (1.0 - FPS_EMA_ALPHA) * ui_fps + FPS_EMA_ALPHA * inst

            # HUD
            s_dt = "N/A" if dt_ms is None else f"{dt_ms:.2f}ms"
            s_new = "NEW" if comp_ok else "HOLD"
            cv2.putText(frame,
                        f"{mode_names[m]}  UI_FPS={ui_fps:.1f}  COMP_FPS={comp_fps:.1f}  dt={s_dt}  {s_new}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if pair_key is not None:
                cv2.putText(frame, f"pair={pair_key}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

            # click info (OVERLAY/DISP)
            if last_click is not None and m != 1:
                x, y, z_m, info = last_click
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)
                if z_m is None:
                    txt = f"({x},{y}) N/A info={info}"
                else:
                    txt = f"({x},{y}) d_med={info['d_med']:.3f}px  Z={z_m*1000:.1f}mm  valid={info['valid']}"
                cv2.rectangle(frame, (10, 60), (min(frame.shape[1]-1, 1250), 95), (0, 0, 0), -1)
                cv2.putText(frame, txt, (15, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # buttons
            frame_w = frame.shape[1]
            rects = compute_button_rects(frame_w)
            draw_button(frame, rects["OVERLAY"], "OVERLAY", active=(m == 0))
            draw_button(frame, rects["PAIR"],    "PAIR",    active=(m == 1))
            draw_button(frame, rects["DISP"],    "DISP",    active=(m == 2))
            draw_button(frame, rects["CLEAR"],   "CLEAR",   active=False)

            show = resize_for_display(frame, DISPLAY_SCALE)
            cv2.imshow("stereo_rt_preview", show)

    finally:
        stop.set()
        time.sleep(0.1)
        camL.stop(); camR.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()