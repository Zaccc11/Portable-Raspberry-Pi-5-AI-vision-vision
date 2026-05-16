import cv2
import numpy as np
import time
import threading
from collections import deque
from picamera2 import Picamera2

# =========================
# Tunables（只改这里）
# =========================
FRAME_RATE = 60
LOW_LATENCY_BUFFER_COUNT = 2
LOW_LATENCY_QUEUE = True

TARGET_UI_FPS = 30.0
TARGET_COMP_FPS = 30.0
UI_PERIOD_S = 1.0 / TARGET_UI_FPS
COMP_PERIOD_S = 1.0 / TARGET_COMP_FPS

CALIB_NPZ = "stereo_params_640.npz"
ROI_BORDER_RATIO = 0.00

BUF_LEN = 4
DT_OK_FOR_COMP_MS = 25.0

DISP_SCALE = 0.5

MIN_DISPARITIES_FULL_TARGET = 32
NUM_DISPARITIES_FULL_TARGET = 192

BLOCK_SIZE = 3
UNIQUENESS = 3
DISP12_MAXDIFF = 1
SPECKLE_WINSIZE = 0
SPECKLE_RANGE = 0

POSTPROC_DEFAULT = "NONE"
WLS_LRC_THRESH = 6
WLS_LAMBDA = 150
WLS_SIGMA_COLOR = 0.7
WLS_DISC_RADIUS = 0
LRC_ONLY_LAMBDA = 0

ENABLE_SPECKLE_REMOVE = True
SPECKLE_MAX_SIZE = 160
SPECKLE_DIFF_PX = 2

OVERLAY_ALPHA = 0.45
DISP_VIS_MAX_FULL = None
DISPLAY_SCALE = 0.75
DRAW_DEBUG_LINES = True
LINES_Y_FRAC = [0.25, 0.50, 0.75]

MEASURE_WIN = 11
MIN_VALID_PIXELS = 40
MIN_DISP_FOR_DEPTH_FULL = 0.5

USE_AE_AWB_CONVERGE_THEN_LOCK = True
SETTLE_S = 1.2
THROW_N = 25

FPS_EMA_ALPHA = 0.15

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE_BIG = 0.62
HUD_SCALE_SMALL = 0.55
HUD_THICK = 2

BTN_W = 140
BTN_H = 38
BTN_PAD = 10
BTN_GAP = 8
BTN_ALPHA_BG = 0.55
# =========================

# =========================
# 稳定性设置
# =========================
cv2.setNumThreads(4)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# =========================
# 通用小工具
# =========================
def round_up_to_multiple(x, m):
    return int((x + (m - 1)) // m) * m


def compute_scaled_disp_params(full_min, full_num, scale):
    full_min = int(full_min)
    full_num = int(full_num)
    if full_num <= 0:
        full_num = 16

    if abs(scale - 1.0) < 1e-6:
        min_s = full_min
        num_s = round_up_to_multiple(full_num, 16)
        return min_s, num_s

    min_s = int(np.floor(full_min * scale + 1e-6))
    num_s_raw = full_num * scale
    num_s = round_up_to_multiple(int(np.ceil(num_s_raw - 1e-6)), 16)
    num_s = max(16, num_s)

    full_max = full_min + full_num
    while (min_s + num_s) / scale < full_max:
        num_s += 16

    return min_s, num_s


# =========================
# SGBM & ximgproc
# =========================
def create_sgbm(min_disp, num_disp):
    bs = int(BLOCK_SIZE)
    P1 = 8 * bs * bs
    P2 = 32 * bs * bs
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disp),
        numDisparities=int(num_disp),
        blockSize=bs,
        P1=P1, P2=P2,
        disp12MaxDiff=int(DISP12_MAXDIFF),
        uniquenessRatio=int(UNIQUENESS),
        speckleWindowSize=int(SPECKLE_WINSIZE),
        speckleRange=int(SPECKLE_RANGE),
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def has_ximgproc():
    return (
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "createRightMatcher")
        and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
    )


def apply_speckle_removal_q4(disp_q4):
    if not ENABLE_SPECKLE_REMOVE:
        return disp_q4
    out = disp_q4.copy()
    max_size = int(SPECKLE_MAX_SIZE)
    diff_q4 = int(SPECKLE_DIFF_PX) * 16
    cv2.filterSpeckles(out, 0, max_size, diff_q4)
    return out


def build_wls_filter(left_matcher, lrc_thresh_px, lam, sigma_color, disc_radius):
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls.setLambda(float(lam))
    wls.setSigmaColor(float(sigma_color))
    wls.setLRCthresh(int(lrc_thresh_px))
    wls.setDepthDiscontinuityRadius(int(disc_radius))
    return wls


# =========================
# 相机采集
# =========================
def capture_one_with_ts(cam):
    frame = cam.capture_array()  # RGB888 -> RGB
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


# =========================
# ROI 裁剪
# =========================
def roi_intersection(roi1, roi2):
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
    have = ("validPixROI1" in params.files) and ("validPixROI2" in params.files)
    if have:
        roi1 = params["validPixROI1"].reshape(-1)
        roi2 = params["validPixROI2"].reshape(-1)
        inter = roi_intersection(roi1, roi2)
        if inter is None:
            x, y, w, h = 0, 0, full_w, full_h
        else:
            x, y, w, h = inter
    else:
        x, y, w, h = 0, 0, full_w, full_h

    if ROI_BORDER_RATIO > 0:
        dx = int(w * ROI_BORDER_RATIO)
        dy = int(h * ROI_BORDER_RATIO)
        x = x + dx
        y = y + dy
        w = max(1, w - 2 * dx)
        h = max(1, h - 2 * dy)

    x = max(0, min(x, full_w - 1))
    y = max(0, min(y, full_h - 1))
    w = max(1, min(w, full_w - x))
    h = max(1, min(h, full_h - y))
    return (x, y, w, h)


def rectify_and_crop_common(rgbL, rgbR, map1_x, map1_y, map2_x, map2_y, crop_xywh):
    rectL = cv2.remap(rgbL, map1_x, map1_y, cv2.INTER_LINEAR)
    rectR = cv2.remap(rgbR, map2_x, map2_y, cv2.INTER_LINEAR)
    x, y, w, h = crop_xywh
    rectL = rectL[y:y + h, x:x + w]
    rectR = rectR[y:y + h, x:x + w]
    return rectL, rectR


# =========================
# 视差转换与显示（按你的要求：不做 RGB<->BGR cvtColor）
# =========================
def q4_scaled_to_full_disp_and_valid(disp_q4_scaled, full_shape_hw, max_disp_full, scale):
    disp_scaled = disp_q4_scaled.astype(np.float32) / 16.0

    H, W = full_shape_hw
    if abs(scale - 1.0) < 1e-6:
        disp_full = disp_scaled
    else:
        disp_full = cv2.resize(disp_scaled, (W, H), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)

    valid = (disp_full > float(MIN_DISP_FOR_DEPTH_FULL)) & (disp_full < float(max_disp_full - 1))
    disp_full[~valid] = 0.0
    return disp_full, valid


def disp_to_colormap_fast(disp_full, valid, max_disp_full):
    if valid is None or np.count_nonzero(valid) == 0:
        h, w = disp_full.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    maxd = float(max_disp_full) if DISP_VIS_MAX_FULL is None else float(DISP_VIS_MAX_FULL)
    maxd = max(1.0, maxd)
    vis = np.clip(disp_full / maxd * 255.0, 0, 255).astype(np.uint8)
    vis[~valid] = 0

    # 关键：overlay 颜色你说原本正确 -> 这里不做任何通道翻转
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def overlay_depth_on_left(rectL_rgb, disp_color):
    if disp_color is None:
        return rectL_rgb
    return cv2.addWeighted(rectL_rgb, 1.0 - OVERLAY_ALPHA, disp_color, OVERLAY_ALPHA, 0.0)


def resize_for_display(img, scale):
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# =========================
# 配对
# =========================
def find_best_pair_latest(Ls, Rs):
    if not Ls or not Rs:
        return None, None
    idL, rgbL, tsL = Ls[-1]
    best = None
    best_dt = 1e18
    for (idR, rgbR, tsR) in Rs:
        dt_ms = abs(int(tsL) - int(tsR)) / 1e6
        if dt_ms < best_dt:
            best_dt = dt_ms
            best = (idL, rgbL, tsL, idR, rgbR, tsR)
    return best, best_dt


# =========================
# 测距（Z = Znew + Zcm）
# =========================
def measure_depth_at(disp_full, valid, x, y, fx, baseline_m, win=11, min_valid=20):
    h, w = disp_full.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, {"reason": "out_of_range"}

    r = win // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    d_patch = disp_full[y0:y1, x0:x1]
    v_patch = valid[y0:y1, x0:x1]
    vals = d_patch[v_patch]

    if vals.size < min_valid:
        return None, {"reason": "too_few_valid", "valid": int(vals.size)}

    d_med = float(np.median(vals))
    if d_med < float(MIN_DISP_FOR_DEPTH_FULL):
        return None, {"reason": "disp_too_small", "d_med": d_med, "valid": int(vals.size)}

    z_m = (fx * baseline_m) / d_med
    z_cm = z_m * 100.0
    return float(z_cm), {"d_med": d_med, "valid": int(vals.size), "win": win}


def z_correction_poly(z_cm: float) -> float:
    return 0.001 * (z_cm ** 2) - 0.0215 * z_cm + 0.0889


def get_fx_baseline_from_params(params):
    if "P1" in params.files:
        fx = float(params["P1"][0, 0])
    else:
        fx = float(params["cameraMatrix1"][0, 0])

    baseline = None
    if "Q" in params.files:
        Q = params["Q"].astype(np.float64)
        if abs(Q[3, 2]) > 1e-12:
            baseline = abs(1.0 / Q[3, 2])
    if baseline is None and "T" in params.files:
        T = params["T"].reshape(-1).astype(np.float64)
        baseline = float(abs(T[0]))
    if baseline is None:
        baseline = 0.06

    baseline_m = baseline / 1000.0 if baseline > 1.0 else baseline
    print(f"[INFO] fx={fx:.2f}px  baseline_m={baseline_m:.5f} m")
    return fx, baseline_m


# =========================
# UI 按钮
# =========================
def draw_button(img, rect, label, active=False):
    x0, y0, x1, y1 = rect
    overlay = img.copy()
    bg = (70, 70, 70) if not active else (130, 130, 130)
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


def compute_button_rects(frame_w, frame_h):
    rects = {}

    xr1 = frame_w - BTN_PAD
    xr0 = xr1 - BTN_W
    y = BTN_PAD
    rects["VIEW_OVERLAY"] = (xr0, y, xr1, y + BTN_H)
    y += BTN_H + BTN_GAP
    rects["VIEW_PAIR"] = (xr0, y, xr1, y + BTN_H)
    y += BTN_H + BTN_GAP
    rects["VIEW_DISP"] = (xr0, y, xr1, y + BTN_H)
    y += BTN_H + BTN_GAP
    rects["CLEAR"] = (xr0, y, xr1, y + BTN_H)

    xl0 = BTN_PAD
    xl1 = xl0 + BTN_W
    yb3 = frame_h - BTN_PAD - BTN_H
    yb2 = yb3 - (BTN_H + BTN_GAP)
    yb1 = yb2 - (BTN_H + BTN_GAP)
    rects["POST_NONE"] = (xl0, yb1, xl1, yb1 + BTN_H)
    rects["POST_LRC"] = (xl0, yb2, xl1, yb2 + BTN_H)
    rects["POST_WLS"] = (xl0, yb3, xl1, yb3 + BTN_H)

    return rects


# =========================
# 主程序
# =========================
def main():
    min_s, num_s = compute_scaled_disp_params(
        MIN_DISPARITIES_FULL_TARGET,
        NUM_DISPARITIES_FULL_TARGET,
        DISP_SCALE
    )
    full_max = MIN_DISPARITIES_FULL_TARGET + NUM_DISPARITIES_FULL_TARGET
    scaled_max_back = (min_s + num_s) / (DISP_SCALE if DISP_SCALE != 0 else 1.0)
    print(f"[INFO] DISP_SCALE={DISP_SCALE} fullDisp=[{MIN_DISPARITIES_FULL_TARGET},{full_max}) "
          f"-> scaledDisp=[{min_s},{min_s + num_s}) scaledMaxBack≈{scaled_max_back:.1f}")

    params = np.load(CALIB_NPZ)
    image_size = tuple(params["image_size"])
    full_w, full_h = int(image_size[0]), int(image_size[1])
    print("[INFO] calib:", CALIB_NPZ, "image_size:", (full_w, full_h))

    map1_x, map1_y = params["map1_x"].astype(np.float32), params["map1_y"].astype(np.float32)
    map2_x, map2_y = params["map2_x"].astype(np.float32), params["map2_y"].astype(np.float32)

    fx, baseline_m = get_fx_baseline_from_params(params)
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
    camL.configure(cfgL)
    camR.configure(cfgR)
    camL.start()
    camR.start()

    if USE_AE_AWB_CONVERGE_THEN_LOCK:
        converge_then_lock(camL, camR, settle_s=SETTLE_S, throw_n=THROW_N)
    else:
        throw_frames(camL, camR, n=30)

    lock = threading.Lock()
    stop = threading.Event()

    rawL = deque(maxlen=BUF_LEN)
    rawR = deque(maxlen=BUF_LEN)
    ids = {"L": 0, "R": 0}

    view_mode = {"val": 0}  # 0=OVERLAY 1=PAIR 2=DISP
    view_names = ["OVERLAY", "PAIR", "DISP"]
    post_mode = {"val": POSTPROC_DEFAULT.upper()}

    shared = {
        "rectL": None, "rectR": None,
        "disp_full": None, "valid": None,
        "disp_color": None,
        "dt_ms": None,
        "comp_fps": 0.0,
        "comp_ok": False,
        "last_click": None,
        "pair_key": None,
        "capL_fps": 0.0,
        "capR_fps": 0.0,
        "post_used": post_mode["val"],
    }

    # 采集线程
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

    threading.Thread(target=cap_thread, args=(camL, "L"), daemon=True).start()
    threading.Thread(target=cap_thread, args=(camR, "R"), daemon=True).start()

    # 计算线程
    def comp_thread():
        comp_fps = 0.0
        t_prev_new = None
        last_pair_key = None
        last_post = None

        left_matcher = None
        right_matcher = None
        wls = None
        use_xi = has_ximgproc()

        def rebuild(post):
            nonlocal left_matcher, right_matcher, wls, last_pair_key
            post = post.upper()
            left_matcher = create_sgbm(min_s, num_s)

            right_matcher = None
            wls = None
            if post != "NONE" and use_xi:
                right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
                if post == "LRC":
                    wls = build_wls_filter(
                        left_matcher,
                        lrc_thresh_px=WLS_LRC_THRESH,
                        lam=LRC_ONLY_LAMBDA,
                        sigma_color=WLS_SIGMA_COLOR,
                        disc_radius=WLS_DISC_RADIUS,
                    )
                else:
                    wls = build_wls_filter(
                        left_matcher,
                        lrc_thresh_px=WLS_LRC_THRESH,
                        lam=WLS_LAMBDA,
                        sigma_color=WLS_SIGMA_COLOR,
                        disc_radius=WLS_DISC_RADIUS,
                    )
            last_pair_key = None

        t_next = time.perf_counter()
        rebuild(post_mode["val"])
        last_post = post_mode["val"]

        while not stop.is_set():
            now = time.perf_counter()
            sleep_s = t_next - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += COMP_PERIOD_S

            with lock:
                Ls = list(rawL)
                Rs = list(rawR)
                post = post_mode["val"]

            if not Ls or not Rs:
                continue

            if post != last_post:
                rebuild(post)
                last_post = post
                with lock:
                    shared["last_click"] = None
                    shared["post_used"] = post

            best, dt_ms = find_best_pair_latest(Ls, Rs)
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

            if abs(DISP_SCALE - 1.0) < 1e-6:
                gL_s, gR_s = gL, gR
                guide_s = rectL
            else:
                gL_s = cv2.resize(gL, None, fx=DISP_SCALE, fy=DISP_SCALE, interpolation=cv2.INTER_AREA)
                gR_s = cv2.resize(gR, None, fx=DISP_SCALE, fy=DISP_SCALE, interpolation=cv2.INTER_AREA)
                guide_s = cv2.resize(rectL, (gL_s.shape[1], gL_s.shape[0]), interpolation=cv2.INTER_AREA)

            dispL_q4_s = left_matcher.compute(gL_s, gR_s)

            if wls is not None and right_matcher is not None:
                dispR_q4_s = right_matcher.compute(gR_s, gL_s)
                dispF_q4_s = wls.filter(dispL_q4_s, guide_s, None, dispR_q4_s)
                dispF_q4_s = apply_speckle_removal_q4(dispF_q4_s)
                disp_q4_s = dispF_q4_s
            else:
                disp_q4_s = dispL_q4_s

            max_disp_full = MIN_DISPARITIES_FULL_TARGET + NUM_DISPARITIES_FULL_TARGET
            disp_full, valid = q4_scaled_to_full_disp_and_valid(
                disp_q4_s, (gL.shape[0], gL.shape[1]), max_disp_full=max_disp_full, scale=DISP_SCALE
            )
            disp_color = disp_to_colormap_fast(disp_full, valid, max_disp_full=max_disp_full)

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
                shared["disp_full"] = disp_full
                shared["valid"] = valid
                shared["disp_color"] = disp_color
                shared["dt_ms"] = dt_ms
                shared["comp_fps"] = comp_fps
                shared["comp_ok"] = True
                shared["pair_key"] = pair_key
                shared["post_used"] = post

    threading.Thread(target=comp_thread, daemon=True).start()

    # 鼠标交互
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if DISPLAY_SCALE < 0.999:
            x0 = int(x / DISPLAY_SCALE)
            y0 = int(y / DISPLAY_SCALE)
        else:
            x0, y0 = x, y

        with lock:
            vmode = view_mode["val"]
            rectL = shared["rectL"]
            rectR = shared["rectR"]
            disp_full = shared["disp_full"]
            valid = shared["valid"]

        if rectL is None:
            return

        if vmode == 1 and rectR is not None:
            frame_w = rectL.shape[1] * 2
            frame_h = rectL.shape[0]
        else:
            frame_w = rectL.shape[1]
            frame_h = rectL.shape[0]

        rects = compute_button_rects(frame_w, frame_h)

        for name, r in rects.items():
            if point_in_rect(x0, y0, r):
                with lock:
                    if name == "VIEW_OVERLAY":
                        view_mode["val"] = 0
                        shared["last_click"] = None
                    elif name == "VIEW_PAIR":
                        view_mode["val"] = 1
                        shared["last_click"] = None
                    elif name == "VIEW_DISP":
                        view_mode["val"] = 2
                        shared["last_click"] = None
                    elif name == "CLEAR":
                        shared["last_click"] = None
                    elif name == "POST_NONE":
                        post_mode["val"] = "NONE"
                        shared["last_click"] = None
                    elif name == "POST_LRC":
                        post_mode["val"] = "LRC"
                        shared["last_click"] = None
                    elif name == "POST_WLS":
                        post_mode["val"] = "WLS"
                        shared["last_click"] = None
                return

        if disp_full is None or valid is None:
            return

        h, w = rectL.shape[:2]
        if vmode == 1:
            if x0 >= w:
                return
            x_use, y_use = x0, y0
        else:
            x_use, y_use = x0, y0

        z_cm, info = measure_depth_at(
            disp_full, valid, x_use, y_use, fx, baseline_m,
            win=MEASURE_WIN, min_valid=MIN_VALID_PIXELS
        )

        z_new = None
        z_sum = None
        if z_cm is not None:
            z_new = z_correction_poly(float(z_cm))
            z_sum = float(z_cm) + float(z_new)

        with lock:
            shared["last_click"] = (x_use, y_use, z_cm, z_new, z_sum, info)

    cv2.namedWindow("stereo_rt_preview", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("stereo_rt_preview", on_mouse)

    ui_fps = 0.0
    t_ui_prev = None
    t_next_ui = time.perf_counter()

    try:
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k in [27, ord("q")]:
                break

            now = time.perf_counter()
            sleep_s = t_next_ui - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next_ui += UI_PERIOD_S

            with lock:
                vmode = view_mode["val"]
                rectL = shared["rectL"]
                rectR = shared["rectR"]
                disp_color = shared["disp_color"]
                dt_ms = shared["dt_ms"]
                comp_fps = shared["comp_fps"]
                comp_ok = shared["comp_ok"]
                last_click = shared["last_click"]
                pair_key = shared["pair_key"]
                capL_fps = shared["capL_fps"]
                capR_fps = shared["capR_fps"]
                post_used = shared["post_used"]
                post_now = post_mode["val"]

            if rectL is None:
                frame = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            else:
                if vmode == 0:
                    frame = overlay_depth_on_left(rectL, disp_color)
                elif vmode == 1:
                    if rectR is None:
                        frame = rectL
                    else:
                        frame = np.hstack([rectL, rectR])
                        if DRAW_DEBUG_LINES:
                            hh, ww = frame.shape[:2]
                            for f in LINES_Y_FRAC:
                                yy = int(hh * f)
                                cv2.line(frame, (0, yy), (ww - 1, yy), (0, 255, 0), 1)
                else:
                    if disp_color is not None:
                        frame = disp_color.copy()
                    else:
                        frame = np.zeros((rectL.shape[0], rectL.shape[1], 3), dtype=np.uint8)

            nowu = time.perf_counter()
            if t_ui_prev is None:
                t_ui_prev = nowu
            else:
                dtu = nowu - t_ui_prev
                t_ui_prev = nowu
                if dtu > 1e-6:
                    inst = 1.0 / dtu
                    ui_fps = (1.0 - FPS_EMA_ALPHA) * ui_fps + FPS_EMA_ALPHA * inst

            s_dt = "N/A" if dt_ms is None else f"{dt_ms:.2f}ms"
            s_new = "NEW" if comp_ok else "HOLD"

            line0 = f"CAP_L={capL_fps:.1f}  CAP_R={capR_fps:.1f}"
            line1 = f"{view_names[vmode]}  UI={ui_fps:.1f}  COMP={comp_fps:.1f}  dt={s_dt}  {s_new}"
            line2 = (f"POST={post_used} (req:{post_now})  dispScale={DISP_SCALE:.2f}  "
                     f"fullDisp=[{MIN_DISPARITIES_FULL_TARGET},{MIN_DISPARITIES_FULL_TARGET + NUM_DISPARITIES_FULL_TARGET})")
            line3 = f"scaledDisp=[{min_s},{min_s + num_s})  pair={pair_key if pair_key is not None else 'N/A'}"

            cv2.putText(frame, line0, (10, 20), HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICK, cv2.LINE_AA)
            cv2.putText(frame, line1, (10, 42), HUD_FONT, HUD_SCALE_BIG, (255, 255, 255), HUD_THICK, cv2.LINE_AA)
            cv2.putText(frame, line2, (10, 64), HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICK, cv2.LINE_AA)
            cv2.putText(frame, line3, (10, 86), HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICK, cv2.LINE_AA)

            if last_click is not None and vmode != 1:
                x, y, z_cm, z_new, z_sum, info = last_click
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)

                if z_sum is None:
                    txt1 = f"({x},{y}) N/A  info={info}"
                    txt2 = ""
                else:
                    txt1 = f"({x},{y}) d_med={info['d_med']:.3f}px  Z={z_sum:.1f}cm  valid={info['valid']}"
                    txt2 = f"(Znew+Z_cm) = {z_new:.2f} + {z_cm:.2f}"

                y_top = 92
                y_bot = 140
                cv2.rectangle(frame, (10, y_top), (min(frame.shape[1] - 1, 1240), y_bot), (0, 0, 0), -1)
                cv2.putText(frame, txt1, (15, 114), HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICK, cv2.LINE_AA)
                if txt2:
                    cv2.putText(frame, txt2, (15, 136), HUD_FONT, HUD_SCALE_SMALL, (255, 255, 255), HUD_THICK, cv2.LINE_AA)

            rects = compute_button_rects(frame.shape[1], frame.shape[0])

            draw_button(frame, rects["VIEW_OVERLAY"], "OVERLAY", active=(vmode == 0))
            draw_button(frame, rects["VIEW_PAIR"], "PAIR", active=(vmode == 1))
            draw_button(frame, rects["VIEW_DISP"], "DISP", active=(vmode == 2))
            draw_button(frame, rects["CLEAR"], "CLEAR", active=False)

            draw_button(frame, rects["POST_NONE"], "POST:NONE", active=(post_now == "NONE"))
            draw_button(frame, rects["POST_LRC"], "POST:LRC", active=(post_now == "LRC"))
            draw_button(frame, rects["POST_WLS"], "POST:WLS", active=(post_now == "WLS"))

            show = resize_for_display(frame, DISPLAY_SCALE)
            cv2.imshow("stereo_rt_preview", show)

    finally:
        stop.set()
        time.sleep(0.1)
        camL.stop()
        camR.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
