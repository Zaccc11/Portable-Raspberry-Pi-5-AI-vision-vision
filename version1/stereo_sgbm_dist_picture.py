# stereo_sgbm_main_B.py
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# =========================
# Tunables (只改这里)
# =========================
FRAME_RATE = 60

# AE/AWB 收敛后锁曝光/增益
USE_AE_AWB_CONVERGE_THEN_LOCK = True
SETTLE_S = 1.2
THROW_N = 25

# 采样多次，选 dt 最小的一对
N_TRIES = 20
SLEEP_BETWEEN = 0.01
PRINT_EACH_DT = True

# Rectify ROI crop
ROI_BORDER_RATIO = 0.06

# SGBM params
MIN_DISPARITIES = 0
#NUM_DISPARITIES = 256    # 必须 16 的倍数
NUM_DISPARITIES = 640    # 必须 16 的倍数
BLOCK_SIZE = 7           # 5/7/9
UNIQUENESS = 10
SPECKLE_WINSIZE = 200    # SGBM 自带 speckle
SPECKLE_RANGE = 4
DISP12_MAXDIFF = 1

# ========= 后处理模式开关 =========
# "NONE": 只输出 raw disparity（disp_raw_*）
# "LRC" : 只做 WLS 内置 LRC gating + speckle removal（lambda=0，几乎不平滑）
# "WLS" : WLS(edge-aware) + 内置 LRC + speckle removal
POSTPROC_MODE = "WLS"     # "NONE" / "LRC" / "WLS"

# ========= Left-Right Check (OpenCV WLS 内置) =========
# 单位：像素。越小越严格。dt ~7ms 时，2 经常会过严 -> 大面积无效
#WLS_LRC_THRESH = 255
WLS_LRC_THRESH = 4
# ========= WLS =========
WLS_LAMBDA = 600            # 建议 800~1500 起步
WLS_SIGMA_COLOR = 0.6     # 建议 0.6~1.0 起步
WLS_DISC_RADIUS = 2       # 0 或 1

# LRC-only 模式下 lambda（0 = 不做平滑）
LRC_ONLY_LAMBDA = 0

# ========= Speckle removal (OpenCV filterSpeckles on Q4) =========
ENABLE_SPECKLE_REMOVE = True
SPECKLE_MAX_SIZE = 200        # 50~400
SPECKLE_DIFF_PX = 2           # 1~3 (像素)

# Debug
SAVE_DEBUG_MASKS = True

# 水平线可视化间隔
HLINE_STEP = 80

# ========= Depth / Click-to-measure =========
BASELINE_M = 0.060        # 60mm (仅做兜底；默认会用标定结果覆盖)
MEASURE_WIN = 11          # 点击测距用的窗口（奇数 7/9/11）
MIN_VALID_PIXELS = 20     # 窗口内至少多少有效像素才输出
MIN_DISP_FOR_DEPTH = 0.5  # 视差太小就不算（会接近无限远）
ENABLE_GUI_CLICK_MEASURE = True  # 打开 cv2.imshow + 鼠标点击测距

# ========= Pre-filter / Texture enhance switches =========
# 先增强纹理再匹配，通常比只调 SGBM 参数有效得多（尤其皮肤/弱纹理）
ENABLE_GAUSSIAN_BLUR = False     # True: 先高斯降噪再做 SGBM（有时会抹纹理）
GAUSS_KSIZE = 3                 # 3/5（必须奇数）
GAUSS_SIGMA = 0                 # 0=OpenCV自动；也可设 0.8/1.2

ENABLE_TEXTURE_ENHANCE = True    # 纹理增强总开关（建议先开）
TEXTURE_MODE = "NONE"     # "CLAHE" / "SHARP" / "CLAHE_SHARP" / "NONE"

# --- CLAHE（局部对比度增强，最常用也最稳）---
CLAHE_CLIPLIMIT = 2.0           # 1.5~3.0
CLAHE_TILE = 8                  # 6/8/10/12

# --- Unsharp mask（轻锐化，提升边缘纹理；过强会放大噪声）---
SHARP_AMOUNT = 0.7              # 0.3~1.2
SHARP_BLUR_K = 5                # 3/5/7（奇数）
# =========================


def has_ximgproc():
    return (
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "createRightMatcher")
        and hasattr(cv2.ximgproc, "createDisparityWLSFilter")
    )


def create_sgbm():
    # P1/P2：控制视差平滑惩罚（越大越“更像平面”，越小越“更允许跳变/噪声”）
    P1 = 8 * BLOCK_SIZE * BLOCK_SIZE
    P2 = 32 * BLOCK_SIZE * BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITIES,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1,
        P2=P2,
        disp12MaxDiff=DISP12_MAXDIFF,
        uniquenessRatio=UNIQUENESS,
        speckleWindowSize=SPECKLE_WINSIZE,
        speckleRange=SPECKLE_RANGE,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


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
    mR = camR.capture_metadata()
    exp = int(mL.get("ExposureTime", 8000))
    gain = float(mL.get("AnalogueGain", 2.0))

    print("[AUTO] before lock:")
    print("  L Exp/Gain:", mL.get("ExposureTime"), mL.get("AnalogueGain"))
    print("  R Exp/Gain:", mR.get("ExposureTime"), mR.get("AnalogueGain"))

    # 注意：这里用“同一个 exp/gain”强行锁两路，通常对匹配更友好（亮度一致）
    controls = {
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": exp,
        "AnalogueGain": gain,
    }
    camL.set_controls(controls)
    camR.set_controls(controls)
    throw_frames(camL, camR, n=20)

    mL2 = camL.capture_metadata()
    mR2 = camR.capture_metadata()
    print("[LOCK] L Exp/Gain:", mL2.get("ExposureTime"), mL2.get("AnalogueGain"),
          "| R Exp/Gain:", mR2.get("ExposureTime"), mR2.get("AnalogueGain"))
    return exp, gain


def capture_one_with_ts(cam):
    frame = cam.capture_array()
    meta = cam.capture_metadata()
    ts = meta.get("SensorTimestamp", time.time_ns())
    return frame, ts


def pick_best_pair(camL, camR, n_tries=20, sleep_between=0.01):
    dts = []
    best = None  # (dt_ns, L, R, tsL, tsR)

    for i in range(n_tries):
        L, tsL = capture_one_with_ts(camL)
        R, tsR = capture_one_with_ts(camR)
        dt = abs(int(tsL) - int(tsR))
        dts.append(dt)

        if PRINT_EACH_DT:
            print(f"[PAIR] try {i+1:02d}/{n_tries} dt = {dt/1e6:.3f} ms")

        if best is None or dt < best[0]:
            best = (dt, L, R, tsL, tsR)

        time.sleep(sleep_between)

    dts_ms = np.array(dts, dtype=np.float64) / 1e6
    stats = {
        "min_ms": float(np.min(dts_ms)),
        "mean_ms": float(np.mean(dts_ms)),
        "p50_ms": float(np.percentile(dts_ms, 50)),
        "p90_ms": float(np.percentile(dts_ms, 90)),
        "max_ms": float(np.max(dts_ms)),
    }
    return best, dts_ms, stats


def rectify_and_crop(rgbL, rgbR, map1_x, map1_y, map2_x, map2_y):
    # 这里必须用 cv2.INTER_LINEAR（你之前报错 INTER_LINEARHOST_LINEA 是拼错常量）
    rectL = cv2.remap(rgbL, map1_x, map1_y, cv2.INTER_LINEAR)
    rectR = cv2.remap(rgbR, map2_x, map2_y, cv2.INTER_LINEAR)

    h, w = rectL.shape[:2]
    x0 = int(w * ROI_BORDER_RATIO); x1 = int(w * (1.0 - ROI_BORDER_RATIO))
    y0 = int(h * ROI_BORDER_RATIO); y1 = int(h * (1.0 - ROI_BORDER_RATIO))
    rectL = rectL[y0:y1, x0:x1]
    rectR = rectR[y0:y1, x0:x1]
    return rectL, rectR


def save_pair_with_hlines(rectL, rectR, out_path, step=80):
    pair = np.hstack([rectL, rectR])
    h, w = rectL.shape[:2]
    pair_lines = pair.copy()
    for y in range(step, h, step):
        cv2.line(pair_lines, (0, y), (2 * w - 1, y), (0, 255, 0), 1)
    cv2.imwrite(out_path, pair_lines)


def apply_texture_enhance(gray_u8):
    """
    输入 uint8 灰度图，输出增强后的 uint8 灰度图
    """
    if (not ENABLE_TEXTURE_ENHANCE) or (TEXTURE_MODE.upper() == "NONE"):
        return gray_u8

    out = gray_u8
    mode = TEXTURE_MODE.upper()

    if "CLAHE" in mode:
        clahe = cv2.createCLAHE(
            clipLimit=float(CLAHE_CLIPLIMIT),
            tileGridSize=(int(CLAHE_TILE), int(CLAHE_TILE))
        )
        out = clahe.apply(out)

    if "SHARP" in mode:
        k = int(SHARP_BLUR_K)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(out, (k, k), 0)
        out_f = out.astype(np.float32)
        blur_f = blur.astype(np.float32)
        out_f = out_f + float(SHARP_AMOUNT) * (out_f - blur_f)
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    return out


def apply_prefilter(gray_u8):
    """
    先纹理增强，再可选高斯降噪（顺序：增强->轻微降噪）
    """
    out = apply_texture_enhance(gray_u8)

    if ENABLE_GAUSSIAN_BLUR:
        k = int(GAUSS_KSIZE)
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), float(GAUSS_SIGMA))

    return out


def disp_q4_to_float_and_valid(disp_q4):
    """
    Q4 int16 -> float32 disparity, 同时给出“真正有效”的 mask
    对 minDisparity=0，OpenCV 常用 invalid = -16 (Q4)
    disparity=0 在数学上是合法值（远处/超范围/平面等），不要用 disp>0 判定有效！
    """
    invalid_q4 = MIN_DISPARITIES * 16 - 16   # minDisp=0 -> -16
    valid = disp_q4 > invalid_q4

    disp = disp_q4.astype(np.float32) / 16.0
    disp[~valid] = 0.0
    disp = np.clip(disp, 0.0, float(NUM_DISPARITIES - 1))
    return disp, valid


def apply_speckle_removal_q4(disp_q4):
    if not ENABLE_SPECKLE_REMOVE:
        return disp_q4
    disp_q4 = disp_q4.copy()
    invalid_q4 = MIN_DISPARITIES * 16 - 16   # minDisp=0 -> -16
    max_size = int(SPECKLE_MAX_SIZE)
    diff_q4 = int(SPECKLE_DIFF_PX) * 16
    cv2.filterSpeckles(disp_q4, invalid_q4, max_size, diff_q4)
    return disp_q4


def build_wls_filter(left_matcher, lrc_thresh_px, lam, sigma_color, disc_radius):
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls.setLambda(float(lam))
    wls.setSigmaColor(float(sigma_color))
    wls.setLRCthresh(int(lrc_thresh_px))
    wls.setDepthDiscontinuityRadius(int(disc_radius))
    return wls


def compute_disp_pipeline(rectL_rgb, rectR_rgb):
    """
    输出：
      disp_raw(float32), raw_valid(bool)
      disp_post(float32 or None), post_valid(bool or None)
      tag
    """
    gL = cv2.cvtColor(rectL_rgb, cv2.COLOR_RGB2GRAY)
    gR = cv2.cvtColor(rectR_rgb, cv2.COLOR_RGB2GRAY)

    # === NEW: prefilter + texture enhance (switchable) ===
    gL = apply_prefilter(gL)
    gR = apply_prefilter(gR)

    left_matcher = create_sgbm()
    dispL_q4 = left_matcher.compute(gL, gR)

    disp_raw, raw_valid = disp_q4_to_float_and_valid(dispL_q4)

    mode = POSTPROC_MODE.upper()
    if mode == "NONE":
        return disp_raw, raw_valid, None, None, "POST_NONE"

    if not has_ximgproc():
        print("[WARN] cv2.ximgproc not available -> fallback to raw only")
        return disp_raw, raw_valid, None, None, "NO_XIMGPROC"

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    dispR_q4 = right_matcher.compute(gR, gL)

    # guide：rectified 彩色左图（BGR, uint8）
    guide_bgr = cv2.cvtColor(rectL_rgb, cv2.COLOR_RGB2BGR)

    if mode == "LRC":
        wls = build_wls_filter(
            left_matcher,
            lrc_thresh_px=WLS_LRC_THRESH,
            lam=LRC_ONLY_LAMBDA,
            sigma_color=WLS_SIGMA_COLOR,
            disc_radius=WLS_DISC_RADIUS,
        )
        dispF_q4 = wls.filter(dispL_q4, guide_bgr, None, dispR_q4)
        dispF_q4 = apply_speckle_removal_q4(dispF_q4)

        disp_post, post_valid = disp_q4_to_float_and_valid(dispF_q4)
        # === NEW ===
        post_valid = dispF_q4 > int(MIN_DISP_FOR_DEPTH * 16)
        disp_post[~post_valid] = 0.0

        valid_ratio = float(np.count_nonzero(post_valid)) / float(post_valid.size)
        print(f"[POST] LRC-only(valid_ratio={valid_ratio*100:.1f}%) "
              f"(LRCthresh={WLS_LRC_THRESH}px, lambda={LRC_ONLY_LAMBDA})")
        return disp_raw, raw_valid, disp_post, post_valid, "POST_LRC_ONLY"

    # mode == "WLS"
    wls = build_wls_filter(
        left_matcher,
        lrc_thresh_px=WLS_LRC_THRESH,
        lam=WLS_LAMBDA,
        sigma_color=WLS_SIGMA_COLOR,
        disc_radius=WLS_DISC_RADIUS,
    )
    dispF_q4 = wls.filter(dispL_q4, guide_bgr, None, dispR_q4)
    dispF_q4 = apply_speckle_removal_q4(dispF_q4)

    disp_post, post_valid = disp_q4_to_float_and_valid(dispF_q4)
    # === NEW: 对 post 的 valid 更严格，避免 0/极小视差污染统计与显示 ===
    post_valid = dispF_q4 > int(MIN_DISP_FOR_DEPTH * 16)
    disp_post[~post_valid] = 0.0
    valid_ratio = float(np.count_nonzero(post_valid)) / float(post_valid.size)
    print(f"[POST] WLS(valid_ratio={valid_ratio*100:.1f}%) "
        f"(LRCthresh={WLS_LRC_THRESH}px, lambda={WLS_LAMBDA}, sigma={WLS_SIGMA_COLOR})")
    return disp_raw, raw_valid, disp_post, post_valid, "POST_WLS"


def disp_to_vis(disp, valid_mask):
    """
    用 valid_mask 做统计和显示；不要用 disp>0
    """
    if valid_mask is None or np.count_nonzero(valid_mask) == 0:
        h, w = disp.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)

    vmin = float(np.percentile(disp[valid_mask], 5))
    vmax = float(np.percentile(disp[valid_mask], 95))
    if vmax - vmin < 1e-3:
        vmax = vmin + 1.0

    vis = np.clip((disp - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    vis[~valid_mask] = 0
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis, color


def save_mask_png(mask, path):
    if mask is None:
        return
    img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(path, img)


def measure_depth_at(disp, valid, x, y, fx, baseline_m, win=11, min_valid=20):
    """
    在 (x,y) 附近 win*win 窗口取有效视差中位数 -> 深度Z(米)
    x,y 是 ROI 图（rectL/disp图）坐标系
    """
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

    z = (fx * baseline_m) / d_med
    return float(z), {"d_med": d_med, "valid": int(vals.size), "win": win}


# ======= NEW: baseline from stereo_params.npz =======
def baseline_from_params(params):
    """
    从 stereo_params.npz 里推 baseline（单位：米）
    优先用 T（标定单位通常是 mm），其次用 P2 推 Tx。
    """
    if "T" in params.files:
        T = np.array(params["T"]).reshape(-1)
        B_mm = float(np.linalg.norm(T))
        return B_mm / 1000.0, "T_norm(mm)"

    if "P2" in params.files:
        P2 = np.array(params["P2"], dtype=np.float64)
        fx = float(P2[0, 0])
        Tx = float(P2[0, 3])
        if abs(fx) > 1e-9:
            B_mm = abs(Tx / fx)
            return B_mm / 1000.0, "P2_tx_over_fx(mm)"

    return None, "NONE"


def main():
    params = np.load("stereo_params.npz")
    image_size = tuple(params["image_size"])
    map1_x, map1_y = params["map1_x"], params["map1_y"]
    map2_x, map2_y = params["map2_x"], params["map2_y"]

    print("[INFO] npz keys:", params.files)

    # === 从 npz 里取 fx（像素） ===
    if "P1" in params.files:
        fx = float(params["P1"][0, 0])
        fx_src = "P1"
    else:
        fx = float(params["cameraMatrix1"][0, 0])
        fx_src = "cameraMatrix1"
    print(f"[INFO] fx(px) = {fx:.3f}  (from {fx_src})")

    # === baseline from calibration ===
    baseline_m, b_src = baseline_from_params(params)
    if baseline_m is None:
        baseline_m = BASELINE_M  # 兜底：还用你写死的 0.060
        b_src = "FALLBACK_CONST"
    print(f"[INFO] baseline(m) = {baseline_m:.6f} (from {b_src})")

    width, height = int(image_size[0]), int(image_size[1])
    print("[INFO] image_size:", image_size)
    print("[INFO] POSTPROC_MODE:", POSTPROC_MODE)
    print("[INFO] prefilter:",
          f"TEXTURE_EN={ENABLE_TEXTURE_ENHANCE}({TEXTURE_MODE}), "
          f"GAUSS_EN={ENABLE_GAUSSIAN_BLUR}(k={GAUSS_KSIZE},sigma={GAUSS_SIGMA})")

    camL = Picamera2(1)
    camR = Picamera2(0)

    cfgL = camL.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE},
    )
    cfgR = camR.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE},
    )
    camL.configure(cfgL)
    camR.configure(cfgR)

    camL.start()
    camR.start()

    if USE_AE_AWB_CONVERGE_THEN_LOCK:
        converge_then_lock(camL, camR, settle_s=SETTLE_S, throw_n=THROW_N)
    else:
        throw_frames(camL, camR, n=30)

    best, dts_ms, stats = pick_best_pair(camL, camR, n_tries=N_TRIES, sleep_between=SLEEP_BETWEEN)
    dt_ns, rgbL, rgbR, tsL, tsR = best
    print(f"[PAIR] best dt = {dt_ns/1e6:.3f} ms")
    print("[DT stats ms]", stats)

    np.savetxt("dt_ms.txt", dts_ms, fmt="%.4f")
    with open("dt_stats.txt", "w", encoding="utf-8") as f:
        f.write(str(stats) + "\n")

    cv2.imwrite("left_raw_rgb.png", rgbL)
    cv2.imwrite("right_raw_rgb.png", rgbR)
    cv2.imwrite("raw_pair_rgb.png", np.hstack([rgbL, rgbR]))

    rectL, rectR = rectify_and_crop(rgbL, rgbR, map1_x, map1_y, map2_x, map2_y)
    cv2.imwrite("rect_left_roi.png", rectL)
    cv2.imwrite("rect_right_roi.png", rectR)
    cv2.imwrite("pair_roi.png", np.hstack([rectL, rectR]))
    save_pair_with_hlines(rectL, rectR, "pair_roi_lines.png", step=HLINE_STEP)

    gL0 = cv2.cvtColor(rectL, cv2.COLOR_RGB2GRAY)
    gR0 = cv2.cvtColor(rectR, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("gray_left.png", gL0)
    cv2.imwrite("gray_right.png", gR0)

    # 额外保存增强后的灰度，方便你肉眼判断“增强有没有把纹理抬起来”
    gL_en = apply_prefilter(gL0)
    gR_en = apply_prefilter(gR0)
    cv2.imwrite("gray_left_enh.png", gL_en)
    cv2.imwrite("gray_right_enh.png", gR_en)

    # === pipeline ===
    disp_raw, raw_valid, disp_post, post_valid, tag = compute_disp_pipeline(rectL, rectR)

    # ---- raw ----
    vis_raw, color_raw = disp_to_vis(disp_raw, raw_valid)
    cv2.imwrite("disp_raw_gray.png", vis_raw)
    cv2.imwrite("disp_raw_color.png", color_raw)
    np.save("disp_raw.npy", disp_raw)
    if SAVE_DEBUG_MASKS:
        save_mask_png(raw_valid, "mask_raw_valid.png")

    # ---- post (still saved as disp_wls_* for downstream compatibility) ----
    if disp_post is not None:
        vis_post, color_post = disp_to_vis(disp_post, post_valid)
        cv2.imwrite("disp_wls_gray.png", vis_post)
        cv2.imwrite("disp_wls_color.png", color_post)
        np.save("disp_wls.npy", disp_post)
        if SAVE_DEBUG_MASKS:
            save_mask_png(post_valid, "mask_post_valid.png")

        print(
            f"[INFO] disp mode: {tag} | raw min/max: {float(disp_raw.min())} {float(disp_raw.max())} "
            f"| post min/max: {float(disp_post.min())} {float(disp_post.max())}"
        )
    else:
        color_post = None
        post_valid = None
        print(f"[INFO] disp mode: {tag} | raw min/max: {float(disp_raw.min())} {float(disp_raw.max())}")

    print("[DONE] saved outputs:")
    print("  pair_roi_lines.png (check alignment)")
    print("  gray_left.png / gray_right.png")
    print("  gray_left_enh.png / gray_right_enh.png (NEW)")
    print("  disp_raw_gray/color + disp_raw.npy")
    print("  disp_wls_gray/color + disp_wls.npy (if enabled)")
    if SAVE_DEBUG_MASKS:
        print("  mask_raw_valid.png mask_post_valid.png (debug valid regions)")

    camL.stop()
    camR.stop()

    # =========================
    # GUI: Click-to-measure
    # =========================
    if not ENABLE_GUI_CLICK_MEASURE:
        return

    # 用 post（如果有）测距；没有就用 raw
    disp_m = disp_post if disp_post is not None else disp_raw
    valid_m = post_valid if post_valid is not None else raw_valid
    color_m = color_post if color_post is not None else color_raw

    # 画面上显示提示文字（NEW: baseline_m）
    overlay = color_m.copy()
    msg1 = f"Click to measure depth: Z = fx*B/d   fx={fx:.1f}px  B={baseline_m*1000:.1f}mm"
    msg2 = "Keys: q/ESC exit | r toggle raw/post (if post exists)"
    msg3 = f"Prefilter: TEX={ENABLE_TEXTURE_ENHANCE}({TEXTURE_MODE}) GAUSS={ENABLE_GAUSSIAN_BLUR}(k={GAUSS_KSIZE})"
    cv2.putText(overlay, msg1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, msg2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, msg3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    state = {
        "use_post": (disp_post is not None),
        "disp_raw": disp_raw, "valid_raw": raw_valid, "color_raw": color_raw,
        "disp_post": disp_post, "valid_post": post_valid, "color_post": color_post,
        "disp": disp_m, "valid": valid_m, "color": overlay,
        "fx": fx, "B": baseline_m,
        "last": None,  # (x,y,z,info)
    }

    def refresh_display():
        img = state["color"].copy()
        if state["last"] is not None:
            x, y, z, info = state["last"]
            cv2.circle(img, (x, y), 5, (255, 255, 255), 2)
            if z is None:
                txt = f"({x},{y})  N/A  info={info}"
            else:
                txt = f"({x},{y})  disp_med={info['d_med']:.3f} px  Z={z:.3f} m  (valid={info['valid']})"
            cv2.rectangle(img, (10, 90), (10 + 1100, 125), (0, 0, 0), -1)
            cv2.putText(img, txt, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("disp_click_measure", img)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        z, info = measure_depth_at(
            state["disp"], state["valid"], x, y,
            fx=state["fx"], baseline_m=state["B"],
            win=MEASURE_WIN, min_valid=MIN_VALID_PIXELS
        )
        state["last"] = (x, y, z, info)
        if z is None:
            print(f"[CLICK] ({x},{y}) -> N/A  info={info}")
        else:
            print(f"[CLICK] ({x},{y}) -> disp_med={info['d_med']:.4f}px  Z={z:.4f} m  info={info}")
        refresh_display()

    def set_mode(use_post: bool):
        if use_post and state["disp_post"] is None:
            return
        if use_post:
            disp = state["disp_post"]; valid = state["valid_post"]; base_color = state["color_post"]
            tagm = "POST"
        else:
            disp = state["disp_raw"]; valid = state["valid_raw"]; base_color = state["color_raw"]
            tagm = "RAW"
        base = base_color.copy()
        cv2.putText(base, f"[MODE] {tagm}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, msg1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, msg2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, msg3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        state["disp"] = disp
        state["valid"] = valid
        state["color"] = base
        state["last"] = None
        refresh_display()

    pair_lines_bgr = cv2.imread("pair_roi_lines.png")  # 读出来就是 BGR

    cv2.namedWindow("disp_click_measure", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("disp_click_measure", on_mouse)

    cv2.namedWindow("pair_roi_lines", cv2.WINDOW_NORMAL)
    cv2.imshow("pair_roi_lines", pair_lines_bgr)

    set_mode(use_post=(disp_post is not None))

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in [27, ord('q')]:  # ESC / q
            break
        if k == ord('r'):
            if disp_post is None:
                continue
            state["use_post"] = not state["use_post"]
            set_mode(state["use_post"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
