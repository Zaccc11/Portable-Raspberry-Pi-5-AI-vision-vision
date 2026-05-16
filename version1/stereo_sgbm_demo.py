import cv2
import numpy as np

def main():
    # 1. 读取左右图（注意：OpenCV 默认是 BGR）
    left  = cv2.imread("left_bgr_fixed.png")
    right = cv2.imread("right_bgr_fixed.png")

    if left is None or right is None:
        raise FileNotFoundError("找不到 left_fixed.png 或 right_fixed.png")

    # ---- 步骤 1：尺寸对齐 & 转灰度 ----
    # 如果两张图尺寸不完全一致，这里强行 resize 一下
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    print("current size is ",w,h)

    left  = left[:h, :w]
    right = right[:h, :w]

    # 也可以在这里缩小一点做实验（加快 SGBM）
    # scale = 0.5
    # left  = cv2.resize(left,  None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # ---- 步骤 2：配置 SGBM 参数 ----
    # 这些参数是比较“通用”的一套，后面可以再针对你房间慢慢调
    window_size = 5          # SAD 窗口大小（奇数）
    min_disp    = 0          # 最小视差
    num_disp    = 128        # 视差范围，必须是 16 的整数倍，例如 64/96/128/160...

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size ** 2,
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # ---- 步骤 3：计算视差 ----
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 无效/负视差设为 NaN，后面方便过滤
    disp[disp <= min_disp] = np.nan

    # ---- 步骤 4：简单后处理：中值滤波 + 掩膜 ----
    disp_valid = np.nan_to_num(disp, nan=0.0)
    disp_valid = cv2.medianBlur(disp_valid, 5)

    # 有效像素掩膜（非 0）
    mask = disp_valid > 0

    # ---- 步骤 5：归一化显示（灰度 & 伪彩色）----
    disp_norm = disp_valid.copy()
    # 只对有效区域做 min-max 归一化，避免被大量 0 拉低
    if np.any(mask):
        min_v = disp_valid[mask].min()
        max_v = disp_valid[mask].max()
        disp_norm = (disp_valid - min_v) / (max_v - min_v + 1e-6)
    else:
        print("警告：几乎没有有效视差，请检查基线、对准、曝光等。")

    disp_gray = (disp_norm * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_gray, cv2.COLORMAP_JET)

    # 把无效区域画成纯黑，更清楚一点
    disp_gray[~mask] = 0
    disp_color[~mask] = 0

    # ---- 步骤 6：保存结果 ----
    cv2.imwrite("disp_gray.png", disp_gray)
    cv2.imwrite("disp_color.png", disp_color)

    print("视差图已保存为 disp_gray.png 和 disp_color.png")

    # ---- （可选）步骤 7：如果你有标定结果，就可以算深度 ----
    # 这里给个模板，等你以后标定完把 f 和 B 填上去
    #
    # f  = 1000.0  # 焦距（像素），从标定得到
    # B  = 0.06    # 基线，单位：米（比如 60mm）
    # Z  = f * B / (disp_valid + 1e-6)   # 深度（米）
    #
    # 然后可以像上面一样归一化，把 Z 画成灰度/伪彩深度图

if __name__ == "__main__":
    main()
