import cv2
import numpy as np
import glob
import os
import re

CHESSBOARD_COLS = 9
CHESSBOARD_ROWS = 6
SQUARE_SIZE = 25.0

LEFT_DIR = "calib_images/left"
RIGHT_DIR = "calib_images/right"

# ====== 你可以改阈值：大于这个就算“坏数据候选” ======
BAD_THRESH_PX = 0.8

def parse_idx_from_name(path, prefix):
    """
    从 left_XX.png / right_XX.png 解析 XX (int)
    """
    base = os.path.basename(path)
    m = re.match(rf"^{prefix}_(\d+)\.png$", base)
    return int(m.group(1)) if m else None


def build_index_map(folder, prefix):
    """
    扫描 folder 下 prefix_XX.png，返回 {XX: filepath}
    只认严格命名：left_01.png / right_01.png
    """
    paths = glob.glob(os.path.join(folder, f"{prefix}_*.png"))
    mp = {}
    for p in paths:
        idx = parse_idx_from_name(p, prefix)
        if idx is not None:
            mp[idx] = p
    return mp


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, cameraMatrix, distCoeffs):
    """逐张图计算重投影误差，返回 per-view RMS (px)"""
    per_view_errors = []
    for i, objp in enumerate(objpoints):
        imgpoints2, _ = cv2.projectPoints(
            objp,
            rvecs[i],
            tvecs[i],
            cameraMatrix,
            distCoeffs
        )
        imgpoints2 = imgpoints2.reshape(-1, 2)
        imgp       = imgpoints[i].reshape(-1, 2)

        # RMS error (px)
        err = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / np.sqrt(len(objp))
        per_view_errors.append(err)

    return per_view_errors


def main():
    # 1) 构造棋盘 3D 点
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 2) 扫描 left/right 文件并按 index 对齐
    left_map = build_index_map(LEFT_DIR, "left")
    right_map = build_index_map(RIGHT_DIR, "right")

    left_set = set(left_map.keys())
    right_set = set(right_map.keys())
    common = sorted(left_set & right_set)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)

    print(f"[INFO] left count={len(left_set)}, right count={len(right_set)}, common pairs={len(common)}")
    if only_left:
        print(f"[WARN] only LEFT (missing RIGHT): {only_left}")
    if only_right:
        print(f"[WARN] only RIGHT (missing LEFT): {only_right}")

    if len(common) == 0:
        raise RuntimeError("没有任何可用配对（common pairs=0），请检查文件命名 left_XX.png/right_XX.png")

    # 3) 逐组检测棋盘角点：记录失败/成功的具体文件名
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    img_size = None

    used_pairs = []     # 只保存“成功检测棋盘”的 pair index
    failed_pairs = []   # 保存“棋盘检测失败”的 pair index

    print(f"\n[INFO] 开始检测棋盘角点... (target={CHESSBOARD_COLS}x{CHESSBOARD_ROWS})")

    for idx in common:
        left_path = left_map[idx]
        right_path = right_map[idx]

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            print(f"[WARN] 读图失败，跳过：left_{idx:02d}.png / right_{idx:02d}.png")
            failed_pairs.append(idx)
            continue

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(
            gray_left, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None
        )
        ret_r, corners_r = cv2.findChessboardCorners(
            gray_right, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None
        )

        if ret_l and ret_r:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners_l = cv2.cornerSubPix(
                gray_left, corners_l, (11, 11), (-1, -1), criteria
            )
            corners_r = cv2.cornerSubPix(
                gray_right, corners_r, (11, 11), (-1, -1), criteria
            )

            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)

            if img_size is None:
                h, w = gray_left.shape[:2]
                img_size = (w, h)
                print("[INFO] img_size set to:", img_size)

            used_pairs.append(idx)
            print(f"✅ OK: left_{idx:02d}.png / right_{idx:02d}.png")
        else:
            failed_pairs.append(idx)
            print(f"❌ FAIL chessboard: left_{idx:02d}.png / right_{idx:02d}.png")

    # 4) 输出失败列表，方便你删
    if failed_pairs:
        print("\n[FAILED LIST] 这些组棋盘检测失败（建议删/重拍）：")
        print("  " + ", ".join([f"{i:02d}" for i in failed_pairs]))

    if img_size is None or len(objpoints) < 5:
        raise RuntimeError(f"有效棋盘样本太少：{len(objpoints)} 组（>=5 才比较稳）")

    print(f"\n[INFO] 一共使用 {len(objpoints)} 组棋盘进行标定，图像尺寸 = {img_size}")

    # 5) 左右单目标定
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None
    )
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None
    )
    print("\n[CALIB] left camera RMS:", ret_l)
    print("[CALIB] right camera RMS:", ret_r)

    # 6) 逐组重投影误差（关键：按 left_XX/right_XX 打印）
    left_view_errors = compute_reprojection_errors(
        objpoints, imgpoints_left, rvecs_l, tvecs_l, mtx_l, dist_l
    )
    right_view_errors = compute_reprojection_errors(
        objpoints, imgpoints_right, rvecs_r, tvecs_r, mtx_r, dist_r
    )

    print("\n[PER-PAIR ERROR] 逐组误差(按PNG index):")
    bad_pairs = []
    for i in range(len(used_pairs)):
        idx = used_pairs[i]
        el = left_view_errors[i]
        er = right_view_errors[i]
        print(f"  pair {idx:02d}  left_{idx:02d}.png / right_{idx:02d}.png   L={el:.3f}px  R={er:.3f}px")

        if el > BAD_THRESH_PX or er > BAD_THRESH_PX:
            bad_pairs.append((idx, el, er))

    if bad_pairs:
        print(f"\n[BAD LIST] (threshold={BAD_THRESH_PX} px) 建议删除/重拍这些组：")
        for idx, el, er in bad_pairs:
            print(f"  pair {idx:02d}: left_{idx:02d}.png / right_{idx:02d}.png   L={el:.3f}px  R={er:.3f}px")
    else:
        print(f"\n[GOOD] 所有 used pairs 都 <= {BAD_THRESH_PX} px")

    # 7) stereoCalibrate
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        100,
        1e-5,
    )

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, \
        R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            mtx_l,
            dist_l,
            mtx_r,
            dist_r,
            img_size,
            criteria=criteria_stereo,
            flags=flags,
        )

    print("\n[STEREO] Stereo RMS:", retval)
    print("[STEREO] R:\n", R)
    print("[STEREO] T:\n", T)

    # 8) stereoRectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        img_size,
        R,
        T,
        alpha=0,
    )

    # 9) 映射表
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, img_size, cv2.CV_32FC1
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, img_size, cv2.CV_32FC1
    )

    print("\n[MAP] map1_x.shape:", map1_x.shape)
    print("[MAP] map2_x.shape:", map2_x.shape)

    # 10) 保存参数
    np.savez(
        "stereo_params.npz",
        image_size=img_size,
        cameraMatrix1=cameraMatrix1,
        distCoeffs1=distCoeffs1,
        cameraMatrix2=cameraMatrix2,
        distCoeffs2=distCoeffs2,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1_x=map1_x,
        map1_y=map1_y,
        map2_x=map2_x,
        map2_y=map2_y,
        used_pairs=np.array(used_pairs, dtype=np.int32),  # ✅ 保存：用了哪些 pair
        failed_pairs=np.array(failed_pairs, dtype=np.int32),  # ✅ 保存：失败哪些 pair
    )

    print("\n✅ 标定完成，参数已保存到 stereo_params.npz")


if __name__ == "__main__":
    main()
