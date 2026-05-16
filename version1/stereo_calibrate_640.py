import cv2
import numpy as np
import glob
import os
import re

# ========= Chessboard =========
CHESSBOARD_COLS = 9
CHESSBOARD_ROWS = 6
SQUARE_SIZE = 25.0  # mm

# ========= 640 dataset folders =========
BASE_DIR = "calib_images_640"
LEFT_DIR = os.path.join(BASE_DIR, "left")
RIGHT_DIR = os.path.join(BASE_DIR, "right")

# ========= Output =========
OUT_NPZ_640 = "stereo_params_640.npz"

# ========= Quality threshold =========
BAD_THRESH_PX = 0.8   # per-view RMS > this -> candidate to delete/recapture


def parse_idx_from_name(path, prefix):
    """
    parse left_001.png / right_001.png -> 1
    """
    base = os.path.basename(path)
    m = re.match(rf"^{prefix}_(\d+)\.png$", base)
    return int(m.group(1)) if m else None


def build_index_map(folder, prefix):
    paths = glob.glob(os.path.join(folder, f"{prefix}_*.png"))
    mp = {}
    for p in paths:
        idx = parse_idx_from_name(p, prefix)
        if idx is not None:
            mp[idx] = p
    return mp


def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D):
    """per-view RMS error (px)"""
    errs = []
    for i, objp in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)
        err = cv2.norm(obs, proj, cv2.NORM_L2) / np.sqrt(len(objp))
        errs.append(float(err))
    return errs


def main():
    # 1) object points
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= float(SQUARE_SIZE)

    # 2) align pairs by index
    left_map = build_index_map(LEFT_DIR, "left")
    right_map = build_index_map(RIGHT_DIR, "right")

    left_set = set(left_map.keys())
    right_set = set(right_map.keys())
    common = sorted(left_set & right_set)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)

    print(f"[INFO] LEFT={len(left_set)} RIGHT={len(right_set)} COMMON={len(common)}")
    if only_left:
        print(f"[WARN] only LEFT (missing RIGHT): {only_left[:20]}{' ...' if len(only_left)>20 else ''}")
    if only_right:
        print(f"[WARN] only RIGHT (missing LEFT): {only_right[:20]}{' ...' if len(only_right)>20 else ''}")

    if len(common) == 0:
        raise RuntimeError("common pairs=0. Check naming left_001.png/right_001.png and folders.")

    # 3) detect chessboard
    objpoints = []
    imgpoints_L = []
    imgpoints_R = []
    img_size = None
    used_pairs = []
    failed_pairs = []

    print(f"\n[INFO] Detect chessboard {CHESSBOARD_COLS}x{CHESSBOARD_ROWS} ...")
    for idx in common:
        lp = left_map[idx]
        rp = right_map[idx]

        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)
        if imgL is None or imgR is None:
            failed_pairs.append(idx)
            print(f"❌ READ FAIL pair {idx:03d}")
            continue

        gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            h, w = gL.shape[:2]
            img_size = (w, h)
            print("[INFO] img_size =", img_size)

        retL, cornersL = cv2.findChessboardCorners(gL, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)
        retR, cornersR = cv2.findChessboardCorners(gR, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        if retL and retR:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(gL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(gR, cornersR, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_L.append(cornersL)
            imgpoints_R.append(cornersR)
            used_pairs.append(idx)
            print(f"✅ OK pair {idx:03d}")
        else:
            failed_pairs.append(idx)
            print(f"❌ FAIL chessboard pair {idx:03d}")

    if img_size is None or len(objpoints) < 5:
        raise RuntimeError(f"有效样本太少：{len(objpoints)}（>=5 才稳）")

    print(f"\n[INFO] used_pairs={len(used_pairs)}  failed_pairs={len(failed_pairs)}  img_size={img_size}")

    # 4) mono calibrate
    retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_L, img_size, None, None)
    retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_R, img_size, None, None)
    print("\n[CALIB] left RMS :", retL)
    print("[CALIB] right RMS:", retR)

    # 5) per-pair error
    errL = compute_reprojection_errors(objpoints, imgpoints_L, rvecsL, tvecsL, K1, D1)
    errR = compute_reprojection_errors(objpoints, imgpoints_R, rvecsR, tvecsR, K2, D2)

    bad_pairs = []
    print("\n[PER-PAIR ERROR]")
    for i, idx in enumerate(used_pairs):
        el = errL[i]
        er = errR[i]
        print(f"  pair {idx:03d}   L={el:.3f}px  R={er:.3f}px")
        if el > BAD_THRESH_PX or er > BAD_THRESH_PX:
            bad_pairs.append((idx, el, er))

    if bad_pairs:
        print(f"\n[BAD LIST] threshold={BAD_THRESH_PX}px 建议删/重拍：")
        for idx, el, er in bad_pairs:
            print(f"  pair {idx:03d}  L={el:.3f}px  R={er:.3f}px")
    else:
        print(f"\n[GOOD] all used pairs <= {BAD_THRESH_PX}px")

    # 6) stereoCalibrate (fix intrinsics)
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    stereo_rms, K1s, D1s, K2s, D2s, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K1, D1, K2, D2, img_size,
        criteria=criteria_stereo, flags=flags
    )
    print("\n[STEREO] RMS:", stereo_rms)
    print("[STEREO] T (mm):", T.ravel())

    # 7) rectify (640-native)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1s, D1s, K2s, D2s, img_size, R, T, alpha=0
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K1s, D1s, R1, P1, img_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2s, D2s, R2, P2, img_size, cv2.CV_32FC1)

    print("\n[MAP] map1:", map1_x.shape, " map2:", map2_x.shape)
    print("[ROI] roi1:", roi1, " roi2:", roi2)

    # 8) save 640 npz
    np.savez(
        OUT_NPZ_640,
        image_size=np.array(img_size, dtype=np.int32),
        cameraMatrix1=K1s, distCoeffs1=D1s,
        cameraMatrix2=K2s, distCoeffs2=D2s,
        R=R, T=T,
        R1=R1, R2=R2,
        P1=P1, P2=P2,
        Q=Q,
        roi1=np.array(roi1, dtype=np.int32),
        roi2=np.array(roi2, dtype=np.int32),
        map1_x=map1_x, map1_y=map1_y,
        map2_x=map2_x, map2_y=map2_y,
        used_pairs=np.array(used_pairs, dtype=np.int32),
        failed_pairs=np.array(failed_pairs, dtype=np.int32),
        note=np.array([f"Native 640 calibration from {BASE_DIR}"], dtype=object),
    )

    print(f"\n✅ Done. Saved native-640 stereo params to: {OUT_NPZ_640}")
    print("   -> 用它去跑你的 640 live pipeline（rectify 用这里的 maps）")


if __name__ == "__main__":
    main()
