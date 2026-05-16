# test_rectify_calib_pair.py
import cv2
import numpy as np
import os

from stereo_calibrate import LEFT_DIR, RIGHT_DIR  # 直接复用你的变量名

def main():
    # 1. 读标定参数
    params = np.load("stereo_params.npz")
    image_size   = tuple(params["image_size"])
    map1_x       = params["map1_x"]
    map1_y       = params["map1_y"]
    map2_x       = params["map2_x"]
    map2_y       = params["map2_y"]

    print("image_size:", image_size)
    print("map1_x.shape:", map1_x.shape)

    # 2. 随便拿一组标定用的图片出来
    left_path  = sorted([p for p in os.listdir(LEFT_DIR) if p.startswith("left_")])[0]
    right_path = sorted([p for p in os.listdir(RIGHT_DIR) if p.startswith("right_")])[0]

    left_img  = cv2.imread(os.path.join(LEFT_DIR, left_path))
    right_img = cv2.imread(os.path.join(RIGHT_DIR, right_path))

    print("left_img.shape:", left_img.shape)

    h, w = left_img.shape[:2]
    if (w, h) != image_size:
        print("[WARN] left_img 尺寸和 image_size 不一致:", (w, h), image_size)

    rect_left  = cv2.remap(left_img,  map1_x, map1_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, map2_x, map2_y, cv2.INTER_LINEAR)

    cv2.imwrite("rect_left_calib.png",  rect_left)
    cv2.imwrite("rect_right_calib.png", rect_right)

    print("Saved rect_left_calib.png, rect_right_calib.png")

if __name__ == "__main__":
    main()
