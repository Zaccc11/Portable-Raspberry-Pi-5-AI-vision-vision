from picamera2 import Picamera2
import time
import cv2
import os

def main():
    save_dir = "calib_images"
    left_dir = os.path.join(save_dir, "left")
    right_dir = os.path.join(save_dir, "right")

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    cam_left = Picamera2(0)
    cam_right = Picamera2(1)


    config = cam_left.create_still_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    cam_left.configure(config)
    cam_right.configure(config)

    cam_left.start()
    cam_right.start()


    time.sleep(2.0)

    idx = 0
    print("已启动双目采集工具。每按一次回车，保存一对棋盘格图像；输入 q + 回车 退出。")

    while True:
        cmd = input("回车拍照，q + 回车退出：")
        if cmd.strip().lower() == "q":
            break

        frame_left = cam_left.capture_array()
        frame_right = cam_right.capture_array()

  
        frame_left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

        left_path = os.path.join(left_dir, f"left_{idx:03d}.png")
        right_path = os.path.join(right_dir, f"right_{idx:03d}.png")

        cv2.imwrite(left_path, frame_left_bgr)
        cv2.imwrite(right_path, frame_right_bgr)

        print(f"保存第 {idx} 组：{left_path}  /  {right_path}")
        idx += 1

    cam_left.close()
    cam_right.close()
    print("已退出采集工具。")

if __name__ == "__main__":
    main()
