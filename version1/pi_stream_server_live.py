import socket
import struct
import pickle
import time
import threading
import select
import math
import shutil
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2

import sgbm_640_30fps as s

try:
    from smbus2 import SMBus
except Exception:
    try:
        from smbus import SMBus
    except Exception:
        SMBus = None


HOST = "0.0.0.0"
PORT = 9999
JPEG_QUALITY = 55
STREAM_FPS = 18.0


def encode_jpg(img, quality=55):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def send_packet(conn, packet):
    data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    conn.sendall(header + data)


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def latest_pair(Ls, Rs):
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


def apply_manual_exposure(camL, camR, exposure_us):
    exposure_us = int(exposure_us)
    exposure_us = max(500, min(30000, exposure_us))
    controls = {
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": exposure_us,
    }
    camL.set_controls(controls)
    camR.set_controls(controls)


def apply_auto_exposure(camL, camR):
    controls = {
        "AeEnable": True,
        "AwbEnable": True,
    }
    camL.set_controls(controls)
    camR.set_controls(controls)


def get_storage_pct():
    try:
        total, used, _ = shutil.disk_usage("/")
        if total <= 0:
            return None
        return 100.0 * used / total
    except Exception:
        return None


def get_power_info():
    try:
        return None, "External"
    except Exception:
        return None, "N/A"


class MPU6050Reader:
    def __init__(self, bus_id=1, addr=0x68):
        self.addr = addr
        self.bus = None
        self.ok = False

        if SMBus is None:
            return

        try:
            self.bus = SMBus(bus_id)
            self.bus.write_byte_data(self.addr, 0x6B, 0)
            self.ok = True
        except Exception:
            self.bus = None
            self.ok = False

    def read_word_2c(self, reg):
        high = self.bus.read_byte_data(self.addr, reg)
        low = self.bus.read_byte_data(self.addr, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def read(self):
        if not self.ok:
            return None

        try:
            ax_raw = self.read_word_2c(0x3B)
            ay_raw = self.read_word_2c(0x3D)
            az_raw = self.read_word_2c(0x3F)

            gx_raw = self.read_word_2c(0x43)
            gy_raw = self.read_word_2c(0x45)
            gz_raw = self.read_word_2c(0x47)

            ax = ax_raw / 16384.0
            ay = ay_raw / 16384.0
            az = az_raw / 16384.0

            gx = gx_raw / 131.0
            gy = gy_raw / 131.0
            gz = gz_raw / 131.0

            roll = math.degrees(math.atan2(ay, az))
            pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

            return {
                "roll": float(roll),
                "pitch": float(pitch),
                "gx": float(gx),
                "gy": float(gy),
                "gz": float(gz),
            }
        except Exception:
            return None


def main():
    min_s, num_s = s.compute_scaled_disp_params(
        s.MIN_DISPARITIES_FULL_TARGET,
        s.NUM_DISPARITIES_FULL_TARGET,
        s.DISP_SCALE
    )
    full_max = s.MIN_DISPARITIES_FULL_TARGET + s.NUM_DISPARITIES_FULL_TARGET

    params = np.load(s.CALIB_NPZ)
    image_size = tuple(params["image_size"])
    full_w, full_h = int(image_size[0]), int(image_size[1])

    map1_x = params["map1_x"].astype(np.float32)
    map1_y = params["map1_y"].astype(np.float32)
    map2_x = params["map2_x"].astype(np.float32)
    map2_y = params["map2_y"].astype(np.float32)

    fx, baseline_m = s.get_fx_baseline_from_params(params)
    crop_xywh = s.compute_crop_from_valid_rois(params, full_w, full_h)

    imu = MPU6050Reader()

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
        controls={"FrameRate": s.FRAME_RATE},
        buffer_count=s.LOW_LATENCY_BUFFER_COUNT,
        queue=s.LOW_LATENCY_QUEUE,
    )
    cfgR = camR.create_video_configuration(
        main={"size": (full_w, full_h), "format": "RGB888"},
        controls={"FrameRate": s.FRAME_RATE},
        buffer_count=s.LOW_LATENCY_BUFFER_COUNT,
        queue=s.LOW_LATENCY_QUEUE,
    )

    camL.configure(cfgL)
    camR.configure(cfgR)
    camL.start()
    camR.start()

    if s.USE_AE_AWB_CONVERGE_THEN_LOCK:
        s.converge_then_lock(camL, camR, settle_s=s.SETTLE_S, throw_n=s.THROW_N)
        exposure_auto_default = False
        mL = camL.capture_metadata()
        exposure_us_default = int(mL.get("ExposureTime", 8000))
    else:
        s.throw_frames(camL, camR, n=30)
        exposure_auto_default = True
        exposure_us_default = 8000

    lock = threading.Lock()
    stop = threading.Event()

    rawL = deque(maxlen=s.BUF_LEN)
    rawR = deque(maxlen=s.BUF_LEN)
    ids = {"L": 0, "R": 0}

    shared = {
        "rectL": None,
        "rectR": None,
        "disp_full": None,
        "valid": None,
        "dt_ms": None,
        "pi_proc_ms": None,
        "comp_fps": 0.0,
        "pair_key": None,
        "post_mode": "NONE",
        "exposure_auto": exposure_auto_default,
        "exposure_us": exposure_us_default,
        "storage_pct": None,
        "battery_pct": None,
        "power_text": "External",
        "imu_roll": None,
        "imu_pitch": None,
        "imu_gx": None,
        "imu_gy": None,
        "imu_gz": None,
        "fx": fx,
        "baseline_m": baseline_m,
    }

    control = {
        "post_mode": "NONE",
        "exposure_auto": exposure_auto_default,
        "exposure_us": exposure_us_default,
    }

    def status_thread():
        while not stop.is_set():
            storage_pct = get_storage_pct()
            battery_pct, power_text = get_power_info()
            imu_data = imu.read() if imu.ok else None

            with lock:
                shared["storage_pct"] = storage_pct
                shared["battery_pct"] = battery_pct
                shared["power_text"] = power_text
                if imu_data is not None:
                    shared["imu_roll"] = imu_data["roll"]
                    shared["imu_pitch"] = imu_data["pitch"]
                    shared["imu_gx"] = imu_data["gx"]
                    shared["imu_gy"] = imu_data["gy"]
                    shared["imu_gz"] = imu_data["gz"]

            time.sleep(0.25)

    def cap_thread(cam, side):
        while not stop.is_set():
            rgb, ts = s.capture_one_with_ts(cam)
            with lock:
                ids[side] += 1
                if side == "L":
                    rawL.append((ids[side], rgb, ts))
                else:
                    rawR.append((ids[side], rgb, ts))

    def comp_thread():
        comp_fps = 0.0
        t_prev_new = None
        last_pair_key = None
        last_mode = None

        left_matcher = None
        right_matcher = None
        wls = None
        use_xi = s.has_ximgproc()

        def rebuild(mode_name):
            nonlocal left_matcher, right_matcher, wls
            mode_name = str(mode_name).upper()
            left_matcher = s.create_sgbm(min_s, num_s)
            right_matcher = None
            wls = None

            if mode_name in ("LRC", "WLS") and use_xi:
                right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
                if mode_name == "LRC":
                    wls = s.build_wls_filter(
                        left_matcher,
                        lrc_thresh_px=s.WLS_LRC_THRESH,
                        lam=s.LRC_ONLY_LAMBDA,
                        sigma_color=s.WLS_SIGMA_COLOR,
                        disc_radius=s.WLS_DISC_RADIUS,
                    )
                else:
                    wls = s.build_wls_filter(
                        left_matcher,
                        lrc_thresh_px=s.WLS_LRC_THRESH,
                        lam=s.WLS_LAMBDA,
                        sigma_color=s.WLS_SIGMA_COLOR,
                        disc_radius=s.WLS_DISC_RADIUS,
                    )

        t_next = time.perf_counter()

        while not stop.is_set():
            now = time.perf_counter()
            sleep_s = t_next - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += s.COMP_PERIOD_S

            with lock:
                Ls = list(rawL)
                Rs = list(rawR)
                current_mode = control["post_mode"]

            if current_mode != last_mode:
                rebuild(current_mode)
                last_mode = current_mode

            if not Ls or not Rs:
                continue

            best, dt_ms = latest_pair(Ls, Rs)
            if best is None or dt_ms is None or dt_ms > s.DT_OK_FOR_COMP_MS:
                with lock:
                    shared["dt_ms"] = dt_ms
                continue

            idL, rgbL, tsL, idR, rgbR, tsR = best
            pair_key = (idL, idR)
            if pair_key == last_pair_key:
                continue

            proc_t0 = time.perf_counter()

            rectL, rectR = s.rectify_and_crop_common(
                rgbL, rgbR, map1_x, map1_y, map2_x, map2_y, crop_xywh
            )

            gL = cv2.cvtColor(rectL, cv2.COLOR_RGB2GRAY)
            gR = cv2.cvtColor(rectR, cv2.COLOR_RGB2GRAY)

            if abs(s.DISP_SCALE - 1.0) < 1e-6:
                gL_s, gR_s = gL, gR
                guide_s = rectL
            else:
                gL_s = cv2.resize(gL, None, fx=s.DISP_SCALE, fy=s.DISP_SCALE, interpolation=cv2.INTER_AREA)
                gR_s = cv2.resize(gR, None, fx=s.DISP_SCALE, fy=s.DISP_SCALE, interpolation=cv2.INTER_AREA)
                guide_s = cv2.resize(rectL, (gL_s.shape[1], gL_s.shape[0]), interpolation=cv2.INTER_AREA)

            dispL_q4_s = left_matcher.compute(gL_s, gR_s)

            if current_mode in ("LRC", "WLS") and wls is not None and right_matcher is not None:
                dispR_q4_s = right_matcher.compute(gR_s, gL_s)
                disp_q4_s = wls.filter(dispL_q4_s, guide_s, None, dispR_q4_s)
                disp_q4_s = s.apply_speckle_removal_q4(disp_q4_s)
            else:
                disp_q4_s = dispL_q4_s

            disp_full, valid = s.q4_scaled_to_full_disp_and_valid(
                disp_q4_s,
                (gL.shape[0], gL.shape[1]),
                max_disp_full=full_max,
                scale=s.DISP_SCALE
            )

            proc_t1 = time.perf_counter()
            pi_proc_ms = (proc_t1 - proc_t0) * 1000.0

            nowc = time.perf_counter()
            if t_prev_new is not None:
                dtc = nowc - t_prev_new
                if dtc > 1e-6:
                    inst = 1.0 / dtc
                    comp_fps = (1.0 - s.FPS_EMA_ALPHA) * comp_fps + s.FPS_EMA_ALPHA * inst
            t_prev_new = nowc

            last_pair_key = pair_key

            with lock:
                shared["rectL"] = rectL
                shared["rectR"] = rectR
                shared["disp_full"] = disp_full.astype(np.float32)
                shared["valid"] = valid
                shared["dt_ms"] = dt_ms
                shared["pi_proc_ms"] = pi_proc_ms
                shared["comp_fps"] = comp_fps
                shared["pair_key"] = pair_key
                shared["post_mode"] = current_mode
                shared["exposure_auto"] = control["exposure_auto"]
                shared["exposure_us"] = control["exposure_us"]

    threading.Thread(target=status_thread, daemon=True).start()
    threading.Thread(target=cap_thread, args=(camL, "L"), daemon=True).start()
    threading.Thread(target=cap_thread, args=(camR, "R"), daemon=True).start()
    threading.Thread(target=comp_thread, daemon=True).start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[STREAM] Waiting on {HOST}:{PORT}")

    conn, addr = server.accept()
    print("[STREAM] Mac connected from:", addr)

    frame_interval = 1.0 / STREAM_FPS
    stream_frame_id = 0

    try:
        while True:
            ready, _, _ = select.select([conn], [], [], 0)

            if ready:
                try:
                    msg_len_bytes = recv_exact(conn, 4)
                    msg_len = struct.unpack("!I", msg_len_bytes)[0]
                    payload = recv_exact(conn, msg_len)
                    msg = pickle.loads(payload)

                    if isinstance(msg, dict):
                        msg_type = msg.get("type", "")

                        if msg_type == "ping":
                            send_packet(conn, {"type": "pong", "ping_id": msg.get("ping_id")})

                        elif msg_type == "set_postproc":
                            mode = str(msg.get("mode", "NONE")).upper()
                            if mode not in ("NONE", "LRC", "WLS"):
                                mode = "NONE"
                            with lock:
                                control["post_mode"] = mode

                        elif msg_type == "set_exposure_mode":
                            exposure_auto = bool(msg.get("auto", True))
                            with lock:
                                control["exposure_auto"] = exposure_auto
                                exposure_us = control["exposure_us"]

                            if exposure_auto:
                                apply_auto_exposure(camL, camR)
                            else:
                                apply_manual_exposure(camL, camR, exposure_us)

                        elif msg_type == "set_exposure_value":
                            exposure_us = int(msg.get("value", 8000))
                            exposure_us = max(500, min(30000, exposure_us))
                            with lock:
                                control["exposure_us"] = exposure_us
                                exposure_auto = control["exposure_auto"]

                            if not exposure_auto:
                                apply_manual_exposure(camL, camR, exposure_us)

                        elif msg_type == "click":
                            x = int(msg["x"])
                            y = int(msg["y"])

                            with lock:
                                disp_full = None if shared["disp_full"] is None else shared["disp_full"].copy()
                                valid = None if shared["valid"] is None else shared["valid"].copy()

                            if disp_full is not None and valid is not None:
                                z_cm, info = s.measure_depth_at(
                                    disp_full,
                                    valid,
                                    x,
                                    y,
                                    fx,
                                    baseline_m,
                                    win=s.MEASURE_WIN,
                                    min_valid=s.MIN_VALID_PIXELS
                                )

                                if z_cm is not None:
                                    z_new = s.z_correction_poly(float(z_cm))
                                    z_sum = float(z_cm) + float(z_new)
                                else:
                                    z_new = None
                                    z_sum = None

                                send_packet(conn, {
                                    "type": "result",
                                    "x": x,
                                    "y": y,
                                    "z_cm": z_cm,
                                    "z_new": z_new,
                                    "z_sum": z_sum,
                                    "info": info,
                                })

                except (ConnectionResetError, BrokenPipeError):
                    break
                except Exception as e:
                    print("[STREAM] recv error:", e)

            t0 = time.time()

            with lock:
                rectL = None if shared["rectL"] is None else shared["rectL"].copy()
                rectR = None if shared["rectR"] is None else shared["rectR"].copy()
                disp_full = None if shared["disp_full"] is None else shared["disp_full"].copy()

                status_packet = {
                    "fps": float(shared["comp_fps"]),
                    "stereo_dt_ms": 0.0 if shared["dt_ms"] is None else float(shared["dt_ms"]),
                    "pi_proc_ms": 0.0 if shared["pi_proc_ms"] is None else float(shared["pi_proc_ms"]),
                    "storage_percent": None if shared["storage_pct"] is None else float(shared["storage_pct"]),
                    "battery_percent": None if shared["battery_pct"] is None else float(shared["battery_pct"]),
                    "power_source": str(shared["power_text"]),
                    "exposure_auto": bool(shared["exposure_auto"]),
                    "exposure_us": int(shared["exposure_us"]),
                    "pair_key": str(shared["pair_key"]),
                    "post_mode": str(shared["post_mode"]),
                    "imu_roll": shared["imu_roll"],
                    "imu_pitch": shared["imu_pitch"],
                    "imu_gx": shared["imu_gx"],
                    "imu_gy": shared["imu_gy"],
                    "imu_gz": shared["imu_gz"],
                    "fx": float(shared["fx"]),
                    "baseline_m": float(shared["baseline_m"]),
                }

            if rectL is not None and rectR is not None and disp_full is not None:
                try:
                    stream_frame_id += 1

                    disp_u16 = np.clip(disp_full * 16.0, 0, 65535).astype(np.uint16)

                    payload = {
                        "type": "frame",
                        "stream_frame_id": stream_frame_id,
                        "left_jpg": encode_jpg(rectL, JPEG_QUALITY),
                        "right_jpg": encode_jpg(rectR, JPEG_QUALITY),
                        "status": status_packet,
                        "disp_shape": disp_u16.shape,
                        "disp_dtype": str(disp_u16.dtype),
                        "disp_scale_q": 16.0,
                        "disp_bytes": disp_u16.tobytes(),
                    }

                    send_packet(conn, payload)

                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception as e:
                    print("[STREAM] send error:", e)

            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        stop.set()
        time.sleep(0.1)
        try:
            camL.stop()
            camR.stop()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        server.close()


if __name__ == "__main__":
    main()