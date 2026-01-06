from __future__ import annotations
import time

import cv2
import numpy as np

from .settings import UiSettings


class FrameProvider:
    """
    Replace THIS with your integration to the stereo pipeline.

    Contract:
      get_frame() -> np.ndarray (BGR image) for preview display.

    You can output:
      - left only
      - side-by-side (left|right)
      - composite (left|right|disparity_color)
    """

    def __init__(self, settings: UiSettings, use_webcam: bool = False) -> None:
        self.s = settings
        self.use_webcam = use_webcam
        self._t0 = time.time()

        self._cap = None
        if use_webcam:
            self._cap = cv2.VideoCapture(0)
            # best-effort set
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.s.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.s.target_fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> np.ndarray:
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return self._make_overlay(frame)

        # fallback: generate synthetic stereo-like frame
        h, w = self.s.height, self.s.width
        t = time.time() - self._t0
        base = np.zeros((h, w, 3), dtype=np.uint8)

        # moving circle
        cx = int((w * 0.5) + (w * 0.25) * np.sin(t))
        cy = int((h * 0.5) + (h * 0.15) * np.cos(t * 0.7))
        cv2.circle(base, (cx, cy), 35, (0, 255, 255), -1)

        # text
        cv2.putText(base, "UI Preview (replace FrameProvider with pipeline output)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # make a "right" view by shifting
        right = np.roll(base, shift=18, axis=1)

        parts = [base]
        if self.s.show_right:
            parts.append(right)

        if self.s.show_disparity:
            # fake disparity visualization
            disp = cv2.cvtColor(cv2.absdiff(base, right), cv2.COLOR_BGR2GRAY)
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
            parts.append(disp)

        frame = np.hstack(parts)
        return self._make_overlay(frame)

    def _make_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        # add a small status overlay area
        cv2.putText(frame_bgr, f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                    (20, frame_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        return frame_bgr
