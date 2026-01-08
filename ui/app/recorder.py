from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class RecordConfig:
    out_dir: Path
    fps: int = 30


class FrameRecorder:
    """
    Minimal recorder:
    - Saves preview frames to MP4 (H264 if available, else mp4v)
    - Also writes a timestamp log (unix seconds)
    """

    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._ts_file = None
        self._start_time = None
        self._frame_size = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, cfg: RecordConfig, frame_size: tuple[int, int]) -> None:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._frame_size = frame_size

        # fallback to mp4v
        fourcc_candidates = ["avc1", "H264", "mp4v"]
        out_path = cfg.out_dir / "recording.mp4"

        writer = None
        for fcc in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*fcc)
            w = cv2.VideoWriter(str(out_path), fourcc, cfg.fps, frame_size)
            if w.isOpened():
                writer = w
                break

        if writer is None:
            raise RuntimeError("Failed to open VideoWriter (no supported codec).")

        self._writer = writer
        self._ts_file = open(cfg.out_dir / "timestamps.csv", "w", encoding="utf-8")
        self._ts_file.write("frame_idx,unix_time\n")
        self._start_time = time.time()

    def write(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        if self._writer is None or self._ts_file is None:
            return
        self._writer.write(frame_bgr)
        self._ts_file.write(f"{frame_idx},{time.time():.6f}\n")

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._writer = None

        if self._ts_file is not None:
            self._ts_file.close()
        self._ts_file = None
        self._start_time = None
        self._frame_size = None
