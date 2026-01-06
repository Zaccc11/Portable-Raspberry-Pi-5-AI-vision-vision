from __future__ import annotations
from dataclasses import dataclass


@dataclass
class UiSettings:
    # Preview
    target_fps: int = 30
    show_disparity: bool = True
    show_right: bool = True

    # Recording
    record_fps: int = 30

    # Resolution for preview generator/capture
    width: int = 848
    height: int = 480
