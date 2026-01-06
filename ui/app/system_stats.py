from __future__ import annotations
import shutil
import os
from dataclasses import dataclass


@dataclass
class SystemStats:
    cpu_temp_c: float | None
    disk_free_gb: float | None


def read_cpu_temp_c() -> float | None:
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(path, "r", encoding="utf-8") as f:
            milli = int(f.read().strip())
        return milli / 1000.0
    except Exception:
        return None


def read_disk_free_gb(path: str = "/") -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return None


def get_system_stats() -> SystemStats:
    return SystemStats(
        cpu_temp_c=read_cpu_temp_c(),
        disk_free_gb=read_disk_free_gb("/"),
    )
