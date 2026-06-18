"""Low-dependency telemetry suitable for Raspberry Pi deployment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time


_PROCESS_START = time.monotonic()


def _memory_stats() -> tuple[float | None, float | None]:
    source = Path("/proc/meminfo")
    if not source.exists():
        return None, None
    values: dict[str, float] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = float(raw.strip().split()[0]) * 1024.0
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, None
    return total, total - available


def _temperature_c() -> float | None:
    for path in (
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/thermal/thermal_zone1/temp"),
    ):
        if path.exists():
            try:
                return float(path.read_text(encoding="utf-8").strip()) / 1000.0
            except (OSError, ValueError):
                continue
    return None


def collect_system_telemetry(storage_path: str | Path = ".") -> dict[str, object]:
    disk = shutil.disk_usage(Path(storage_path).resolve())
    memory_total, memory_used = _memory_stats()
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except (AttributeError, OSError):
        load_1m = load_5m = load_15m = None
    if hasattr(os, "uname"):
        hostname = os.uname().nodename
    else:
        hostname = os.environ.get("COMPUTERNAME", "windows-node")
    return {
        "hostname": hostname,
        "process_uptime_s": time.monotonic() - _PROCESS_START,
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
        "temperature_c": _temperature_c(),
        "memory_total_bytes": memory_total,
        "memory_used_bytes": memory_used,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "cpu_count": os.cpu_count(),
    }
