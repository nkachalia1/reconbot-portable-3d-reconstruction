"""Pi-friendly image capture helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ._optional import require_cv2
from .io_utils import ensure_dir


@dataclass(frozen=True)
class CaptureResult:
    written_images: int
    output_dir: str


def capture_sequence(
    output_dir: str | Path,
    camera_index: int = 0,
    count: int = 30,
    interval_s: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    preview: bool = False,
    prefix: str = "capture",
) -> CaptureResult:
    if count <= 0:
        raise ValueError("count must be positive")
    if interval_s < 0:
        raise ValueError("interval_s must be non-negative")

    cv2 = require_cv2()
    target_dir = ensure_dir(output_dir)
    camera = cv2.VideoCapture(camera_index)
    if width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    written = 0
    try:
        while written < count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            timestamp_ms = int(time.time() * 1000)
            output_file = target_dir / f"{prefix}_{timestamp_ms}_{written:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            written += 1

            if preview:
                cv2.imshow("reconbot capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if written < count:
                time.sleep(interval_s)
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()

    return CaptureResult(written, str(target_dir))
