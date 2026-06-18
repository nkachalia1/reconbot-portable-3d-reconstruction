"""Frame extraction and image quality gating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._optional import require_cv2
from .io_utils import ensure_dir


@dataclass(frozen=True)
class FrameExtractionResult:
    written_frames: int
    skipped_blurry_frames: int
    total_frames_seen: int
    output_dir: str


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 10,
    min_blur: float = 0.0,
    max_frames: int | None = None,
    prefix: str = "frame",
) -> FrameExtractionResult:
    if every_n <= 0:
        raise ValueError("every_n must be positive")
    cv2 = require_cv2()
    target_dir = ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    written = 0
    skipped_blurry = 0
    seen = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if seen % every_n != 0:
            seen += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < min_blur:
            skipped_blurry += 1
            seen += 1
            continue
        output_file = target_dir / f"{prefix}_{seen:06d}.jpg"
        cv2.imwrite(str(output_file), frame)
        written += 1
        seen += 1
        if max_frames is not None and written >= max_frames:
            break

    capture.release()
    return FrameExtractionResult(written, skipped_blurry, seen, str(target_dir))
