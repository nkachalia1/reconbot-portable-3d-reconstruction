"""Image quality measurements shared by live capture and offline processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._optional import require_cv2


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float
    mean_intensity: float
    dark_fraction: float
    bright_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "blur_score": self.blur_score,
            "mean_intensity": self.mean_intensity,
            "dark_fraction": self.dark_fraction,
            "bright_fraction": self.bright_fraction,
        }


def measure_frame_quality(frame: np.ndarray) -> FrameQuality:
    cv2 = require_cv2()
    image = np.asarray(frame)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return FrameQuality(
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        mean_intensity=float(gray.mean()),
        dark_fraction=float((gray <= 20).mean()),
        bright_fraction=float((gray >= 235).mean()),
    )
