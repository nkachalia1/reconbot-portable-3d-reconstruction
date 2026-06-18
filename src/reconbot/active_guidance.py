"""Active perception guidance from adjacent-frame feature overlap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._optional import require_cv2
from .features import detect_and_match, geometric_inlier_ratio


@dataclass(frozen=True)
class Guidance:
    match_count: int
    geometric_inlier_ratio: float
    recommendation: str
    code: str = "unknown"
    median_motion_px: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "match_count": self.match_count,
            "geometric_inlier_ratio": self.geometric_inlier_ratio,
            "recommendation": self.recommendation,
            "code": self.code,
            "median_motion_px": self.median_motion_px,
        }


def guidance_code_from_stats(
    match_count: int,
    inlier_ratio: float,
    median_motion_px: float = 0.0,
    blur_score: float | None = None,
    min_blur: float = 50.0,
) -> str:
    if blur_score is not None and blur_score < min_blur:
        return "slow_down"
    if match_count < 80:
        return "recover_overlap"
    if inlier_ratio < 0.30:
        return "hold_steady"
    if (match_count > 900 and inlier_ratio > 0.70) or median_motion_px < 12.0:
        return "add_baseline"
    if inlier_ratio > 0.45:
        return "continue_arc"
    return "intermediate_frame"


def guidance_text(code: str, arc_direction: str = "right") -> str:
    direction = "left" if arc_direction.lower() == "left" else "right"
    messages = {
        "reference_frame": f"Reference frame captured. Begin a slow lateral arc to the {direction}.",
        "slow_down": "Slow down and hold the camera steady before the next capture.",
        "recover_overlap": "Overlap was lost: move closer or return toward the previous viewpoint.",
        "hold_steady": "Hold steady and avoid rotation until feature geometry becomes consistent.",
        "add_baseline": f"Overlap is very high: move farther {direction} to add lateral baseline.",
        "continue_arc": f"Good viewpoint: continue the smooth arc to the {direction}.",
        "intermediate_frame": f"Capture an intermediate frame, then continue {direction}.",
    }
    return messages.get(code, "Capture another viewpoint while keeping the target centered.")


def guidance_from_stats(match_count: int, inlier_ratio: float) -> str:
    return guidance_text(guidance_code_from_stats(match_count, inlier_ratio))


def analyze_pair(
    image_a: str | Path,
    image_b: str | Path,
    method: str = "SIFT",
    max_features: int = 4000,
) -> Guidance:
    cv2 = require_cv2()
    img_a = cv2.imread(str(image_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(image_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None:
        raise FileNotFoundError(f"Could not read image: {image_a}")
    if img_b is None:
        raise FileNotFoundError(f"Could not read image: {image_b}")
    result = detect_and_match(img_a, img_b, method=method, max_features=max_features)
    inlier_ratio = geometric_inlier_ratio(result.matched_points_a, result.matched_points_b)
    if result.match_count:
        motion = result.matched_points_b - result.matched_points_a
        median_motion_px = float(np.median(np.linalg.norm(motion, axis=1)))
    else:
        median_motion_px = 0.0
    code = guidance_code_from_stats(result.match_count, inlier_ratio, median_motion_px)
    return Guidance(
        match_count=result.match_count,
        geometric_inlier_ratio=inlier_ratio,
        recommendation=guidance_text(code),
        code=code,
        median_motion_px=median_motion_px,
    )
