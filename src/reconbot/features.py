"""Feature detection, matching, and sequence benchmarking."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._optional import require_cv2
from .io_utils import ensure_dir, list_images


@dataclass(frozen=True)
class FeatureMatchResult:
    method: str
    keypoints_a: np.ndarray
    keypoints_b: np.ndarray
    matched_points_a: np.ndarray
    matched_points_b: np.ndarray
    distances: np.ndarray
    detect_time_s: float
    match_time_s: float

    @property
    def match_count(self) -> int:
        return int(len(self.distances))


def _create_detector(method: str, max_features: int):
    cv2 = require_cv2()
    normalized = method.upper()
    if normalized == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not include SIFT. Install opencv-contrib-python.")
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    if normalized == "ORB":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING
    raise ValueError(f"Unsupported feature method: {method}")


def detect_and_match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    method: str = "SIFT",
    max_features: int = 4000,
    ratio: float = 0.75,
) -> FeatureMatchResult:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    cv2 = require_cv2()
    detector, norm_type = _create_detector(method, max_features)

    if image_a.ndim == 3:
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    if image_b.ndim == 3:
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    keypoints_a, descriptors_a = detector.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(image_b, None)
    detect_time = time.perf_counter() - t0

    if descriptors_a is None or descriptors_b is None:
        return FeatureMatchResult(
            method=method.upper(),
            keypoints_a=np.empty((0, 2)),
            keypoints_b=np.empty((0, 2)),
            matched_points_a=np.empty((0, 2)),
            matched_points_b=np.empty((0, 2)),
            distances=np.empty((0,)),
            detect_time_s=detect_time,
            match_time_s=0.0,
        )

    t1 = time.perf_counter()
    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = []
    for candidate in raw_matches:
        if len(candidate) < 2:
            continue
        first, second = candidate
        if first.distance <= ratio * second.distance:
            good.append(first)
    match_time = time.perf_counter() - t1

    pts_a_all = np.asarray([kp.pt for kp in keypoints_a], dtype=float)
    pts_b_all = np.asarray([kp.pt for kp in keypoints_b], dtype=float)
    if not good:
        matched_a = np.empty((0, 2), dtype=float)
        matched_b = np.empty((0, 2), dtype=float)
        distances = np.empty((0,), dtype=float)
    else:
        idx_a = np.asarray([m.queryIdx for m in good], dtype=int)
        idx_b = np.asarray([m.trainIdx for m in good], dtype=int)
        matched_a = pts_a_all[idx_a]
        matched_b = pts_b_all[idx_b]
        distances = np.asarray([m.distance for m in good], dtype=float)

    return FeatureMatchResult(
        method=method.upper(),
        keypoints_a=pts_a_all,
        keypoints_b=pts_b_all,
        matched_points_a=matched_a,
        matched_points_b=matched_b,
        distances=distances,
        detect_time_s=detect_time,
        match_time_s=match_time,
    )


def geometric_inlier_ratio(points_a: np.ndarray, points_b: np.ndarray) -> float:
    cv2 = require_cv2()
    pts_a = np.asarray(points_a, dtype=float).reshape(-1, 2)
    pts_b = np.asarray(points_b, dtype=float).reshape(-1, 2)
    if len(pts_a) < 8:
        return 0.0
    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return 0.0
    return float(mask.reshape(-1).astype(bool).mean())


def benchmark_sequence(
    image_path: str | Path,
    output_csv: str | Path,
    methods: tuple[str, ...] = ("SIFT", "ORB"),
    max_features: int = 4000,
    ratio: float = 0.75,
) -> list[dict[str, object]]:
    cv2 = require_cv2()
    images = list_images(image_path)
    if len(images) < 2:
        raise RuntimeError(f"Need at least two images to benchmark features in {image_path}")

    rows: list[dict[str, object]] = []
    for idx, (path_a, path_b) in enumerate(zip(images[:-1], images[1:], strict=False)):
        image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            continue
        for method in methods:
            result = detect_and_match(image_a, image_b, method, max_features, ratio)
            inlier_ratio = geometric_inlier_ratio(result.matched_points_a, result.matched_points_b)
            rows.append(
                {
                    "pair_index": idx,
                    "image_a": str(path_a),
                    "image_b": str(path_b),
                    "method": result.method,
                    "keypoints_a": len(result.keypoints_a),
                    "keypoints_b": len(result.keypoints_b),
                    "matches": result.match_count,
                    "geometric_inlier_ratio": inlier_ratio,
                    "detect_time_s": result.detect_time_s,
                    "match_time_s": result.match_time_s,
                    "total_time_s": result.detect_time_s + result.match_time_s,
                }
            )

    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    fieldnames = [
        "pair_index",
        "image_a",
        "image_b",
        "method",
        "keypoints_a",
        "keypoints_b",
        "matches",
        "geometric_inlier_ratio",
        "detect_time_s",
        "match_time_s",
        "total_time_s",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
