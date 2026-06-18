"""Frame extraction and image quality gating."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from ._optional import require_cv2
from .io_utils import ensure_dir


@dataclass(frozen=True)
class FrameExtractionResult:
    written_frames: int
    skipped_blurry_frames: int
    total_frames_seen: int
    output_dir: str
    skipped_redundant_frames: int = 0
    sampled_frames: int = 0
    selected_source_indices: tuple[int, ...] = ()
    median_motion_px: float | None = None
    median_overlap_ratio: float | None = None


@dataclass(frozen=True)
class _KeyframeCandidate:
    source_index: int
    blur_score: float
    motion_px: float
    overlap_ratio: float
    encoded_jpeg: bytes


def _preview_gray(cv2, frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _motion_and_overlap(cv2, previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    previous_points, previous_descriptors = detector.detectAndCompute(previous, None)
    current_points, current_descriptors = detector.detectAndCompute(current, None)
    if (
        previous_descriptors is None
        or current_descriptors is None
        or len(previous_points) < 20
        or len(current_points) < 20
    ):
        return math.inf, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 12:
        return math.inf, len(good) / max(min(len(previous_points), len(current_points)), 1)

    displacements = [
        np.linalg.norm(
            np.asarray(current_points[match.trainIdx].pt)
            - np.asarray(previous_points[match.queryIdx].pt)
        )
        for match in good
    ]
    overlap = len(good) / max(min(len(previous_points), len(current_points)), 1)
    return float(np.median(displacements)), float(overlap)


def _limit_temporal_candidates(
    candidates: list[_KeyframeCandidate],
    max_frames: int | None,
) -> list[_KeyframeCandidate]:
    if max_frames is None or len(candidates) <= max_frames:
        return candidates
    if max_frames < 2:
        return [candidates[0]]

    selected = [candidates[0]]
    interior = candidates[1:-1]
    bucket_count = max_frames - 2
    for bucket in range(bucket_count):
        start = math.floor(bucket * len(interior) / bucket_count)
        end = math.floor((bucket + 1) * len(interior) / bucket_count)
        choices = interior[start : max(start + 1, end)]
        selected.append(
            max(
                choices,
                key=lambda item: (
                    math.log1p(item.blur_score)
                    + min(item.motion_px if math.isfinite(item.motion_px) else 40.0, 40.0)
                    + 12.0 * min(item.overlap_ratio, 0.5)
                ),
            )
        )
    selected.append(candidates[-1])
    return selected


def extract_adaptive_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    every_n: int = 3,
    min_blur: float = 15.0,
    max_frames: int = 120,
    min_motion_px: float = 5.0,
    max_gap_s: float = 0.75,
    prefix: str = "frame",
) -> FrameExtractionResult:
    """Select sharp, motion-distinct keyframes while retaining full-orbit coverage."""
    if every_n <= 0:
        raise ValueError("every_n must be positive")
    if max_frames < 20:
        raise ValueError("max_frames must be at least 20")
    cv2 = require_cv2()
    target_dir = ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
    max_gap_frames = max(every_n, round(max_gap_s * fps))
    candidates: list[_KeyframeCandidate] = []
    previous_selected_gray: np.ndarray | None = None
    last_selected_index = -max_gap_frames
    last_sharp: tuple[int, float, np.ndarray, bytes] | None = None
    skipped_blurry = 0
    skipped_redundant = 0
    sampled = 0
    seen = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if seen % every_n != 0:
            seen += 1
            continue
        sampled += 1
        gray = _preview_gray(cv2, frame)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < min_blur:
            skipped_blurry += 1
            seen += 1
            continue
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not encoded_ok:
            seen += 1
            continue
        encoded_bytes = encoded.tobytes()
        last_sharp = (seen, blur_score, gray, encoded_bytes)

        if previous_selected_gray is None:
            motion_px, overlap_ratio = math.inf, 1.0
            accept = True
        else:
            motion_px, overlap_ratio = _motion_and_overlap(
                cv2,
                previous_selected_gray,
                gray,
            )
            accept = (
                motion_px >= min_motion_px
                or seen - last_selected_index >= max_gap_frames
            )
        if accept:
            candidates.append(
                _KeyframeCandidate(
                    source_index=seen,
                    blur_score=blur_score,
                    motion_px=motion_px,
                    overlap_ratio=overlap_ratio,
                    encoded_jpeg=encoded_bytes,
                )
            )
            previous_selected_gray = gray
            last_selected_index = seen
        else:
            skipped_redundant += 1
        seen += 1

    capture.release()
    if last_sharp and (
        not candidates or candidates[-1].source_index != last_sharp[0]
    ):
        motion_px, overlap_ratio = (
            _motion_and_overlap(cv2, previous_selected_gray, last_sharp[2])
            if previous_selected_gray is not None
            else (math.inf, 1.0)
        )
        candidates.append(
            _KeyframeCandidate(
                source_index=last_sharp[0],
                blur_score=last_sharp[1],
                motion_px=motion_px,
                overlap_ratio=overlap_ratio,
                encoded_jpeg=last_sharp[3],
            )
        )

    selected = _limit_temporal_candidates(candidates, max_frames)
    for old_image in target_dir.glob(f"{prefix}_*.jpg"):
        old_image.unlink()
    for candidate in selected:
        (target_dir / f"{prefix}_{candidate.source_index:06d}.jpg").write_bytes(
            candidate.encoded_jpeg
        )

    motions = [
        item.motion_px
        for item in selected
        if math.isfinite(item.motion_px)
    ]
    overlaps = [item.overlap_ratio for item in selected]
    return FrameExtractionResult(
        written_frames=len(selected),
        skipped_blurry_frames=skipped_blurry,
        total_frames_seen=seen,
        output_dir=str(target_dir),
        skipped_redundant_frames=skipped_redundant
        + max(0, len(candidates) - len(selected)),
        sampled_frames=sampled,
        selected_source_indices=tuple(item.source_index for item in selected),
        median_motion_px=float(np.median(motions)) if motions else None,
        median_overlap_ratio=float(np.median(overlaps)) if overlaps else None,
    )


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
