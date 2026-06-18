"""Quantitative metrics for reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import reprojection_errors


@dataclass(frozen=True)
class PointCloudMetrics:
    point_count: int
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extent: list[float]
    bbox_volume: float
    points_per_cubic_meter: float | None
    occupied_voxels: int
    voxel_coverage_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "bbox_extent": self.bbox_extent,
            "bbox_volume": self.bbox_volume,
            "points_per_cubic_meter": self.points_per_cubic_meter,
            "occupied_voxels": self.occupied_voxels,
            "voxel_coverage_ratio": self.voxel_coverage_ratio,
        }


def point_cloud_metrics(points: np.ndarray, voxel_size: float = 0.05) -> PointCloudMetrics:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return PointCloudMetrics(0, [], [], [], 0.0, None, 0, None)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    extent = bbox_max - bbox_min
    volume = float(np.prod(extent))
    density = None if volume <= 0 else float(len(pts) / volume)

    voxel_indices = np.floor((pts - bbox_min.reshape(1, 3)) / voxel_size).astype(int)
    occupied = np.unique(voxel_indices, axis=0)
    grid_shape = np.maximum(voxel_indices.max(axis=0) + 1, 1)
    possible_voxels = int(np.prod(grid_shape))
    coverage = None if possible_voxels == 0 else float(len(occupied) / possible_voxels)

    return PointCloudMetrics(
        point_count=int(len(pts)),
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist(),
        bbox_extent=extent.tolist(),
        bbox_volume=volume,
        points_per_cubic_meter=density,
        occupied_voxels=int(len(occupied)),
        voxel_coverage_ratio=coverage,
    )


def reprojection_summary(
    points_3d: np.ndarray,
    observed_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> dict[str, float]:
    errors = reprojection_errors(points_3d, observed_pixels, camera_matrix, rotation, translation)
    if len(errors) == 0:
        return {"count": 0.0, "mean_px": 0.0, "median_px": 0.0, "rmse_px": 0.0, "max_px": 0.0}
    return {
        "count": float(len(errors)),
        "mean_px": float(errors.mean()),
        "median_px": float(np.median(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
        "max_px": float(errors.max()),
    }


def track_length_distribution(track_lengths: list[int] | np.ndarray) -> dict[str, float]:
    lengths = np.asarray(track_lengths, dtype=float).reshape(-1)
    if len(lengths) == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(lengths)),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "min": float(lengths.min()),
        "max": float(lengths.max()),
    }
