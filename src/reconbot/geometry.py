"""Pure NumPy geometry routines for calibrated monocular reconstruction."""

from __future__ import annotations

import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    return np.hstack([pts, ones])


def from_homogeneous(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    denom = pts[..., -1:]
    if np.any(np.abs(denom) < eps):
        raise ValueError("Cannot convert homogeneous points with near-zero scale")
    return pts[..., :-1] / denom


def make_projection(camera_matrix: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3, 1)
    return k @ np.hstack([r, t])


def project_points(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=float).reshape(3)

    camera_points = (r @ pts.T).T + t.reshape(1, 3)
    pixels_h = (k @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, camera_points[:, 2]


def reprojection_errors(
    points_3d: np.ndarray,
    observed_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    projected, _ = project_points(points_3d, camera_matrix, rotation, translation)
    observed = np.asarray(observed_pixels, dtype=float).reshape(-1, 2)
    if len(projected) != len(observed):
        raise ValueError("points_3d and observed_pixels must have the same length")
    return np.linalg.norm(projected - observed, axis=1)


def triangulate_points_dlt(
    projection_a: np.ndarray,
    projection_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    p1 = np.asarray(projection_a, dtype=float).reshape(3, 4)
    p2 = np.asarray(projection_b, dtype=float).reshape(3, 4)
    pts1 = np.asarray(points_a, dtype=float).reshape(-1, 2)
    pts2 = np.asarray(points_b, dtype=float).reshape(-1, 2)
    if len(pts1) != len(pts2):
        raise ValueError("points_a and points_b must have the same length")

    points_3d = np.empty((len(pts1), 3), dtype=float)
    for idx, ((u1, v1), (u2, v2)) in enumerate(zip(pts1, pts2, strict=True)):
        a = np.vstack(
            [
                u1 * p1[2, :] - p1[0, :],
                v1 * p1[2, :] - p1[1, :],
                u2 * p2[2, :] - p2[0, :],
                v2 * p2[2, :] - p2[1, :],
            ]
        )
        _, _, vh = np.linalg.svd(a)
        homogeneous = vh[-1, :]
        points_3d[idx] = from_homogeneous(homogeneous)
    return points_3d


def camera_center_from_extrinsics(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    return -r.T @ t


def points_in_front(
    points_3d: np.ndarray,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 3)
    r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=float).reshape(3)
    camera_points = (r @ pts.T).T + t.reshape(1, 3)
    return camera_points[:, 2] > eps


def rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
