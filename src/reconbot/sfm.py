"""Two-view Structure-from-Motion baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._optional import require_cv2
from .evaluation import reprojection_summary
from .features import detect_and_match
from .geometry import (
    camera_center_from_extrinsics,
    make_projection,
    points_in_front,
    triangulate_points_dlt,
)
from .io_utils import ensure_dir, load_intrinsics_yaml, save_point_cloud_ply, write_json


@dataclass(frozen=True)
class PairSfmResult:
    output_dir: str
    point_count: int
    match_count: int
    pose_inlier_count: int
    mean_reprojection_error_px: float


def _sample_colors(image_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    rounded = np.rint(points).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, w - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, h - 1)
    bgr = image_bgr[rounded[:, 1], rounded[:, 0], :]
    return bgr[:, ::-1]


def run_pair_sfm(
    image_a: str | Path,
    image_b: str | Path,
    intrinsics_path: str | Path,
    output_dir: str | Path,
    method: str = "SIFT",
    max_features: int = 4000,
    ratio: float = 0.75,
) -> PairSfmResult:
    cv2 = require_cv2()
    target_dir = ensure_dir(output_dir)
    camera_matrix, distortion, intrinsics = load_intrinsics_yaml(intrinsics_path)

    img_a_color = cv2.imread(str(image_a), cv2.IMREAD_COLOR)
    img_b_color = cv2.imread(str(image_b), cv2.IMREAD_COLOR)
    if img_a_color is None:
        raise FileNotFoundError(f"Could not read image: {image_a}")
    if img_b_color is None:
        raise FileNotFoundError(f"Could not read image: {image_b}")
    img_a_gray = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(img_b_color, cv2.COLOR_BGR2GRAY)

    matches = detect_and_match(img_a_gray, img_b_gray, method, max_features, ratio)
    if matches.match_count < 8:
        raise RuntimeError(f"Need at least 8 matches for essential matrix; found {matches.match_count}")

    pts_a = matches.matched_points_a.astype(np.float64)
    pts_b = matches.matched_points_b.astype(np.float64)
    pts_a_u = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), camera_matrix, distortion, P=camera_matrix)
    pts_b_u = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), camera_matrix, distortion, P=camera_matrix)
    pts_a_u = pts_a_u.reshape(-1, 2)
    pts_b_u = pts_b_u.reshape(-1, 2)

    essential, essential_mask = cv2.findEssentialMat(
        pts_a_u,
        pts_b_u,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None:
        raise RuntimeError("Essential matrix estimation failed")
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        pts_a_u,
        pts_b_u,
        camera_matrix,
        mask=essential_mask,
    )

    pose_inliers = pose_mask.reshape(-1).astype(bool)
    inlier_a = pts_a_u[pose_inliers]
    inlier_b = pts_b_u[pose_inliers]
    if len(inlier_a) < 8:
        raise RuntimeError(f"Need at least 8 pose inliers; found {len(inlier_a)}")

    projection_a = make_projection(camera_matrix, np.eye(3), np.zeros(3))
    projection_b = make_projection(camera_matrix, rotation, translation.reshape(3))
    points_3d = triangulate_points_dlt(projection_a, projection_b, inlier_a, inlier_b)

    front_a = points_in_front(points_3d)
    front_b = points_in_front(points_3d, rotation, translation.reshape(3))
    finite = np.isfinite(points_3d).all(axis=1)
    keep = front_a & front_b & finite
    points_3d = points_3d[keep]
    inlier_a = inlier_a[keep]
    inlier_b = inlier_b[keep]

    colors = _sample_colors(img_a_color, inlier_a)
    save_point_cloud_ply(target_dir / "points.ply", points_3d, colors)

    report_a = reprojection_summary(points_3d, inlier_a, camera_matrix)
    report_b = reprojection_summary(points_3d, inlier_b, camera_matrix, rotation, translation.reshape(3))
    center_a = [0.0, 0.0, 0.0]
    center_b = camera_center_from_extrinsics(rotation, translation.reshape(3)).tolist()

    metrics = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "intrinsics": str(intrinsics_path),
        "intrinsics_image_width": intrinsics.get("image_width"),
        "intrinsics_image_height": intrinsics.get("image_height"),
        "feature_method": matches.method,
        "keypoints_a": int(len(matches.keypoints_a)),
        "keypoints_b": int(len(matches.keypoints_b)),
        "matches": matches.match_count,
        "pose_inliers": int(len(inlier_a)),
        "points_3d": int(len(points_3d)),
        "camera_centers": [center_a, center_b],
        "reprojection_camera_a": report_a,
        "reprojection_camera_b": report_b,
        "scale_note": "Monocular two-view SfM recovers structure up to an arbitrary scale.",
    }
    write_json(target_dir / "pair_sfm_metrics.json", metrics)

    mean_error = float((report_a["mean_px"] + report_b["mean_px"]) / 2.0)
    return PairSfmResult(
        output_dir=str(target_dir),
        point_count=int(len(points_3d)),
        match_count=matches.match_count,
        pose_inlier_count=int(len(inlier_a)),
        mean_reprojection_error_px=mean_error,
    )
