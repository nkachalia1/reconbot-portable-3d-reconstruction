"""Camera calibration from checkerboard image sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._optional import require_cv2
from .io_utils import list_images, save_intrinsics_yaml


@dataclass(frozen=True)
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]
    reprojection_error_px: float
    per_image_errors_px: list[float]
    used_images: list[str]
    rejected_images: list[str]


def _object_points(board_cols: int, board_rows: int, square_size_m: float) -> np.ndarray:
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_m)
    return objp


def calibrate_from_images(
    image_path: str | Path,
    board_cols: int,
    board_rows: int,
    square_size_m: float,
) -> CalibrationResult:
    cv2 = require_cv2()
    images = list_images(image_path)
    if not images:
        raise FileNotFoundError(f"No calibration images found in {image_path}")

    obj_template = _object_points(board_cols, board_rows, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    rejected_images: list[str] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_file in images:
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected_images.append(str(image_file))
            continue
        image_size = (int(image.shape[1]), int(image.shape[0]))
        found, corners = cv2.findChessboardCorners(image, (board_cols, board_rows), None)
        if not found:
            rejected_images.append(str(image_file))
            continue
        refined = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        object_points.append(obj_template.copy())
        image_points.append(refined)
        used_images.append(str(image_file))

    if image_size is None:
        raise RuntimeError("Could not read any calibration images")
    if len(object_points) < 5:
        raise RuntimeError(
            f"Need at least 5 usable checkerboard images; found {len(object_points)} usable images"
        )

    _, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_image_errors: list[float] = []
    all_squared_errors: list[np.ndarray] = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, distortion)
        observed = imgp.reshape(-1, 2)
        projected_2d = projected.reshape(-1, 2)
        distances = np.linalg.norm(observed - projected_2d, axis=1)
        per_image_errors.append(float(np.sqrt(np.mean(distances**2))))
        all_squared_errors.append(distances**2)

    reprojection_error = float(np.sqrt(np.mean(np.concatenate(all_squared_errors))))
    return CalibrationResult(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        reprojection_error_px=reprojection_error,
        per_image_errors_px=per_image_errors,
        used_images=used_images,
        rejected_images=rejected_images,
    )


def calibrate_and_save(
    image_path: str | Path,
    board_cols: int,
    board_rows: int,
    square_size_m: float,
    output_path: str | Path,
) -> CalibrationResult:
    result = calibrate_from_images(image_path, board_cols, board_rows, square_size_m)
    save_intrinsics_yaml(
        output_path,
        result.camera_matrix,
        result.distortion_coefficients,
        result.image_size,
        result.reprojection_error_px,
        result.per_image_errors_px,
        metadata={
            "board_cols": board_cols,
            "board_rows": board_rows,
            "square_size_m": square_size_m,
            "used_images": len(result.used_images),
            "rejected_images": len(result.rejected_images),
        },
    )
    return result
