import math
import unittest

import numpy as np

from reconbot.evaluation import point_cloud_metrics, reprojection_summary
from reconbot.geometry import (
    camera_center_from_extrinsics,
    make_projection,
    project_points,
    rotation_y,
    triangulate_points_dlt,
)


class GeometryTests(unittest.TestCase):
    def test_projection_round_trip_with_triangulation(self):
        camera_matrix = np.asarray(
            [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation_a = np.eye(3)
        translation_a = np.zeros(3)
        rotation_b = rotation_y(math.radians(5.0))
        translation_b = np.asarray([-0.4, 0.0, 0.0])

        points_world = np.asarray(
            [
                [-0.2, -0.1, 3.0],
                [0.1, 0.2, 4.0],
                [0.3, -0.2, 5.0],
                [-0.4, 0.15, 4.5],
            ]
        )
        pixels_a, _ = project_points(points_world, camera_matrix, rotation_a, translation_a)
        pixels_b, _ = project_points(points_world, camera_matrix, rotation_b, translation_b)
        projection_a = make_projection(camera_matrix, rotation_a, translation_a)
        projection_b = make_projection(camera_matrix, rotation_b, translation_b)

        reconstructed = triangulate_points_dlt(projection_a, projection_b, pixels_a, pixels_b)

        np.testing.assert_allclose(reconstructed, points_world, atol=1e-8)

    def test_camera_center_from_extrinsics(self):
        rotation = np.eye(3)
        translation = np.asarray([-1.0, 0.0, 0.0])

        center = camera_center_from_extrinsics(rotation, translation)

        np.testing.assert_allclose(center, np.asarray([1.0, 0.0, 0.0]))

    def test_reprojection_summary(self):
        camera_matrix = np.asarray(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points = np.asarray([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])
        observed, _ = project_points(points, camera_matrix)

        summary = reprojection_summary(points, observed, camera_matrix)

        self.assertEqual(summary["count"], 2.0)
        self.assertAlmostEqual(summary["rmse_px"], 0.0, places=10)

    def test_point_cloud_metrics(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        metrics = point_cloud_metrics(points, voxel_size=0.5)

        self.assertEqual(metrics.point_count, 5)
        self.assertEqual(metrics.bbox_extent, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(metrics.points_per_cubic_meter, 5.0)
        self.assertGreater(metrics.occupied_voxels, 0)
        self.assertLessEqual(metrics.voxel_coverage_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
