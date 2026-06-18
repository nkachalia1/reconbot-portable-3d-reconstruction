import unittest

import numpy as np

from reconbot.metric_scale import estimate_metric_alignment


class MetricScaleTests(unittest.TestCase):
    def test_estimates_height_and_aligns_floor(self):
        floor = np.asarray(
            [[x, y, 0.0] for x in np.linspace(-2, 2, 8) for y in np.linspace(-2, 2, 8)]
        )
        floor_faces = []
        width = 8
        for row in range(width - 1):
            for col in range(width - 1):
                a = row * width + col
                b = a + 1
                c = a + width
                d = c + 1
                floor_faces.extend([[a, b, c], [b, d, c]])

        base = len(floor)
        object_vertices = np.asarray(
            [
                [-0.2, -0.2, 0.2],
                [0.2, -0.2, 0.2],
                [0.2, 0.2, 0.2],
                [-0.2, 0.2, 0.2],
                [-0.2, -0.2, 1.0],
                [0.2, -0.2, 1.0],
                [0.2, 0.2, 1.0],
                [-0.2, 0.2, 1.0],
            ]
        )
        object_faces = np.asarray(
            [
                [base + 0, base + 1, base + 4],
                [base + 1, base + 5, base + 4],
                [base + 1, base + 2, base + 5],
                [base + 2, base + 6, base + 5],
                [base + 2, base + 3, base + 6],
                [base + 3, base + 7, base + 6],
                [base + 3, base + 0, base + 7],
                [base + 0, base + 4, base + 7],
                [base + 4, base + 5, base + 6],
                [base + 4, base + 6, base + 7],
            ]
        )
        vertices = np.vstack([floor, object_vertices])
        faces = np.vstack([np.asarray(floor_faces), object_faces])

        alignment = estimate_metric_alignment(vertices, faces, reference_height_m=0.5)
        transformed = alignment.transform(vertices)

        self.assertAlmostEqual(alignment.scale_m_per_unit, 0.5, places=6)
        self.assertAlmostEqual(np.max(transformed[:, 2]), 0.5, places=6)
        self.assertAlmostEqual(np.median(transformed[: len(floor), 2]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
