import tempfile
import unittest
from pathlib import Path

import numpy as np

from reconbot.mesh_crop import crop_binary_triangle_ply
from reconbot.metric_scale import load_triangle_mesh


class MeshCropTests(unittest.TestCase):
    def test_crops_plain_triangle_mesh(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            dtype="<f4",
        )
        face_dtype = np.dtype([("count", "u1"), ("indices", "<u4", (3,))])
        faces = np.zeros(2, dtype=face_dtype)
        faces["count"] = 3
        faces["indices"] = [[0, 1, 2], [3, 4, 5]]
        header = (
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"element vertex 6\n"
            b"property float32 x\n"
            b"property float32 y\n"
            b"property float32 z\n"
            b"element face 2\n"
            b"property list uint8 uint32 vertex_indices\n"
            b"end_header\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ply"
            output = Path(directory) / "cropped.ply"
            source.write_bytes(header + vertices.tobytes() + faces.tobytes())

            vertex_count, face_count = crop_binary_triangle_ply(
                source,
                output,
                (-0.1, -0.1, -0.1),
                (1.1, 1.1, 0.1),
            )
            loaded_vertices, loaded_faces = load_triangle_mesh(output)

        self.assertEqual(vertex_count, 3)
        self.assertEqual(face_count, 1)
        self.assertEqual(loaded_vertices.shape, (3, 3))
        np.testing.assert_array_equal(loaded_faces, [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
