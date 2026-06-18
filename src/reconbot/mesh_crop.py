"""Crop OpenMVS binary triangle meshes while preserving texture coordinates."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .io_utils import ensure_dir


def crop_binary_triangle_ply(
    input_path: str | Path,
    output_path: str | Path,
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
) -> tuple[int, int]:
    source = Path(input_path)
    data = source.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError("PLY header does not contain end_header")
    header_end += len(marker)
    header = data[:header_end]
    if b"format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary little-endian PLY files are supported")

    vertex_match = re.search(rb"element vertex (\d+)", header)
    face_match = re.search(rb"element face (\d+)", header)
    if vertex_match is None or face_match is None:
        raise ValueError("PLY must contain vertex and face elements")
    vertex_count = int(vertex_match.group(1))
    face_count = int(face_match.group(1))

    vertices = np.frombuffer(
        data,
        dtype="<f4",
        count=vertex_count * 3,
        offset=header_end,
    ).reshape(vertex_count, 3)
    face_offset = header_end + vertex_count * 3 * np.dtype("<f4").itemsize
    textured = b"property list uint8 float32 texcoord" in header
    if textured:
        face_dtype = np.dtype(
            [
                ("vertex_count", "u1"),
                ("indices", "<u4", (3,)),
                ("texcoord_count", "u1"),
                ("texcoords", "<f4", (6,)),
            ]
        )
    else:
        face_dtype = np.dtype([("vertex_count", "u1"), ("indices", "<u4", (3,))])

    faces = np.frombuffer(data, dtype=face_dtype, count=face_count, offset=face_offset).copy()
    if not np.all(faces["vertex_count"] == 3):
        raise ValueError("PLY must contain triangle faces")
    if textured and not np.all(faces["texcoord_count"] == 6):
        raise ValueError("Textured triangle faces must contain six texture coordinates")

    lower = np.asarray(min_xyz, dtype=float)
    upper = np.asarray(max_xyz, dtype=float)
    inside = np.all((vertices >= lower) & (vertices <= upper), axis=1)
    keep_faces = np.all(inside[faces["indices"]], axis=1)
    cropped_faces = faces[keep_faces]
    used_vertices = np.unique(cropped_faces["indices"])
    if len(used_vertices) == 0:
        raise ValueError("Crop bounds did not retain any faces")

    remap = np.full(vertex_count, -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices))
    cropped_faces["indices"] = remap[cropped_faces["indices"]].astype("<u4")
    cropped_vertices = np.asarray(vertices[used_vertices], dtype="<f4")

    output_header = re.sub(
        rb"element vertex \d+",
        f"element vertex {len(cropped_vertices)}".encode(),
        header,
        count=1,
    )
    output_header = re.sub(
        rb"element face \d+",
        f"element face {len(cropped_faces)}".encode(),
        output_header,
        count=1,
    )
    crop_comment = (
        f"comment CropBounds {lower.tolist()} {upper.tolist()}\n".encode()
    )
    output_header = output_header.replace(marker, crop_comment + marker)

    target = Path(output_path)
    ensure_dir(target.parent)
    target.write_bytes(
        output_header + cropped_vertices.tobytes() + cropped_faces.tobytes()
    )
    return len(cropped_vertices), len(cropped_faces)
