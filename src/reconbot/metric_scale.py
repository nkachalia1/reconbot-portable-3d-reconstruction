"""Estimate metric scale from a known object height and transform binary PLY meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .io_utils import ensure_dir, write_json


@dataclass(frozen=True)
class MetricAlignment:
    floor_origin: np.ndarray
    rotation: np.ndarray
    scale_m_per_unit: float
    model_height: float
    peak_height: float
    object_vertex_count: int
    floor_sigma: float

    def transform(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return ((pts - self.floor_origin) @ self.rotation.T) * self.scale_m_per_unit


def _read_ply_header(data: bytes) -> tuple[bytes, int, int]:
    marker = b"end_header\n"
    offset = data.find(marker)
    if offset < 0:
        raise ValueError("PLY header does not contain end_header")
    header_end = offset + len(marker)
    header = data[:header_end]
    if b"format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary little-endian PLY files are supported")
    vertex_match = re.search(rb"element vertex (\d+)", header)
    face_match = re.search(rb"element face (\d+)", header)
    if vertex_match is None or face_match is None:
        raise ValueError("PLY must contain vertex and face elements")
    return header, int(vertex_match.group(1)), int(face_match.group(1))


def load_triangle_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = Path(path).read_bytes()
    header, vertex_count, face_count = _read_ply_header(data)
    expected_vertex_properties = (
        b"property float32 x\nproperty float32 y\nproperty float32 z\n"
    )
    if expected_vertex_properties not in header:
        raise ValueError("Reference mesh must store float32 x, y, z vertices")

    vertex_bytes = vertex_count * 3 * np.dtype("<f4").itemsize
    vertices = np.frombuffer(
        data,
        dtype="<f4",
        count=vertex_count * 3,
        offset=len(header),
    ).reshape(vertex_count, 3).astype(float)

    face_dtype = np.dtype([("count", "u1"), ("indices", "<u4", (3,))])
    faces = np.frombuffer(
        data,
        dtype=face_dtype,
        count=face_count,
        offset=len(header) + vertex_bytes,
    )
    if not np.all(faces["count"] == 3):
        raise ValueError("Reference mesh must contain triangle faces")
    return vertices, faces["indices"].astype(np.int64)


def _fit_floor_plane(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    center = np.median(points, axis=0)

    for _ in range(3):
        _, eigenvectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
        normal = eigenvectors[:, 0]
        distances = (points - center) @ normal
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        sigma = max(1.4826 * mad, 1e-9)
        inliers = np.abs(distances - median) < max(3.0 * sigma, 0.015)
        center = points[inliers].mean(axis=0)

    _, eigenvectors = np.linalg.eigh(np.cov(points[inliers] - center, rowvar=False))
    normal = eigenvectors[:, 0]
    distances = (points - center) @ normal
    if abs(np.quantile(distances, 0.001)) > abs(np.quantile(distances, 0.999)):
        normal = -normal
        distances = -distances
    floor_sigma = 1.4826 * float(np.median(np.abs(distances - np.median(distances))))
    return center, normal, floor_sigma


def _largest_elevated_component(
    faces: np.ndarray,
    heights: np.ndarray,
    threshold: float,
) -> np.ndarray:
    active = heights > threshold
    parent = np.arange(len(heights), dtype=np.int32)
    sizes = np.ones(len(heights), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if sizes[root_a] < sizes[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        sizes[root_a] += sizes[root_b]

    for triangle in faces:
        a, b, c = (int(value) for value in triangle)
        if active[a] and active[b]:
            union(a, b)
        if active[b] and active[c]:
            union(b, c)
        if active[c] and active[a]:
            union(c, a)

    active_indices = np.flatnonzero(active)
    if len(active_indices) == 0:
        raise ValueError("No vertices were found above the floor plane")
    roots = np.asarray([find(int(index)) for index in active_indices])
    unique_roots, counts = np.unique(roots, return_counts=True)
    largest_root = unique_roots[int(np.argmax(counts))]
    return active_indices[roots == largest_root]


def estimate_metric_alignment(
    vertices: np.ndarray,
    faces: np.ndarray,
    reference_height_m: float,
    top_quantile: float = 0.999,
) -> MetricAlignment:
    if reference_height_m <= 0:
        raise ValueError("reference_height_m must be positive")
    if not 0.9 <= top_quantile <= 1.0:
        raise ValueError("top_quantile must be between 0.9 and 1.0")

    floor_center, normal, floor_sigma = _fit_floor_plane(vertices)
    heights = (vertices - floor_center) @ normal
    threshold = max(10.0 * floor_sigma, 0.1 * float(np.max(heights)))
    object_indices = _largest_elevated_component(faces, heights, threshold)
    object_heights = heights[object_indices]
    model_height = float(np.quantile(object_heights, top_quantile))
    peak_height = float(np.max(object_heights))
    scale = reference_height_m / model_height

    x_axis = np.asarray([1.0, 0.0, 0.0])
    x_axis -= normal * float(x_axis @ normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.asarray([0.0, 1.0, 0.0])
        x_axis -= normal * float(x_axis @ normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.vstack([x_axis, y_axis, normal])

    object_center = np.median(vertices[object_indices], axis=0)
    floor_origin = object_center - normal * float((object_center - floor_center) @ normal)
    return MetricAlignment(
        floor_origin=floor_origin,
        rotation=rotation,
        scale_m_per_unit=scale,
        model_height=model_height,
        peak_height=peak_height,
        object_vertex_count=len(object_indices),
        floor_sigma=floor_sigma,
    )


def transform_binary_ply(
    input_path: str | Path,
    output_path: str | Path,
    alignment: MetricAlignment,
) -> Path:
    source = Path(input_path)
    data = source.read_bytes()
    header, vertex_count, _ = _read_ply_header(data)
    body = bytearray(data[len(header) :])
    vertices = np.frombuffer(body, dtype="<f4", count=vertex_count * 3).reshape(vertex_count, 3)
    transformed = alignment.transform(vertices).astype("<f4")
    vertices[:] = transformed

    comments = (
        b"comment Units meters\n"
        b"comment Floor aligned to Z=0\n"
        + f"comment MetricScale {alignment.scale_m_per_unit:.12g} meters_per_input_unit\n".encode()
    )
    output_header = header.replace(b"end_header\n", comments + b"end_header\n")
    target = Path(output_path)
    ensure_dir(target.parent)
    target.write_bytes(output_header + body)
    return target


def scale_meshes_to_metric(
    reference_mesh: str | Path,
    mesh_paths: list[str | Path],
    output_dir: str | Path,
    reference_height_m: float,
    report_path: str | Path,
    top_quantile: float = 0.999,
) -> tuple[MetricAlignment, list[Path]]:
    vertices, faces = load_triangle_mesh(reference_mesh)
    alignment = estimate_metric_alignment(
        vertices,
        faces,
        reference_height_m,
        top_quantile,
    )
    output_root = ensure_dir(output_dir)
    outputs = [
        transform_binary_ply(path, output_root / f"{Path(path).stem}_metric.ply", alignment)
        for path in mesh_paths
    ]
    report = {
        "reference_height_m": reference_height_m,
        "reference_height_mm": reference_height_m * 1000.0,
        "model_height_units": alignment.model_height,
        "connected_peak_height_units": alignment.peak_height,
        "top_quantile": top_quantile,
        "scale_m_per_input_unit": alignment.scale_m_per_unit,
        "object_vertex_count": alignment.object_vertex_count,
        "floor_sigma_input_units": alignment.floor_sigma,
        "floor_origin_input_units": alignment.floor_origin.tolist(),
        "rotation_to_z_up": alignment.rotation.tolist(),
        "outputs": [str(path) for path in outputs],
    }
    write_json(report_path, report)
    return alignment, outputs
