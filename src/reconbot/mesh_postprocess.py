"""Mesh cleanup, topology validation, scale normalization, and publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ._optional import require_open3d
from .io_utils import ensure_dir, write_json


SUPPORTED_MESH_FORMATS = ("obj", "ply", "stl", "glb")


@dataclass(frozen=True)
class MeshTopologyReport:
    vertex_count: int
    face_count: int
    connected_components: int
    degenerate_faces: int
    boundary_edges: int
    non_manifold_edges: int
    edge_manifold: bool
    watertight: bool
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    extents: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MeshPublicationResult:
    output_dir: Path
    exports: tuple[Path, ...]
    report: Path
    before: MeshTopologyReport
    after: MeshTopologyReport
    scale_factor: float


def _face_components(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    parent = np.arange(len(faces), dtype=np.int64)
    sizes = np.ones(len(faces), dtype=np.int64)
    first_face_by_vertex: dict[int, int] = {}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a = find(left)
        b = find(right)
        if a == b:
            return
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]

    for face_index, face in enumerate(faces):
        for vertex in face:
            prior = first_face_by_vertex.setdefault(int(vertex), face_index)
            union(face_index, prior)
    return len({find(index) for index in range(len(faces))})


def analyze_mesh_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    area_epsilon: float = 1e-12,
) -> MeshTopologyReport:
    """Compute format-independent manufacturing checks from triangle arrays."""
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("Mesh contains no vertices")
    if len(triangles) and (np.min(triangles) < 0 or np.max(triangles) >= len(points)):
        raise ValueError("Mesh faces reference vertices outside the vertex array")

    if len(triangles):
        vectors_a = points[triangles[:, 1]] - points[triangles[:, 0]]
        vectors_b = points[triangles[:, 2]] - points[triangles[:, 0]]
        doubled_areas = np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)
        degenerate = int(np.count_nonzero(doubled_areas <= area_epsilon * 2.0))
        edges = np.vstack(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        )
        edges.sort(axis=1)
        _, incidence = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = int(np.count_nonzero(incidence == 1))
        non_manifold_edges = int(np.count_nonzero(incidence > 2))
    else:
        degenerate = 0
        boundary_edges = 0
        non_manifold_edges = 0

    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    edge_manifold = non_manifold_edges == 0
    return MeshTopologyReport(
        vertex_count=len(points),
        face_count=len(triangles),
        connected_components=_face_components(triangles),
        degenerate_faces=degenerate,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        edge_manifold=edge_manifold,
        watertight=bool(
            len(triangles)
            and degenerate == 0
            and boundary_edges == 0
            and edge_manifold
        ),
        bounds_min=tuple(float(value) for value in bounds_min),
        bounds_max=tuple(float(value) for value in bounds_max),
        extents=tuple(float(value) for value in bounds_max - bounds_min),
    )


def _legacy_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _clean_mesh(mesh, *, remove_non_manifold: bool = True) -> None:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if remove_non_manifold:
        mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        mesh.compute_vertex_normals()
        if mesh.is_orientable():
            mesh.orient_triangles()


def _fill_holes_open3d(mesh, hole_size: float):
    o3d = require_open3d()
    tensor_mesh_type = getattr(getattr(o3d, "t", None), "geometry", None)
    tensor_mesh_type = getattr(tensor_mesh_type, "TriangleMesh", None)
    if tensor_mesh_type is None or not hasattr(tensor_mesh_type, "from_legacy"):
        raise RuntimeError("This Open3D build does not provide tensor mesh hole filling")
    tensor_mesh = tensor_mesh_type.from_legacy(mesh)
    if not hasattr(tensor_mesh, "fill_holes"):
        raise RuntimeError("Upgrade Open3D to a build that provides TriangleMesh.fill_holes")
    return tensor_mesh.fill_holes(hole_size).to_legacy()


def _cap_boundary_cycles(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Close residual manifold boundary cycles with one centroid fan per cycle."""
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in triangles:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(start), int(end))))
            incidence.setdefault(key, []).append((int(start), int(end)))

    if any(len(orientations) > 2 for orientations in incidence.values()):
        raise RuntimeError("Cannot cap a mesh that still contains non-manifold edges")
    boundary = [orientations[0] for orientations in incidence.values() if len(orientations) == 1]
    if not boundary:
        return points.copy(), triangles.copy(), 0

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge_index, (start, end) in enumerate(boundary):
        adjacency.setdefault(start, []).append((edge_index, end))
        adjacency.setdefault(end, []).append((edge_index, start))
    odd_vertices = [vertex for vertex, edges in adjacency.items() if len(edges) % 2]
    if odd_vertices:
        raise RuntimeError(
            "Residual mesh boundary contains open chains rather than closed cycles; "
            "inspect it in MeshLab before fabrication."
        )

    unused = set(range(len(boundary)))
    closed_walks: list[list[int]] = []
    while unused:
        first_edge = min(unused)
        stack = [boundary[first_edge][0]]
        circuit: list[int] = []
        while stack:
            vertex = stack[-1]
            while adjacency[vertex] and adjacency[vertex][-1][0] not in unused:
                adjacency[vertex].pop()
            if adjacency[vertex]:
                edge_index, neighbor = adjacency[vertex].pop()
                unused.remove(edge_index)
                stack.append(neighbor)
            else:
                circuit.append(stack.pop())
        closed_walks.append(list(reversed(circuit)))

    cycles: list[list[int]] = []
    for walk in closed_walks:
        path: list[int] = []
        positions: dict[int, int] = {}
        for vertex in walk:
            if vertex not in positions:
                positions[vertex] = len(path)
                path.append(vertex)
                continue
            start = positions[vertex]
            cycle = path[start:] + [vertex]
            if len(cycle) >= 4:
                cycles.append(cycle)
            for removed in path[start + 1 :]:
                positions.pop(removed, None)
            path = path[: start + 1]

    if sum(len(cycle) - 1 for cycle in cycles) != len(boundary):
        raise RuntimeError("Could not decompose every residual boundary edge into cycles")

    new_points = points.tolist()
    new_faces = triangles.tolist()
    boundary_orientation = {
        tuple(sorted((start, end))): (start, end) for start, end in boundary
    }
    for cycle in cycles:
        vertices_in_cycle = cycle[:-1]
        center_index = len(new_points)
        new_points.append(points[vertices_in_cycle].mean(axis=0).tolist())
        for left, right in zip(cycle, cycle[1:]):
            start, end = boundary_orientation[tuple(sorted((left, right)))]
            # Reverse the existing boundary direction so adjacent faces have
            # opposite orientation along their shared edge.
            new_faces.append([end, start, center_index])
    return (
        np.asarray(new_points, dtype=float),
        np.asarray(new_faces, dtype=np.int64),
        len(cycles),
    )


def _cap_manifold_boundaries(mesh) -> int:
    o3d = require_open3d()
    vertices, faces = _legacy_arrays(mesh)
    colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
    capped_vertices, capped_faces, cap_count = _cap_boundary_cycles(vertices, faces)
    if not cap_count:
        return 0

    mesh.vertices = o3d.utility.Vector3dVector(capped_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(capped_faces)
    if colors is not None:
        # Every fan uses one shared center, colored from its boundary vertices.
        center_colors: list[np.ndarray] = []
        for center_index in range(len(vertices), len(capped_vertices)):
            touching = capped_faces[len(faces) :]
            touching = touching[np.any(touching == center_index, axis=1)]
            source = np.unique(touching[touching != center_index])
            center_colors.append(colors[source].mean(axis=0))
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.vstack([colors, np.asarray(center_colors)])
        )
    if mesh.has_triangle_uvs():
        mesh.triangle_uvs = o3d.utility.Vector2dVector()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return cap_count


def postprocess_mesh(
    input_path: str | Path,
    output_dir: str | Path,
    formats: Iterable[str] = SUPPORTED_MESH_FORMATS,
    *,
    fill_holes: bool = True,
    hole_size: float = 1_000_000.0,
    require_watertight: bool = True,
    scale_factor: float = 1.0,
    target_max_dimension_m: float | None = None,
    normalize_origin: bool = True,
) -> MeshPublicationResult:
    """Repair a triangle mesh and export fabrication and real-time formats."""
    o3d = require_open3d()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    requested_formats = tuple(dict.fromkeys(value.lower().lstrip(".") for value in formats))
    unsupported = sorted(set(requested_formats) - set(SUPPORTED_MESH_FORMATS))
    if unsupported:
        raise ValueError(f"Unsupported mesh formats: {', '.join(unsupported)}")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if target_max_dimension_m is not None and target_max_dimension_m <= 0:
        raise ValueError("target_max_dimension_m must be positive")

    mesh = o3d.io.read_triangle_mesh(str(source), enable_post_processing=True)
    if mesh.is_empty() or not len(mesh.triangles):
        raise ValueError(f"Could not read a triangle mesh from {source}")
    before = analyze_mesh_topology(*_legacy_arrays(mesh))

    _clean_mesh(mesh)
    repair_passes = 0
    fallback_caps = 0
    if fill_holes:
        for repair_passes in range(1, 4):
            mesh = _fill_holes_open3d(mesh, hole_size)
            # Removing non-manifold edges immediately after filling can reopen
            # otherwise closed boundary loops. First preserve the fill and only
            # remove offending edges when the measured topology requires it.
            _clean_mesh(mesh, remove_non_manifold=False)
            topology = analyze_mesh_topology(*_legacy_arrays(mesh))
            if topology.watertight:
                break
            if topology.non_manifold_edges:
                mesh.remove_non_manifold_edges()
                mesh.remove_unreferenced_vertices()
        topology = analyze_mesh_topology(*_legacy_arrays(mesh))
        if topology.boundary_edges and not topology.non_manifold_edges:
            fallback_caps = _cap_manifold_boundaries(mesh)

    effective_scale = float(scale_factor)
    extents = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=float)
    if target_max_dimension_m is not None:
        largest = float(np.max(extents))
        if largest <= 0:
            raise ValueError("Mesh has zero spatial extent")
        effective_scale *= target_max_dimension_m / (largest * effective_scale)
    mesh.scale(effective_scale, center=(0.0, 0.0, 0.0))

    if normalize_origin:
        bounds = mesh.get_axis_aligned_bounding_box()
        center = np.asarray(bounds.get_center(), dtype=float)
        minimum = np.asarray(bounds.get_min_bound(), dtype=float)
        mesh.translate((-center[0], -center[1], -minimum[2]))

    after = analyze_mesh_topology(*_legacy_arrays(mesh))
    if require_watertight and not after.watertight:
        raise RuntimeError(
            "Automatic repair did not produce a watertight mesh: "
            f"{after.boundary_edges} boundary edges and "
            f"{after.non_manifold_edges} non-manifold edges and "
            f"{after.degenerate_faces} degenerate faces remain. "
            "Increase --hole-size or inspect the mesh in MeshLab."
        )

    root = ensure_dir(output_dir).resolve()
    exports: list[Path] = []
    for extension in requested_formats:
        target = root / f"model.{extension}"
        success = o3d.io.write_triangle_mesh(
            str(target),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=True,
        )
        if not success or not target.is_file():
            raise RuntimeError(f"Open3D could not export {extension.upper()} to {target}")
        exports.append(target)

    report_path = root / "mesh_quality.json"
    write_json(
        report_path,
        {
            "input": str(source),
            "units": "meters" if target_max_dimension_m is not None else "input-units",
            "scale_factor": effective_scale,
            "fill_holes": fill_holes,
            "hole_size": hole_size,
            "repair_passes": repair_passes,
            "fallback_boundary_caps": fallback_caps,
            "normalize_origin": normalize_origin,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "exports": [str(path) for path in exports],
        },
    )
    return MeshPublicationResult(
        output_dir=root,
        exports=tuple(exports),
        report=report_path,
        before=before,
        after=after,
        scale_factor=effective_scale,
    )
