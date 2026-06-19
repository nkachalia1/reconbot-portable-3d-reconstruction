from __future__ import annotations

import numpy as np

from reconbot.mesh_postprocess import _cap_boundary_cycles, analyze_mesh_topology


def test_tetrahedron_is_watertight():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=float,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=int,
    )

    report = analyze_mesh_topology(vertices, faces)

    assert report.watertight is True
    assert report.boundary_edges == 0
    assert report.non_manifold_edges == 0
    assert report.connected_components == 1


def test_open_square_reports_four_boundary_edges():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    report = analyze_mesh_topology(vertices, faces)

    assert report.watertight is False
    assert report.boundary_edges == 4
    assert report.non_manifold_edges == 0

    capped_vertices, capped_faces, cap_count = _cap_boundary_cycles(vertices, faces)
    capped = analyze_mesh_topology(capped_vertices, capped_faces)
    assert cap_count == 1
    assert capped.watertight is True
    assert capped.boundary_edges == 0


def test_non_manifold_edge_and_disconnected_component_are_counted():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [4, 0, 0],
            [3, 1, 0],
        ],
        dtype=float,
    )
    faces = np.array(
        [[0, 1, 2], [1, 0, 3], [0, 1, 4], [5, 6, 7]],
        dtype=int,
    )

    report = analyze_mesh_topology(vertices, faces)

    assert report.non_manifold_edges == 1
    assert report.edge_manifold is False
    assert report.connected_components == 2
    assert report.watertight is False


def test_touching_boundary_cycles_are_capped_independently():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=float,
    )
    # Two open triangles touch only at vertex zero, whose boundary degree is four.
    faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=int)

    capped_vertices, capped_faces, cap_count = _cap_boundary_cycles(vertices, faces)
    capped = analyze_mesh_topology(capped_vertices, capped_faces)

    assert cap_count == 2
    assert capped.boundary_edges == 0
    assert capped.non_manifold_edges == 0


def test_degenerate_face_is_reported():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    report = analyze_mesh_topology(vertices, np.array([[0, 1, 2]], dtype=int))

    assert report.degenerate_faces == 1
    assert report.extents == (2.0, 0.0, 0.0)
