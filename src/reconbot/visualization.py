"""Optional Open3D visualization helpers."""

from __future__ import annotations

from pathlib import Path

from ._optional import require_open3d


def view_point_cloud(path: str | Path) -> None:
    o3d = require_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    o3d.visualization.draw_geometries([cloud])
