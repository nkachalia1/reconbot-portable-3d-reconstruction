"""Optional dependency helpers."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is needed but unavailable."""


def require_cv2():
    """Import OpenCV only when a command actually needs it."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "OpenCV is required for this command. Install with "
            "`python -m pip install -e .` or install `opencv-contrib-python`."
        ) from exc
    return cv2


def require_open3d():
    """Import Open3D only for visualization or mesh publication commands."""
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "Open3D is required for this command. Install with "
            "`python -m pip install -e .[mesh]`."
        ) from exc
    return o3d
