"""Run the laptop reconstruction worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from reconbot.reconstruction_worker import WorkerConfig, create_reconstruction_app


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    default_openmvs = (
        Path.home()
        / "Downloads"
        / "OpenMVS_Windows_x64"
        / "vc17"
        / "x64"
        / "Release"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path(os.environ.get("RECONBOT_LIBRARY_ROOT", "data/reconstruction_library")),
    )
    parser.add_argument(
        "--openmvs-bin",
        type=Path,
        default=Path(os.environ.get("RECONBOT_OPENMVS_BIN", default_openmvs)),
    )
    parser.add_argument(
        "--wsl-distro",
        default=os.environ.get("RECONBOT_WSL_DISTRO", "Ubuntu-22.04"),
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=int(os.environ.get("RECONBOT_MAX_KEYFRAMES", "120")),
    )
    parser.add_argument(
        "--fallback-keyframes",
        type=int,
        default=int(os.environ.get("RECONBOT_FALLBACK_KEYFRAMES", "180")),
    )
    parser.add_argument(
        "--no-wsl-staging",
        action="store_true",
        help="Run COLMAP directly on the mounted Windows project instead of WSL storage.",
    )
    parser.add_argument("--keep-work", action="store_true")
    return parser


def create_app_from_args(args: argparse.Namespace):
    root = args.project_root.resolve()
    return create_reconstruction_app(
        WorkerConfig(
            project_root=root,
            library_root=(root / args.library_root).resolve()
            if not args.library_root.is_absolute()
            else args.library_root.resolve(),
            seed_root=root / "dashboard" / "public" / "reconstructions",
            intrinsics=root / "configs" / "camera_intrinsics.yaml",
            wsl_distro=args.wsl_distro,
            openmvs_bin=args.openmvs_bin.resolve(),
            max_frames=args.max_keyframes,
            fallback_max_frames=args.fallback_keyframes,
            use_wsl_staging=not args.no_wsl_staging,
            keep_work=args.keep_work,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    app = create_app_from_args(args)
    print(f"ReconBot reconstruction worker: http://{args.host}:{args.port}")
    print(f"OpenMVS: {args.openmvs_bin.resolve()}")
    print(f"Library: {args.library_root}")
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
