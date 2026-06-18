"""Run the laptop webcam as the ReconBot camera node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading

from reconbot.camera_node import (
    DirectoryFrameSource,
    OpenCVCameraSource,
    create_camera_app,
)
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
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read images from a directory instead of opening a webcam.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("RECONBOT_TOKEN"),
        help="Shared API token; defaults to RECONBOT_TOKEN.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=Path("data/camera_recordings"),
        help="Directory for MP4 recordings made by the field dashboard.",
    )
    parser.add_argument(
        "--no-reconstruction-worker",
        action="store_true",
        help="Run only the webcam service and leave port 5002 disabled.",
    )
    parser.add_argument("--reconstruction-port", type=int, default=5002)
    parser.add_argument(
        "--openmvs-bin",
        type=Path,
        default=Path(os.environ.get("RECONBOT_OPENMVS_BIN", default_openmvs)),
    )
    parser.add_argument(
        "--wsl-distro",
        default=os.environ.get("RECONBOT_WSL_DISTRO", "Ubuntu-22.04"),
    )
    parser.set_defaults(project_root=project_root)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source_dir:
        source = DirectoryFrameSource(args.source_dir)
        source_label = f"simulation directory {args.source_dir}"
    else:
        source = OpenCVCameraSource(args.camera_index, args.width, args.height)
        source_label = f"webcam index {args.camera_index}"

    app = create_camera_app(source, args.token, args.recording_dir)
    if not args.no_reconstruction_worker:
        root = args.project_root.resolve()
        worker = create_reconstruction_app(
            WorkerConfig(
                project_root=root,
                library_root=root / "data" / "reconstruction_library",
                seed_root=root / "dashboard" / "public" / "reconstructions",
                intrinsics=root / "configs" / "camera_intrinsics.yaml",
                wsl_distro=args.wsl_distro,
                openmvs_bin=args.openmvs_bin.resolve(),
            )
        )
        thread = threading.Thread(
            target=lambda: worker.run(
                host=args.host,
                port=args.reconstruction_port,
                threaded=True,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="reconbot-reconstruction-worker",
        )
        thread.start()
        print(
            "ReconBot reconstruction worker listening on "
            f"http://{args.host}:{args.reconstruction_port}"
        )
    print(f"ReconBot camera node using {source_label}")
    print(f"Listening on http://{args.host}:{args.port}")
    try:
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    finally:
        app.extensions["reconbot_recorder"].close()
        source.close()


if __name__ == "__main__":
    main()
