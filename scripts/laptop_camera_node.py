"""Run the laptop webcam as the ReconBot camera node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from reconbot.camera_node import (
    DirectoryFrameSource,
    OpenCVCameraSource,
    create_camera_app,
)


def build_parser() -> argparse.ArgumentParser:
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
