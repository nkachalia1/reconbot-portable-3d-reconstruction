"""Run both ReconBot nodes locally with a recorded image sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

from werkzeug.serving import make_server

from reconbot.camera_node import DirectoryFrameSource, create_camera_app
from reconbot.coordinator import HttpCameraClient, create_coordinator_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/frames/session_003"),
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("data/field_sessions/demo"),
    )
    parser.add_argument(
        "--dashboard-dist",
        type=Path,
        default=Path("dashboard/dist"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5001)
    parser.add_argument("--coordinator-port", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = DirectoryFrameSource(args.source_dir)
    camera_app = create_camera_app(source, recording_dir=args.session_root / "camera")
    camera_server = make_server(args.host, args.camera_port, camera_app, threaded=True)
    camera_thread = threading.Thread(target=camera_server.serve_forever, daemon=True)
    camera_thread.start()

    camera_url = f"http://{args.host}:{args.camera_port}"
    coordinator_app = create_coordinator_app(
        HttpCameraClient(camera_url),
        args.session_root,
        args.dashboard_dist,
    )
    coordinator_server = make_server(
        args.host,
        args.coordinator_port,
        coordinator_app,
        threaded=True,
    )
    print(f"Recorded camera node: {camera_url}")
    print(f"Field dashboard: http://{args.host}:{args.coordinator_port}")
    print("Press Ctrl+C to stop both nodes.")
    try:
        coordinator_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator_server.shutdown()
        camera_server.shutdown()
        camera_app.extensions["reconbot_recorder"].close()
        source.close()


if __name__ == "__main__":
    main()
