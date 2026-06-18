"""Run the Raspberry Pi field coordinator and dashboard server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from reconbot.coordinator import (
    HttpCameraClient,
    HttpReconstructionClient,
    create_coordinator_app,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--camera-url",
        default=os.environ.get("RECONBOT_CAMERA_URL", "http://127.0.0.1:5001"),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("RECONBOT_TOKEN"),
        help="Shared camera-node token; defaults to RECONBOT_TOKEN.",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path(os.environ.get("RECONBOT_SESSION_ROOT", "data/field_sessions")),
    )
    parser.add_argument(
        "--dashboard-dist",
        type=Path,
        default=Path(os.environ.get("RECONBOT_DASHBOARD_DIST", "dashboard/dist")),
    )
    parser.add_argument(
        "--reconstruction-url",
        default=os.environ.get("RECONBOT_RECONSTRUCTION_URL"),
        help="Laptop reconstruction worker URL, for example http://10.0.0.198:5002.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = HttpCameraClient(args.camera_url, token=args.token)
    reconstruction_client = (
        HttpReconstructionClient(args.reconstruction_url)
        if args.reconstruction_url
        else None
    )
    app = create_coordinator_app(
        camera_client=client,
        session_root=args.session_root,
        dashboard_dist=args.dashboard_dist,
        reconstruction_client=reconstruction_client,
    )
    print(f"ReconBot Pi coordinator: http://{args.host}:{args.port}")
    print(f"Camera node: {args.camera_url}")
    print(f"Reconstruction worker: {args.reconstruction_url or 'not configured'}")
    print(f"Session storage: {args.session_root.resolve()}")
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
