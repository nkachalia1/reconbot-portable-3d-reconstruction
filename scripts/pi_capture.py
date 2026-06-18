"""Thin wrapper for Raspberry Pi image capture.

Usage:
    python scripts/pi_capture.py --output data/raw/session_001 --count 60 --interval-s 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconbot.capture import capture_sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    result = capture_sequence(
        args.output,
        camera_index=args.camera_index,
        count=args.count,
        interval_s=args.interval_s,
        width=args.width,
        height=args.height,
        preview=args.preview,
    )
    print(f"Wrote {result.written_images} images to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
