"""Report sharpness statistics for sampled frames in a field video."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reconbot._optional import require_cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every-n", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.every_n <= 0:
        raise ValueError("--every-n must be positive")

    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    scores: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        frame_index += 1
    capture.release()

    if not scores:
        raise RuntimeError("No frames were decoded")

    values = np.asarray(scores)
    print(f"Video: {args.video}")
    print(f"Decoded frames: {frame_index}")
    print(f"Sampled every {args.every_n}: {len(scores)} frames")
    print("Blur-score percentiles:")
    for percentile in (0, 10, 25, 50, 75, 90, 100):
        print(f"  p{percentile:>3}: {np.percentile(values, percentile):.2f}")
    print("Frames retained by threshold:")
    for threshold in (10, 15, 20, 25, 30, 35, 40):
        retained = int(np.count_nonzero(values >= threshold))
        print(f"  >= {threshold:>2}: {retained:>3} / {len(scores)}")


if __name__ == "__main__":
    main()
