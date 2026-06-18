"""Summarize feature and verified-match connectivity in a COLMAP database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    connection = sqlite3.connect(args.database)
    try:
        image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        keypoint_rows = [
            row[0] for row in connection.execute("SELECT rows FROM keypoints WHERE rows > 0")
        ]
        verified_rows = [
            row[0]
            for row in connection.execute(
                "SELECT rows FROM two_view_geometries WHERE rows > 0"
            )
        ]
    finally:
        connection.close()

    print(f"Database: {args.database}")
    print(f"Images: {image_count}")
    if keypoint_rows:
        print(
            "Features per image: "
            f"median={np.median(keypoint_rows):.0f}, "
            f"min={np.min(keypoint_rows)}, max={np.max(keypoint_rows)}"
        )
    print(f"Verified image pairs: {len(verified_rows)}")
    if not verified_rows:
        return

    values = np.asarray(verified_rows)
    print(
        "Verified correspondences per pair: "
        f"median={np.median(values):.0f}, p90={np.percentile(values, 90):.0f}, "
        f"max={np.max(values)}"
    )
    for threshold in (15, 30, 50, 100):
        count = int(np.count_nonzero(values >= threshold))
        print(f"Pairs with >= {threshold:>3} correspondences: {count}")


if __name__ == "__main__":
    main()
