from __future__ import annotations

from pathlib import Path

import numpy as np

from reconbot.frame_extraction import extract_adaptive_keyframes
from reconbot.reconstruction_worker import _write_loop_pairs


def test_adaptive_keyframes_cover_full_video_and_cap_redundancy(tmp_path: Path):
    import cv2

    video = tmp_path / "orbit.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    for index in range(100):
        shift = (index // 3) * 2
        frame = np.roll(texture, shift, axis=1)
        writer.write(frame)
    writer.release()

    result = extract_adaptive_keyframes(
        video,
        tmp_path / "frames",
        every_n=1,
        min_blur=5,
        max_frames=20,
        min_motion_px=3,
        max_gap_s=0.4,
    )

    assert result.written_frames == 20
    assert result.selected_source_indices[0] == 0
    assert result.selected_source_indices[-1] == 99
    assert result.skipped_redundant_frames > 0
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 20


def test_loop_pairs_connect_end_of_orbit_to_start(tmp_path: Path):
    images = [tmp_path / f"frame_{index:06d}.jpg" for index in range(40)]
    pair_file = tmp_path / "pairs.txt"

    count = _write_loop_pairs(images, pair_file, window=5)
    lines = pair_file.read_text(encoding="utf-8").splitlines()

    assert count == 25
    assert "frame_000000.jpg frame_000039.jpg" in lines
    assert all("frame_00000" in line.split()[0] for line in lines)
