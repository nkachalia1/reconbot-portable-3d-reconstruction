"""Download the latest Pi field recording, extract frames, and optionally run COLMAP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

from reconbot.frame_extraction import extract_frames
from reconbot.io_utils import load_intrinsics_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinator-url",
        default="http://pi5.local:5000",
        help="Base URL of the Raspberry Pi coordinator.",
    )
    parser.add_argument("--every-n", type=int, default=5)
    parser.add_argument("--min-blur", type=float, default=40.0)
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("configs/camera_intrinsics.yaml"),
        help="OpenCV calibration YAML used to seed COLMAP intrinsics.",
    )
    parser.add_argument(
        "--run-colmap",
        action="store_true",
        help="Run single-camera sparse reconstruction after extraction.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse an existing downloaded video and extracted frames, and rebuild "
            "incomplete COLMAP output for the same session."
        ),
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help=(
            "Reuse the downloaded video, replace extracted frames using the current "
            "sampling and blur settings, and rebuild incomplete COLMAP output."
        ),
    )
    return parser


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    environment = os.environ.copy()
    # Importing opencv-python points Qt at cv2's private plugins. COLMAP is also a
    # Qt application and must use the system plugins supplied by its own package.
    for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        environment.pop(variable, None)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.reextract:
        raise ValueError("Use either --resume or --reextract, not both")
    base_url = args.coordinator_url.rstrip("/")
    with urlopen(base_url + "/api/field/status", timeout=15.0) as response:
        status = json.loads(response.read().decode("utf-8"))
    session = status.get("session")
    if not session or not session.get("latest_video"):
        raise RuntimeError("The Pi coordinator has no completed field video")

    session_id = str(session["session_id"])
    video_path = Path("data/raw") / f"{session_id}.mp4"
    frame_dir = Path("data/frames") / session_id
    output_dir = Path("outputs/colmap") / session_id
    video_path.parent.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
    artifacts_exist = video_path.exists() or bool(existing_frames)
    if artifacts_exist and not (args.resume or args.reextract):
        raise RuntimeError(
            f"Local artifacts for {session_id} already exist; rerun with --resume "
            "or --reextract"
        )

    if args.reextract:
        if not video_path.exists():
            raise RuntimeError(f"Cannot re-extract {session_id}: local video is missing")
        if frame_dir.exists():
            print(f"Replacing extracted frames: {frame_dir}")
            shutil.rmtree(frame_dir)
        result = extract_frames(
            video_path,
            frame_dir,
            every_n=args.every_n,
            min_blur=args.min_blur,
            max_frames=args.max_frames,
        )
        written_frames = result.written_frames
        print(f"Video: {video_path}")
        print(f"Sharp frames: {written_frames}")
        print(f"Blur-rejected samples: {result.skipped_blurry_frames}")
    elif args.resume:
        if not video_path.exists() or not existing_frames:
            raise RuntimeError(
                f"Cannot resume {session_id}: both the video and extracted frames are required"
            )
        written_frames = len(existing_frames)
        print(f"Resuming {session_id} with existing local artifacts")
        print(f"Video: {video_path}")
        print(f"Sharp frames: {written_frames}")
    else:
        print(f"Downloading {session_id} from {base_url}")
        with urlopen(base_url + "/api/field/video.mp4", timeout=300.0) as response:
            with video_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        result = extract_frames(
            video_path,
            frame_dir,
            every_n=args.every_n,
            min_blur=args.min_blur,
            max_frames=args.max_frames,
        )
        written_frames = result.written_frames
        print(f"Video: {video_path}")
        print(f"Sharp frames: {written_frames}")
        print(f"Blur-rejected samples: {result.skipped_blurry_frames}")

    if not args.run_colmap:
        print("Next: rerun this command with --run-colmap.")
        return
    if written_frames < 20:
        raise RuntimeError("Fewer than 20 sharp frames were extracted; record a steadier orbit")

    if (args.resume or args.reextract) and output_dir.exists():
        print(f"Removing incomplete COLMAP output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "database.db"
    if database.exists():
        raise RuntimeError(f"COLMAP output already exists: {database}")

    feature_command = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(frame_dir),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera",
        "1",
    ]
    if args.intrinsics.exists():
        camera_matrix, distortion, _ = load_intrinsics_yaml(args.intrinsics)
        distortion_values = distortion.reshape(-1).tolist()
        while len(distortion_values) < 4:
            distortion_values.append(0.0)
        params = [
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
            *distortion_values[:4],
        ]
        feature_command.extend(
            ["--ImageReader.camera_params", ",".join(str(value) for value in params)]
        )
        print(f"Using calibrated intrinsics: {args.intrinsics}")
    _run(feature_command)
    _run(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.guided_matching",
            "1",
        ]
    )
    try:
        _run(
            [
                "colmap",
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(frame_dir),
                "--output_path",
                str(sparse_dir),
            ]
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "COLMAP could not initialize a connected sparse model. Confirm that the "
            "camera physically translated around a stationary, rigid, textured target; "
            "rotating a person or object in front of a fixed camera does not provide a "
            "valid SfM baseline."
        ) from error
    model = sparse_dir / "0"
    if model.exists():
        _run(["colmap", "model_analyzer", "--path", str(model)])
        print(f"Sparse reconstruction ready: {model}")
    else:
        raise RuntimeError("COLMAP did not produce sparse/0; inspect the mapper output")


if __name__ == "__main__":
    main()
