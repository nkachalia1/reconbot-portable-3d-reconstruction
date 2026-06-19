"""Command line interface for the portable reconstruction robot."""

from __future__ import annotations

import argparse
import sys

from ._optional import MissingDependencyError


def _add_common_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default="SIFT", choices=["SIFT", "ORB"])
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.75)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconbot")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Calibrate a monocular camera from checkerboard images.")
    calibrate.add_argument("--images", required=True)
    calibrate.add_argument("--board-cols", type=int, required=True, help="Inner checkerboard corners across.")
    calibrate.add_argument("--board-rows", type=int, required=True, help="Inner checkerboard corners down.")
    calibrate.add_argument("--square-size-m", type=float, required=True)
    calibrate.add_argument("--output", required=True)

    capture = sub.add_parser("capture", help="Capture a Pi/laptop camera image sequence.")
    capture.add_argument("--output", required=True)
    capture.add_argument("--camera-index", type=int, default=0)
    capture.add_argument("--count", type=int, default=30)
    capture.add_argument("--interval-s", type=float, default=1.0)
    capture.add_argument("--width", type=int)
    capture.add_argument("--height", type=int)
    capture.add_argument("--preview", action="store_true")

    extract = sub.add_parser("extract-frames", help="Extract sharp frames from a video.")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--every-n", type=int, default=10)
    extract.add_argument("--min-blur", type=float, default=0.0)
    extract.add_argument("--max-frames", type=int)

    benchmark = sub.add_parser("feature-benchmark", help="Compare SIFT and ORB on adjacent frames.")
    benchmark.add_argument("--images", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--methods", nargs="+", default=["SIFT", "ORB"], choices=["SIFT", "ORB"])
    benchmark.add_argument("--max-features", type=int, default=4000)
    benchmark.add_argument("--ratio", type=float, default=0.75)

    guidance = sub.add_parser("guidance", help="Suggest the next capture move from feature overlap.")
    guidance.add_argument("--image-a", required=True)
    guidance.add_argument("--image-b", required=True)
    guidance.add_argument("--method", default="SIFT", choices=["SIFT", "ORB"])
    guidance.add_argument("--max-features", type=int, default=4000)

    pair = sub.add_parser("pair-sfm", help="Run a two-view sparse reconstruction.")
    pair.add_argument("--image-a", required=True)
    pair.add_argument("--image-b", required=True)
    pair.add_argument("--intrinsics", required=True)
    pair.add_argument("--output", required=True)
    _add_common_feature_args(pair)

    metrics = sub.add_parser("pointcloud-metrics", help="Compute point cloud density and coverage metrics.")
    metrics.add_argument("--ply", required=True)
    metrics.add_argument("--voxel-size", type=float, default=0.05)
    metrics.add_argument("--output", required=True)

    view = sub.add_parser("view-cloud", help="Open a point cloud in Open3D.")
    view.add_argument("--ply", required=True)

    metric = sub.add_parser(
        "metric-scale",
        help="Scale binary PLY meshes from a known object height and align the floor to Z=0.",
    )
    metric.add_argument("--reference-mesh", required=True)
    metric.add_argument("--mesh", nargs="+", required=True)
    metric.add_argument("--reference-height-m", type=float, required=True)
    metric.add_argument("--top-quantile", type=float, default=0.999)
    metric.add_argument("--output-dir", required=True)
    metric.add_argument("--report", required=True)

    crop = sub.add_parser(
        "crop-mesh",
        help="Crop an OpenMVS binary triangle PLY to axis-aligned metric bounds.",
    )
    crop.add_argument("--input", required=True)
    crop.add_argument("--output", required=True)
    crop.add_argument("--min-xyz", nargs=3, type=float, required=True)
    crop.add_argument("--max-xyz", nargs=3, type=float, required=True)

    neural = sub.add_parser(
        "neural-reconstruct",
        help="Plan or run Nerfstudio Nerfacto, Instant-NGP, or Gaussian Splatting.",
    )
    neural.add_argument("--images", required=True)
    neural.add_argument("--colmap-model", required=True)
    neural.add_argument("--output-dir", required=True)
    neural.add_argument(
        "--method",
        choices=["nerfacto", "instant-ngp", "splatfacto"],
        default="nerfacto",
    )
    neural.add_argument("--max-iterations", type=int, default=10000)
    neural.add_argument(
        "--plan-only",
        action="store_true",
        help="Write reproducible commands without requiring CUDA or Nerfstudio.",
    )

    publish = sub.add_parser(
        "publish-mesh",
        help="Repair, validate, normalize, and export a mesh for printing/CAD/game use.",
    )
    publish.add_argument("--input", required=True)
    publish.add_argument("--output-dir", required=True)
    publish.add_argument(
        "--formats",
        nargs="+",
        default=["obj", "ply", "stl", "glb"],
        choices=["obj", "ply", "stl", "glb"],
    )
    publish.add_argument("--hole-size", type=float, default=1_000_000.0)
    publish.add_argument("--no-fill-holes", action="store_true")
    publish.add_argument("--allow-open", action="store_true")
    publish.add_argument("--scale-factor", type=float, default=1.0)
    publish.add_argument("--target-max-dimension-m", type=float)
    publish.add_argument("--keep-origin", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "calibrate":
            from .calibration import calibrate_and_save

            result = calibrate_and_save(
                args.images,
                args.board_cols,
                args.board_rows,
                args.square_size_m,
                args.output,
            )
            print(f"Wrote intrinsics: {args.output}")
            print(f"Used images: {len(result.used_images)}")
            print(f"Rejected images: {len(result.rejected_images)}")
            print(f"Reprojection error: {result.reprojection_error_px:.4f} px")
            return 0

        if args.command == "capture":
            from .capture import capture_sequence

            result = capture_sequence(
                args.output,
                args.camera_index,
                args.count,
                args.interval_s,
                args.width,
                args.height,
                args.preview,
            )
            print(f"Wrote {result.written_images} images to {result.output_dir}")
            return 0

        if args.command == "extract-frames":
            from .frame_extraction import extract_frames

            result = extract_frames(
                args.video,
                args.output,
                args.every_n,
                args.min_blur,
                args.max_frames,
            )
            print(f"Wrote frames: {result.written_frames}")
            print(f"Skipped blurry frames: {result.skipped_blurry_frames}")
            print(f"Total frames seen: {result.total_frames_seen}")
            return 0

        if args.command == "feature-benchmark":
            from .features import benchmark_sequence

            rows = benchmark_sequence(
                args.images,
                args.output,
                tuple(args.methods),
                args.max_features,
                args.ratio,
            )
            print(f"Wrote {len(rows)} benchmark rows to {args.output}")
            return 0

        if args.command == "guidance":
            from .active_guidance import analyze_pair

            result = analyze_pair(args.image_a, args.image_b, args.method, args.max_features)
            print(f"Matches: {result.match_count}")
            print(f"Geometric inlier ratio: {result.geometric_inlier_ratio:.3f}")
            print(result.recommendation)
            return 0

        if args.command == "pair-sfm":
            from .sfm import run_pair_sfm

            result = run_pair_sfm(
                args.image_a,
                args.image_b,
                args.intrinsics,
                args.output,
                args.method,
                args.max_features,
                args.ratio,
            )
            print(f"Wrote reconstruction to {result.output_dir}")
            print(f"Matches: {result.match_count}")
            print(f"Pose inliers: {result.pose_inlier_count}")
            print(f"3D points: {result.point_count}")
            print(f"Mean reprojection error: {result.mean_reprojection_error_px:.4f} px")
            return 0

        if args.command == "pointcloud-metrics":
            from .evaluation import point_cloud_metrics
            from .io_utils import load_ascii_ply_points, write_json

            points = load_ascii_ply_points(args.ply)
            report = point_cloud_metrics(points, args.voxel_size).to_dict()
            write_json(args.output, report)
            print(f"Wrote point cloud metrics to {args.output}")
            print(f"Points: {report['point_count']}")
            print(f"Voxel coverage ratio: {report['voxel_coverage_ratio']}")
            return 0

        if args.command == "view-cloud":
            from .visualization import view_point_cloud

            view_point_cloud(args.ply)
            return 0

        if args.command == "metric-scale":
            from .metric_scale import scale_meshes_to_metric

            alignment, outputs = scale_meshes_to_metric(
                args.reference_mesh,
                args.mesh,
                args.output_dir,
                args.reference_height_m,
                args.report,
                args.top_quantile,
            )
            print(f"Model height: {alignment.model_height:.6f} units")
            print(f"Metric scale: {alignment.scale_m_per_unit:.9f} m/unit")
            for output in outputs:
                print(f"Wrote metric mesh: {output}")
            print(f"Wrote scale report: {args.report}")
            return 0

        if args.command == "crop-mesh":
            from .mesh_crop import crop_binary_triangle_ply

            vertex_count, face_count = crop_binary_triangle_ply(
                args.input,
                args.output,
                tuple(args.min_xyz),
                tuple(args.max_xyz),
            )
            print(f"Wrote cropped mesh: {args.output}")
            print(f"Vertices: {vertex_count}")
            print(f"Faces: {face_count}")
            return 0

        if args.command == "neural-reconstruct":
            from .neural_reconstruction import (
                build_nerfstudio_plan,
                run_nerfstudio,
                write_neural_plan,
            )

            plan = build_nerfstudio_plan(
                args.images,
                args.colmap_model,
                args.output_dir,
                args.method,
                args.max_iterations,
            )
            plan_path = write_neural_plan(
                f"{args.output_dir}/neural_plan.json",
                plan,
            )
            print(f"Wrote neural plan: {plan_path}")
            if args.plan_only:
                return 0
            result = run_nerfstudio(plan, args.output_dir)
            print(f"Neural method: {result.method}")
            print(f"Export kind: {result.export_kind}")
            print(f"Primary artifact: {result.primary_artifact}")
            print(f"Report: {result.report}")
            return 0

        if args.command == "publish-mesh":
            from .mesh_postprocess import postprocess_mesh

            result = postprocess_mesh(
                args.input,
                args.output_dir,
                args.formats,
                fill_holes=not args.no_fill_holes,
                hole_size=args.hole_size,
                require_watertight=not args.allow_open,
                scale_factor=args.scale_factor,
                target_max_dimension_m=args.target_max_dimension_m,
                normalize_origin=not args.keep_origin,
            )
            print(f"Watertight: {result.after.watertight}")
            print(f"Boundary edges: {result.after.boundary_edges}")
            print(f"Non-manifold edges: {result.after.non_manifold_edges}")
            for output in result.exports:
                print(f"Wrote mesh: {output}")
            print(f"Wrote quality report: {result.report}")
            return 0

    except MissingDependencyError as exc:
        parser.exit(2, f"{exc}\n")

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
