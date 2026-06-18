# Codex Task Board

Use these as tickets for Codex. Each task has a bounded scope and a clear artifact.

## Phase 1: Calibration

Prompt:

```text
Add a calibration report command that reads camera_intrinsics.yaml and writes an HTML report with camera matrix, distortion coefficients, accepted images, rejected images, and per-image reprojection error plot.
```

Acceptance:

- Produces `outputs/<session>/calibration_report.html`.
- Includes reprojection error statistics.
- Handles missing images cleanly.

## Phase 2: Data Quality

Prompt:

```text
Add an image quality gate that rejects blurry, dark, or overexposed images before SfM. Save a CSV explaining why each frame was accepted or rejected.
```

Acceptance:

- Computes Laplacian blur score.
- Computes exposure histogram metrics.
- Produces `quality_report.csv`.

## Phase 3: Multi-View SfM

Prompt:

```text
Extend the two-view SfM module into an incremental SfM pipeline over an ordered frame sequence. Save camera poses, sparse tracks, and a PLY point cloud.
```

Acceptance:

- Uses calibrated intrinsics.
- Adds frames only when enough inlier matches exist.
- Saves camera trajectory JSON.

## Phase 4: COLMAP Metrics

Prompt:

```text
Implement a COLMAP sparse model parser for cameras, images, and points3D. Compute track length distribution, reprojection error, and point density.
```

Acceptance:

- Reads COLMAP text export.
- Produces `colmap_metrics.json`.
- Has unit tests on a tiny synthetic model fixture.

## Phase 5: Dashboard

Prompt:

```text
Build a Flask dashboard for the Pi showing latest captured image, feature overlap status, battery/runtime notes, and links to reconstruction artifacts.
```

Acceptance:

- Runs with `python -m reconbot.dashboard`.
- Uses generated JSON/CSV artifacts.
- Does not require Open3D on the Pi.
