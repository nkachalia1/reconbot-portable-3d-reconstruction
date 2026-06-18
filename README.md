# Portable Monocular 3D Reconstruction Robot

A battery-powered Raspberry Pi 5 perception system that captures monocular images, estimates camera motion, reconstructs sparse and dense 3D structure, and reports quantitative reconstruction quality.

This project is intentionally framed as a robotics/computer vision system, not a one-off OpenCV demo. It connects calibration, feature tracking, pose estimation, mapping, deployment constraints, and evaluation.

## Hardware

- Raspberry Pi 5, 8 GB
- 20,000 mAh USB-C power bank
- Laptop webcam or USB webcam
- Laptop for heavier Python/OpenCV/COLMAP processing

## What This Demonstrates

- Computer vision: calibration, feature extraction, matching, epipolar geometry
- 3D reconstruction: essential matrix, pose recovery, triangulation, point clouds
- Robotics systems engineering: field capture pipeline, Pi deployment, status artifacts
- Real-world constraints: battery, camera blur, storage, compute split between Pi and laptop
- Quantitative evaluation: reprojection error, inlier ratio, point count, density, coverage

## Repo Layout

```text
src/reconbot/              Python package and CLI
scripts/                   Pi and laptop helper entry points
docs/                      runbooks, evaluation plan, interview framing
data/calibration/          checkerboard images
data/raw/                  raw field captures or videos
data/frames/               extracted image sequences
outputs/                   generated metrics, point clouds, reports
configs/                   camera intrinsics and project config
tests/                     NumPy-only tests for core geometry
```

## Setup

Create a virtual environment on the laptop:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .[viz,dashboard,dev]
```

On a Raspberry Pi, use the lighter install first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If OpenCV wheels are slow on the Pi, install OpenCV through apt and then install this package:

```bash
sudo apt update
sudo apt install -y python3-opencv
python -m pip install -e . --no-deps
```

## First Milestone

1. Print a checkerboard and collect 20 to 40 images into `data/calibration/`.
2. Calibrate the camera:

```powershell
reconbot calibrate --images data/calibration --board-cols 9 --board-rows 6 --square-size-m 0.024 --output configs/camera_intrinsics.yaml
```

3. Capture or extract frames:

```powershell
reconbot extract-frames --video data/raw/chair_walkaround.mp4 --output data/frames/chair --every-n 8 --min-blur 40
```

4. Benchmark features:

```powershell
reconbot feature-benchmark --images data/frames/chair --output outputs/chair_feature_benchmark.csv
```

5. Build a two-view sparse reconstruction:

```powershell
reconbot pair-sfm --image-a data/frames/chair/frame_000000.jpg --image-b data/frames/chair/frame_000008.jpg --intrinsics configs/camera_intrinsics.yaml --output outputs/chair_pair
```

6. Evaluate the generated cloud:

```powershell
reconbot pointcloud-metrics --ply outputs/chair_pair/points.ply --voxel-size 0.05 --output outputs/chair_pair/pointcloud_metrics.json
```

## COLMAP Dense Reconstruction

Use this repo for capture, calibration, feature diagnostics, and metrics. Use COLMAP for production-grade sparse and dense reconstruction:

```bash
colmap feature_extractor --database_path outputs/colmap/database.db --image_path data/frames/chair
colmap exhaustive_matcher --database_path outputs/colmap/database.db
mkdir -p outputs/colmap/sparse
colmap mapper --database_path outputs/colmap/database.db --image_path data/frames/chair --output_path outputs/colmap/sparse
```

See [docs/colmap_pipeline.md](docs/colmap_pipeline.md) for the full laptop workflow.

## Resume Bullets

- Designed a portable Raspberry Pi perception pipeline for monocular 3D scene reconstruction under battery, compute, storage, and motion blur constraints.
- Implemented calibration, feature benchmarking, epipolar pose estimation, triangulation, and point cloud quality metrics using Python, NumPy, OpenCV, and COLMAP.
- Reconstructed indoor scenes from handheld monocular image sequences and evaluated results with reprojection error, geometric inlier ratio, point density, and coverage metrics.
- Built a field-to-laptop workflow that separates lightweight robotic data collection from heavier offline reconstruction and visualization.

## How To Use Codex On This Project

Treat Codex like a junior robotics engineer. Give it scoped tickets with acceptance criteria:

- "Add a calibration report plot showing per-image reprojection error."
- "Implement image quality gating for blur and exposure before frames are admitted to a dataset."
- "Add a COLMAP model parser that computes track length and reprojection error from sparse reconstructions."
- "Build a Flask dashboard that shows live captures, feature overlap, and latest reconstruction metrics."
- "Create tests for triangulation, projection, and pose composition."

The project becomes interview-worthy when every module produces artifacts you can show: YAML intrinsics, CSV benchmarks, JSON metrics, PLY point clouds, screenshots, and a short engineering writeup.

## Reconstruction Dashboard

The React/Three.js dashboard under `dashboard/` provides:

- Interactive textured reconstructions with orbit, grid, wireframe, reset, and
  fullscreen controls.
- A persistent reconstruction history that swaps the active model, source
  video, and quantitative metrics together.
- Field recording controls that can submit the saved MP4 directly to the
  laptop reconstruction worker.
- Session 002 versus Session 003 quantitative evaluation.
- End-to-end system execution trace and deployment constraints.
- Embedded portfolio rotation clip.

Build and run:

```powershell
cd dashboard
npm install
node scripts/convert-textured-ply.mjs
npm run build
npm run preview
```

Open `http://127.0.0.1:4173`.

Automated verification:

```powershell
node scripts/verify-dashboard.mjs
```

The verifier exercises all views at desktop and mobile sizes, captures
screenshots under `outputs/dashboard/`, and performs a nonblank WebGL canvas
pixel check.

## Two-Node Field System

The deployed system uses the laptop webcam as a network sensor, the Raspberry
Pi as the field coordinator, and a laptop worker for heavy reconstruction:

```text
Laptop camera (:5001) ------> Raspberry Pi coordinator/dashboard (:5000)
Laptop reconstruction (:5002) <- video + job requests -----------+
```

Run the recorded-data simulation from Ubuntu/WSL:

```bash
python3 -m pip install -e ".[dashboard,dev]"
python3 scripts/local_field_demo.py
```

Then open `http://127.0.0.1:5000` and select **Field**. For real hardware, run
`scripts/laptop_camera_node.py` in native Windows Python and
`scripts/pi_coordinator.py` on the Pi. WSL is suitable for reconstruction and
simulation, but it does not normally expose the built-in laptop webcam. See
[docs/hardware_runbook.md](docs/hardware_runbook.md) for the Windows
environment, network, firewall, systemd, and field-capture instructions.

The native Windows laptop command starts both the camera service and the
reconstruction worker. The worker drives COLMAP in WSL and the Windows OpenMVS
CPU executables:

```powershell
$env:RECONBOT_TOKEN = "replace-with-a-long-random-token"
$env:RECONBOT_OPENMVS_BIN = "$HOME\Downloads\OpenMVS_Windows_x64\vc17\x64\Release"
.\.venv-win\Scripts\python scripts/laptop_camera_node.py
```

Configure the Pi with both laptop endpoints:

```bash
export RECONBOT_CAMERA_URL=http://LAPTOP_IPV4:5001
export RECONBOT_RECONSTRUCTION_URL=http://LAPTOP_IPV4:5002
python scripts/pi_coordinator.py
```

After recording, click **Reconstruct video** in the Field tab. Progress is
reported for quality gating, sparse SfM, dense MVS, meshing, texturing, and GLB
publishing. Successful runs are added to the front-page history automatically;
deleting a history entry removes both its saved video and model.
