# COLMAP Pipeline

This repo includes an educational two-view SfM baseline. For interview-grade dense reconstructions, run COLMAP on the laptop after collecting images.

## Folder

```text
data/frames/session_001/
outputs/colmap/session_001/
```

## Sparse Reconstruction

For frames extracted from one fixed webcam, all images must share one camera.
Using COLMAP's default `single_camera=0` creates an independent intrinsic model
for every frame and can overfit the reconstruction.

```bash
mkdir -p outputs/colmap/session_001/sparse
colmap feature_extractor \
  --database_path outputs/colmap/session_001/database.db \
  --image_path data/frames/session_001 \
  --ImageReader.camera_model OPENCV \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "fx,fy,cx,cy,k1,k2,p1,p2"

colmap exhaustive_matcher \
  --database_path outputs/colmap/session_001/database.db \
  --SiftMatching.guided_matching 1

colmap mapper \
  --database_path outputs/colmap/session_001/database.db \
  --image_path data/frames/session_001 \
  --output_path outputs/colmap/session_001/sparse
```

Read `fx`, `fy`, `cx`, `cy`, `k1`, `k2`, `p1`, and `p2` from
`configs/camera_intrinsics.yaml`. For an uncalibrated camera, omit
`camera_params`, but keep `single_camera=1`.

Validate the result:

```bash
colmap model_analyzer --path outputs/colmap/session_001/sparse/0
```

For a single-webcam recording, `Cameras` should be `1`. Also record registered
image ratio, point count, observation count, mean track length, and mean
reprojection error.

On Windows PowerShell, use `scripts/run_colmap_sparse.ps1`:

```powershell
.\scripts\run_colmap_sparse.ps1 -ImagePath data/frames/session_001 -OutputPath outputs/colmap/session_001
```

## Dense Reconstruction

```bash
mkdir -p outputs/colmap/session_001/dense
colmap image_undistorter \
  --image_path data/frames/session_001 \
  --input_path outputs/colmap/session_001/sparse/0 \
  --output_path outputs/colmap/session_001/dense \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path outputs/colmap/session_001/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path outputs/colmap/session_001/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path outputs/colmap/session_001/dense/fused.ply
```

COLMAP's PatchMatch stereo requires a CUDA-enabled build. The Ubuntu 22.04
COLMAP 3.7 package is built without CUDA. On a machine without an NVIDIA GPU,
use OpenMVS as the CPU dense-reconstruction backend after running
`image_undistorter`:

```bash
mkdir -p outputs/openmvs/session_001
cd outputs/openmvs/session_001
InterfaceCOLMAP \
  -i ../../colmap/session_001/dense \
  -o scene.mvs \
  --image-folder ../../colmap/session_001/dense/images
DensifyPointCloud scene.mvs
ReconstructMesh scene_dense.mvs -p scene_dense.ply
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply
```

## What To Save

- Sparse model screenshots.
- `fused.ply` dense cloud.
- A short screen recording rotating the reconstruction.
- Feature benchmark CSV from this repo.
- Calibration YAML.
- Point cloud metrics JSON.

## Interview Talking Point

OpenCV code shows that you understand the geometry. COLMAP shows that you can integrate a mature robotics/vision tool when reliability matters.
