# Reconstruction Run Report: 2026-06-17

## Dataset

- Source: laptop webcam video `WIN_20260617_18_37_02_Pro.mp4`
- Resolution: 1280 x 720
- Extracted frames: 81
- Checkerboard calibration RMS error: 0.818 px
- Calibration images used: 17 of 95

## Sparse Model Comparison

| Metric | Session 001: independent cameras | Session 002: shared calibrated camera |
| --- | ---: | ---: |
| Camera models | 59 | 1 |
| Registered images | 59 / 81 (72.8%) | 59 / 81 (72.8%) |
| Sparse 3D points | 1,539 | 4,095 |
| Observations | 12,097 | 24,332 |
| Mean track length | 7.86 | 5.94 |
| Mean observations/image | 205.03 | 412.41 |
| Mean reprojection error | 1.320 px | 1.414 px |

## Interpretation

Session 001 used one camera model per registered image. That allowed bundle
adjustment to change focal length and distortion independently for every frame,
which is physically incorrect for a fixed webcam.

Session 002 used one shared OpenCV camera initialized from checkerboard
calibration. It recovered 2.66 times as many 3D points and 2.01 times as many
observations. Its slightly higher reprojection error is an honest tradeoff from
enforcing the real sensor constraint.

The refined shared camera remained close to calibration:

```text
fx=958.07, fy=960.63, cx=616.28, cy=383.79
```

Session 002 is the model to use for downstream dense reconstruction and project
reporting.

## Deployment Constraint

The installed Ubuntu COLMAP 3.7 build reports `without CUDA`, and the laptop
uses Intel Iris Xe graphics. COLMAP PatchMatch stereo therefore cannot run on
this machine. Image undistortion completed for all 59 registered images under:

```text
outputs/colmap/session_002/dense
```

Use the CPU build of OpenMVS for dense point-cloud and mesh reconstruction, or
move the prepared dense workspace to a CUDA-capable machine.

## CPU Dense Reconstruction

OpenMVS 2.4.0 completed dense reconstruction on the Windows laptop:

| Metric | Result |
| --- | ---: |
| Calibrated images | 59 |
| Estimated depth maps | 58 |
| Dense points before ROI trimming | 303,321 |
| Final dense points | 232,999 |
| Densification runtime | 13 min 14 sec |
| Peak densification memory | 1.17 GB |
| Mesh vertices | 57,198 |
| Mesh faces | 114,243 |
| Mesh reconstruction runtime | 5.4 sec |
| Texture atlas | 1 x 2048 px |
| Texture patches | 8,053 |
| Texturing runtime | 1 min 2 sec |

Final artifacts:

```text
outputs/openmvs/session_002/scene_dense.ply
outputs/openmvs/session_002/scene_dense_mesh.ply
outputs/openmvs/session_002/scene_dense_texture.ply
outputs/openmvs/session_002/scene_dense_texture0.png
```

This demonstrates a deployment-driven backend change: COLMAP provided robust
SfM and calibrated camera poses, while CPU OpenMVS replaced CUDA-only COLMAP
PatchMatch for dense geometry on Intel integrated graphics.
