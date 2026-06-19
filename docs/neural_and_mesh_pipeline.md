# Neural Reconstruction And Fabrication Pipeline

## Purpose

ReconBot supports two complementary reconstruction paths from the same accepted
keyframes and calibrated COLMAP camera poses:

```text
field video -> quality gate -> COLMAP sparse model
                              |-> OpenMVS dense mesh (CPU default)
                              |-> Nerfacto / Instant-NGP field -> Poisson mesh
                              `-> Splatfacto -> Gaussian splat

triangle mesh -> cleanup -> hole filling -> topology checks -> scale normalization
              -> OBJ + PLY + STL + GLB + mesh_quality.json
```

OpenMVS remains the dependable path for the current Intel Iris Xe laptop.
Nerfacto, Instant-NGP, and Splatfacto require a CUDA-capable NVIDIA environment;
the adapter can be planned and tested without that hardware, but training cannot.

## Neural Backends

The neural adapter uses Nerfstudio so each method receives the same images and
COLMAP poses instead of rerunning camera estimation. This makes backend
comparisons meaningful and prevents neural training from silently using a
different input trajectory.

- `nerfacto`: high-quality NeRF pipeline with Poisson mesh export.
- `instant-ngp`: faster hash-grid NeRF pipeline with Poisson mesh export.
- `splatfacto`: 3D Gaussian Splatting for real-time novel-view rendering.

Gaussian splats are a rendering representation, not a watertight triangle mesh.
Use Splatfacto when visual fidelity and interactive rendering matter. Use a NeRF
mesh export or OpenMVS when CAD, collision geometry, or 3D printing is the goal.

Generate an auditable plan without CUDA:

```bash
reconbot neural-reconstruct \
  --images data/frames/field_test_3 \
  --colmap-model outputs/colmap/field_test_3/sparse/0 \
  --output-dir outputs/neural/field_test_3 \
  --method nerfacto \
  --plan-only
```

Run the plan in a CUDA-capable Nerfstudio environment by removing `--plan-only`.
Choose `instant-ngp` or `splatfacto` with `--method` to use the other adapters.
Every run writes `neural_plan.json`, and completed runs also write a report with
the selected method, commands, configuration, artifact path, and elapsed time.

## Field Dashboard Integration

Start the native Windows camera/worker with neural backends enabled and point it
at the Nerfstudio virtual environment inside WSL:

```powershell
$env:RECONBOT_ENABLE_NEURAL = "1"
$env:RECONBOT_NERFSTUDIO_ENV = "/home/USER/nerfstudio/.venv"
$env:RECONBOT_NEURAL_ITERATIONS = "10000"
.\.venv-win\Scripts\python scripts/laptop_camera_node.py
```

The Field tab only enables GPU options when the worker advertises them. OpenMVS
stays selected when neural support is unavailable. Splatfacto is intentionally a
CLI export until the web viewer supports native splat rendering.

## Watertight Mesh Publication

Install the mesh extras in the environment that runs publication:

```powershell
.\.venv-win\Scripts\python -m pip install -e ".[mesh,dashboard]"
$env:RECONBOT_ENABLE_MESH_EXPORTS = "1"
```

Or repair and publish an existing triangle mesh directly:

```bash
reconbot publish-mesh \
  --input outputs/model.ply \
  --output-dir outputs/fabrication/model \
  --formats obj ply stl glb \
  --target-max-dimension-m 0.6223
```

The publication stage performs duplicate and degenerate cleanup, removes
non-manifold edges, orients triangles, fills holes, normalizes scale and origin,
and then recomputes topology. Strict mode fails instead of labeling an open mesh
as watertight. Use `--allow-open` only for game or visualization assets where an
open surface is acceptable.

`mesh_quality.json` records before/after values for:

- vertex and triangle counts
- connected components
- degenerate faces
- boundary edges
- non-manifold edges
- watertight status
- bounds, extents, and applied scale factor

Fabrication-enabled field runs expose OBJ, PLY, STL, GLB, and the quality report
as downloads attached to that reconstruction's persistent history record.

## Scale Normalization

Monocular SfM has an arbitrary global scale. A watertight mesh is not metric
until a physical reference is supplied. Use either:

- `--scale-factor` when a calibrated meters-per-SfM-unit factor already exists.
- `--target-max-dimension-m` when a known dimension corresponds to the mesh's
  largest bounding-box extent.

For irregular scenes, use `reconbot metric-scale` or a measured pair of landmarks
instead of assuming the largest extent is the known object dimension.

## Evaluation Matrix

Evaluate all backends on the same held-out field recording and report:

| Metric | OpenMVS | Nerfacto | Instant-NGP | Splatfacto |
| --- | --- | --- | --- | --- |
| Registered images / reprojection error | Yes | Shared COLMAP result | Shared COLMAP result | Shared COLMAP result |
| Training or dense runtime | Yes | Yes | Yes | Yes |
| Mesh vertices / faces | Yes | Yes | Yes | Not intrinsic |
| Boundary / non-manifold edges | Yes | Yes | Yes | Not intrinsic |
| Watertight result | Yes | Yes | Yes | Not intrinsic |
| Novel-view quality | Optional | PSNR / SSIM / LPIPS | PSNR / SSIM / LPIPS | PSNR / SSIM / LPIPS |

This comparison turns the new methods into a quantitative robotics experiment,
not merely a second renderer.
