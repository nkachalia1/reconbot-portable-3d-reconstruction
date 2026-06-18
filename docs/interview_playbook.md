# Interview Playbook

## Thirty-Second Pitch

I built a portable monocular 3D reconstruction system with a Raspberry Pi 5
field coordinator and a networked laptop webcam node. The live system rejects
blurry or poorly exposed frames, measures ORB overlap, and gives active
viewpoint guidance while persisting session telemetry. Offline, I calibrated
the camera, recovered a 173-view trajectory in COLMAP, generated a four-million
point dense cloud, and scaled the final mesh to metric units using a measured
reference.

## Architecture

```text
Laptop webcam -> camera service -> Wi-Fi -> Pi coordinator -> field dataset
                                                     |
                                                     v
                                               live dashboard

field dataset -> COLMAP SfM -> OpenMVS dense cloud/mesh -> metric scaling
```

## Best Technical Details To Explain

- Monocular reconstruction recovers scale only up to an unknown factor.
- Calibration matters because biased intrinsics distort pose and triangulation.
- Essential matrix estimation uses calibrated correspondences and RANSAC.
- Triangulation quality depends on both feature accuracy and camera baseline.
- Inlier ratio is a useful practical metric when ground-truth matches are unavailable.
- The Pi/laptop split is a systems decision driven by power and compute constraints.
- Active perception converts match count, geometric inliers, blur, and image
  motion into a next-viewpoint recommendation.
- Session metadata records accepted/rejected frames and node telemetry, making
  deployment failures measurable instead of anecdotal.

## Strong Demo Artifacts

- `configs/camera_intrinsics.yaml`
- `outputs/<session>/features.csv`
- `outputs/<session>/pair_sfm_metrics.json`
- `outputs/<session>/points.ply`
- Dense COLMAP `fused.ply`
- Screenshot of point cloud with camera trajectory
- `data/field_sessions/<session>/captures.jsonl`
- Live Field dashboard showing guidance, latency, temperature, and disk budget

## Likely Interview Questions

What is the difference between fundamental and essential matrices?

Answer: the fundamental matrix relates pixel coordinates between two uncalibrated views, while the essential matrix relates normalized calibrated camera coordinates. With known intrinsics, the essential matrix can be decomposed into relative rotation and translation direction.

Why is monocular scale ambiguous?

Answer: if all 3D points and camera translations are scaled by the same factor, the projected image points do not change. A known object size, measured baseline, IMU, wheel odometry, or depth sensor can resolve scale.

Why use COLMAP if you wrote OpenCV SfM code?

Answer: the OpenCV implementation demonstrates the geometry and gives controllable metrics. COLMAP is the robust production tool for bundle adjustment and dense MVS, so integrating it is the practical engineering choice.

What failed?

Answer with a real experiment: glossy objects, blank walls, motion blur, too little baseline, or repeated texture. Then explain the metric that exposed it, such as low inlier ratio or high reprojection error.

Why use two computers?

Answer: the available sensor was a built-in laptop webcam, so I exposed it as a
small authenticated camera service. The Pi remains the portable coordinator
and system-of-record, while dense MVS runs offline on the laptop. That boundary
also let me test timeout, storage, telemetry, and operator-feedback behavior.
