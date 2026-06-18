# Field Reconstruction Report: field_test_2

## Run Summary

- Capture mode: Raspberry Pi coordinator with laptop webcam camera node over Wi-Fi
- Session ID: `field_test_2`
- Source video: `data/raw/field_test_2.mp4`
- Video frames decoded: 490
- Extracted sharp frames: 103
- Sampling interval: every 3 video frames
- Blur threshold: 15
- Reconstruction target: stationary textured object, with the camera physically moved in an orbit

This was the first successful end-to-end field capture using the Pi-hosted
workflow instead of manually importing a local video.

## Deployment Constraint

The previous field attempt, `field_test_1`, failed because the camera was fixed
while the subject moved. Monocular SfM requires camera translation around a
mostly rigid, stationary scene. For `field_test_2`, the capture procedure was
changed so the target stayed still and the camera moved around it.

## Frame Quality

| Metric | Result |
| --- | ---: |
| Decoded video frames | 490 |
| Sampled frames | 164 |
| Retained frames at blur >= 15 | 103 |
| Median blur score | 17.83 |
| 90th percentile blur score | 46.18 |

An earlier threshold of 40 retained only 14 frames, which was too strict for the
webcam footage. Lowering the threshold to 15 preserved enough usable overlap
while still rejecting the blurriest frames.

## Feature Matching

| Metric | Result |
| --- | ---: |
| Images in COLMAP database | 103 |
| Median features per image | 1,031 |
| Feature range | 606 to 1,939 |
| Verified image pairs | 2,639 |
| Median verified correspondences/pair | 122 |
| 90th percentile correspondences/pair | 350 |
| Pairs with >= 100 correspondences | 1,609 |

## Sparse Reconstruction

| Metric | Result |
| --- | ---: |
| Registered images | 100 / 103 |
| Sparse 3D points | 11,560 |
| Observations | 66,148 |
| Mean track length | 5.72 |
| Mean observations/image | 661.48 |
| Mean reprojection error | 1.195 px |

The registered camera poses form a coherent arc around the target. Some
background and outlier points remain, but the trajectory and central structure
are consistent enough for dense reconstruction.

## Dense Reconstruction

| Metric | Result |
| --- | ---: |
| Fused depth maps | 98 |
| Raw fused depths | 42,155,201 |
| Dense points before ROI trimming | 642,301 |
| Final dense points | 395,093 |
| Densification runtime | 9 min 8 sec |
| Peak densification memory | 2.01 GB |

The densification job was safely paused after completed depth maps had been
written, then resumed. OpenMVS reused cached `.dmap` files and completed without
restarting from zero.

## Mesh And Texture

| Metric | Result |
| --- | ---: |
| Mesh vertices | 85,215 |
| Mesh faces | 170,169 |
| Mesh runtime | 21 sec |
| Peak mesh memory | 443 MB |
| Texture patches | 12,593 |
| Texture atlas | 1 x 2048 px |
| Texturing runtime | 2 min 10 sec |
| Peak texturing memory | 784 MB |

Final artifacts:

```text
outputs/openmvs/field_test_2/scene_dense.ply
outputs/openmvs/field_test_2/scene_dense_mesh.ply
outputs/openmvs/field_test_2/scene_dense_texture.ply
outputs/openmvs/field_test_2/scene_dense_texture0.png
dashboard/public/assets/reconstruction.glb
dashboard/public/assets/reconstruction-texture.png
```

## Interview Takeaway

This run demonstrates a deployed robotics perception pipeline rather than only
an offline vision demo: remote capture coordination, video transfer, frame
quality filtering, calibrated SfM, dense MVS, cache-aware recovery, and
quantitative reconstruction evaluation.
