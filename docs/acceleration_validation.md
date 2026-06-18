# Accelerated Pipeline Validation

## Benchmark Dataset

- Source: successful tape-measure Session 003 video
- Video: 2,675 frames, 89.17 seconds, 1280 x 720
- Original reconstruction: 173 selected frames

## Adaptive Keyframes

The accelerated profile sampled every third video frame and selected 120
motion-distinct keyframes across the complete recording.

| Metric | Result |
| --- | ---: |
| Sampled frames | 892 |
| Selected keyframes | 120 |
| Redundant samples removed | 773 |
| First source frame | 0 |
| Last source frame | 2,673 |
| Median selected-frame motion | 6.12 px |
| Median feature overlap | 46.4% |
| Selection runtime | 22.7 sec |

The first and last usable views were retained so the selected sequence still
covered the full camera orbit.

## Accelerated Sparse SfM

COLMAP ran from WSL's native filesystem using sequential matching with overlap
15, quadratic overlap, and 225 explicit end-to-start loop pairs.

| Metric | Original Session 003 | Accelerated validation |
| --- | ---: | ---: |
| Selected images | 173 | 120 |
| Registered images | 173 / 173 | 120 / 120 |
| Registration ratio | 100% | 100% |
| Sparse points | 31,685 | 20,872 |
| Observations | 220,309 | 108,796 |
| Mean track length | 6.95 | 5.21 |
| Mean reprojection error | 0.856 px | 0.822 px |

The lower sparse point count is expected because 30.6% fewer camera views were
processed. Registration remained complete and reprojection error improved by
0.034 px, so the sparse quality gate accepted the accelerated result.

Measured accelerated front-end runtimes:

| Stage | Runtime |
| --- | ---: |
| Adaptive keyframes | 0:23 |
| SIFT feature extraction | 0:38 |
| Sequential matching | 3:50 |
| Loop-pair matching | 0:18 |
| Mapping and bundle adjustment | 1:51 |

## Expected Dense Impact

Session 003 OpenMVS densification took 16:36 for 173 depth maps. The accelerated
profile requests at most 120 depth maps while leaving OpenMVS resolution,
neighbor-view, fusion, mesh, and texture quality settings unchanged. The
expected dense-stage reduction is approximately proportional to the 30.6%
reduction in views, subject to scene content and cache behavior.

If sparse registration falls below 92% or landmark density falls below 60
points per registered image, the worker automatically retries with up to 180
keyframes and exhaustive matching rather than proceeding with a lower-quality
model.
