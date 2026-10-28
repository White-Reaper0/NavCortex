# Reuse-first reuse strategy

Status: living document. Every claim here points at code, a test, or a measured
report; unverified items say so.

## 1. Principle

Reuse the smallest proven component that removes real work, behind a narrow
PRISM-owned adapter. Do not reimplement mature algorithms, and do not adopt a
component just because it exists. The burden of proof is on adoption: a
candidate must add capability the repository does not have, fit the offline
Windows/CPU constraints, and be representable in the existing observation
contract without inventing measurements.

The reverse rule applies too: a candidate that runs only by fabricating sensor
data (synthetic IMU, synthetic depth from arbitrary-scale COLMAP, predicted
labels treated as ground truth) is rejected regardless of code quality.

## 2. Decision vocabulary

| Disposition | Meaning |
| --- | --- |
| KEEP | Already used; no change required by this milestone. |
| ADAPT | Concept or algorithm reused inside PRISM-owned code; upstream source not vendored. |
| EVALUATE | Implemented as an optional, local-only adapter and measured; not the default. |
| OPTIONAL | Usable if a host has the dependency; absence is reported, never hidden. |
| REJECT | Prohibited by project constraints or technically unsuitable. |
| BLOCKED | Justified candidate, but missing input, host support, or licensing certainty. |

Capability status vocabulary: IMPLEMENTED, EXPERIMENTAL, VALIDATED, UNVALIDATED,
BLOCKED, NOT IMPLEMENTED.

## 3. Constraints that shape decisions

* Windows 10, Python 3.11, CPU only, 8 GB RAM, no paid APIs.
* YOLO-family neural architectures only. No new U-Net, OmniUnet, MiDaS, Depth
  Anything or other non-YOLO network may be introduced.
* No Ultralytics package or AGPL model weights. LibreYOLO9t remains the baseline
  detector.
* No OpenVINS. No live-camera testing in this milestone.
* Never fabricate poses, depth, IMU samples, detector results, or labels.
* COLMAP is a local diagnostic dependency only: no vendoring, no redistribution.
* The repository has no usable Git metadata. No destructive Git operations and
  no assumption of a clean working tree.
* Frozen files are not edited: `prism/config.py`, `prism/types.py`,
  `prism/io_utils.py`, `prism/viz_common.py`, `prism/perception/geometry.py`.

## 4. Reuse decisions in one line each

| Candidate | Disposition | Why |
| --- | --- | --- |
| LibreYOLO9t | KEEP | Existing detector baseline; hash-pinned local checkpoint. |
| Norfair | KEEP | BSD-3 tracker already integrated with an IoU fallback; image-space only. |
| OpenCV | KEEP | Decode, ORB motion, DNN ONNX runtime already present. |
| COLMAP | KEEP | Offline sparse-model diagnostic; local-only, no metric claim. |
| Open3D | KEEP | Optional native-unit sparse point inspection/PLY export. |
| YOLOP (detection) | EVALUATE | Works on CPU with a pinned ONNX, but different taxonomy, no accuracy labels, no decisive speed win; optional provider only. |
| YOLOP (segmentation) | REJECT as evidence | BDD100K drivable-area/lane output is not off-road traversability; recorded as model statistics only. |
| Basalt (VIO) | BLOCKED | Requires IMU + camera-IMU calibration that the clips do not contain; mono shortcut would not be VIO. |
| RTAB-Map | BLOCKED | Needs RGB-D or supported logs; synthetic depth from COLMAP is prohibited. |
| stella_vslam | BLOCKED | WSL/C++ build chain; no binary on the host. |
| grid_map | ADAPT | Multilayer elevation/variance/normal/traversability *concept* reused in a ROS-free Python adapter. |
| elevation_mapping | BLOCKED | ROS + metric elevation inputs absent; layered cells covered by the adapter. |
| traversability_estimation | BLOCKED | ROS + metric elevation + ground-truth traversability absent. |
| Ultralytics weights | REJECT | AGPL weights prohibited. |
| OpenVINS | REJECT | Explicitly out of scope. |
| MiDaS / Depth Anything / U-Net / OmniUnet | REJECT for new work | Non-YOLO architectures prohibited; legacy demo caches are not extended. |

Machine-readable records: `tools/reuse_candidates.json`.

## 5. What was actually reused and implemented

1. **Provider contract** (`prism/integrations/provider_contract.py`):
   `DecodedFrame -> VideoReplayInput`, no control output.
2. **YOLOP ONNX provider** (`prism/integrations/yolop_provider.py`):
   SHA-256 pinned local ONNX, OpenCV DNN CPU, letterbox + ImageNet
   preprocessing from upstream `tools/demo.py`, detection-only observation,
   segmentation exposed only as descriptive statistics.
3. **ROS-free multilayer terrain grid** (`prism/integrations/terrain_grid.py`):
   grid_map's multilayer idea, implemented with NumPy/OpenCV; unknown cells
   remain `UNKNOWN` with NaN geometry; variance and normals are explicitly not
   synthesised.
4. **Localization candidate record** (updated `prism/integrations/localization.py`):
   Basalt added as BLOCKED with sensor-availability facts.
5. **Bounded evaluation tool** (`tools/evaluate_yolop_bounded.py`): same 27
   frames, latency/memory/counts, segmentation statistics, baseline comparison,
   accuracy declared unmeasured.
6. **Validation wiring** (`tools/run_prism_dataset_validation.py`): the mapping
   interface test now also emits a layered-grid summary and a check that
   unobserved cells stay UNKNOWN.

## 6. Adoption checklist for any future candidate

1. Capability gap must be named and traced to a failing or missing PRISM path.
2. License file fetched and recorded with URL and date; weight/dataset terms
   recorded separately from code terms.
3. Sensor and calibration prerequisites confirmed from data on disk, not assumed.
4. Host feasibility measured (build time, RAM, Windows support) before coding.
5. Adapter must map into `Observation` without adding unmeasured fields.
6. Bounded test on the same frames as the baseline; accuracy only claimed with
   reviewed labels.
7. Fallback path retained and tested (model absent -> explicit UNAVAILABLE).
8. Non-empty output directories are never overwritten by evaluation tools.

## 7. Known limits of this strategy

* No Rosbag/ROS runtime is available, so ROS-native candidates stay blocked.
* The clips have no IMU, no camera-IMU calibration, and no reviewed object or
  terrain annotations, which caps what any adopted model can be *validated* on.
* YOLOP's segmentation heads are from a road-scene dataset; they cannot be
  repurposed as off-road traversability without new labels and evaluation.
