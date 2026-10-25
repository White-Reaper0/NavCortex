# Claims register

Each row is a claim the repository is allowed to make, with its evidence and its
limit. Unverified or unmeasured items are labelled. If a claim is not here, it is
not supported by this milestone.

## A. Verified / validated

| ID | Claim | Status | Evidence | Limit |
| --- | --- | --- | --- | --- |
| C1 | The offline integrated replay runs end-to-end on `clips/clip_01.mp4` without live input, network, or vehicle control. | VALIDATED | `output/prism_integrated_test*/run_report.json`; integrated-runner test | Software path only; not navigation readiness. |
| C2 | The YOLOP ONNX (SHA-256 `86d6e8b6…a07369`) loads in OpenCV DNN on CPU and produces detections on the recorded frames. | VALIDATED (execution) | `output/yolop_bounded_evaluation.json`; 27/27 frames, 42 detections, no errors | Accuracy not measured; taxonomy differs from LibreYOLO. |
| C3 | On 27 frames, YOLOP latency p50 = 298.43 ms, p95 = 473.63 ms, max = 2827.08 ms; LibreYOLO baseline p50 = 408.25 ms, p95 = 4648.22 ms. | VALIDATED (measurement) | Same report; one CPU host, one run | Timing is host- and load-dependent; the tail favours neither model decisively. |
| C4 | Segmentation output statistics can be recorded without becoming evidence. | VALIDATED (implementation) | `segmentation_summary`; `tests/test_reuse_integrations.py` | Descriptive statistics only; no accuracy and no traversability meaning. |
| C5 | Unknown terrain cells remain UNKNOWN through the layered grid adapter, and slope/roughness are NaN where the neighbourhood is unmeasured. | VALIDATED (unit behaviour) | `tests/test_reuse_integrations.py` | Synthetic grids; real terrain not evaluated. |
| C6 | TUM `xyz`, 80 frames, RGB-only ORB: tracking success 76/79 attempts = 96.2 %, 3 implausible-yaw losses, ATE RMSE 0.0316 m after post-hoc Sim(3), RPE trans RMSE 0.0108 m, path-length ratio 1.374. | VALIDATED (benchmark run) | `output/tum_rgbd_validation_xyz.json` | Monocular scale unobservable; metres conditional on Sim(3); one sequence. |
| C6b | TUM `rpy`, 80 frames: tracking 69/79 = 87.3 %, 10 losses, ATE RMSE 0.0298 m, RPE trans 0.00535 m, RPE rot 1.023°, path ratio 1.387; the geometric check fails only `ate_path_fraction` because 0.0298 m / 0.247 m reference path = 0.121 > 0.10. | VALIDATED (benchmark run) | `output/tum_rgbd_validation_rpy.json` | The path-fraction threshold is ill-conditioned for near-stationary sequences; the absolute ATE limit (0.1 m) passes. Reported as a measured failure, not tuned away. |
| C6c | TUM `360`, 80 frames: tracking 65/79 = 82.3 %, 14 losses, ATE RMSE 0.1057 m (above the 0.1 m limit), RPE trans 0.0169 m, RPE rot 0.945°, path ratio 1.595; reference-input mapping-interface test refuses (no support-plane fit). | VALIDATED (benchmark run) | `output/tum_rgbd_validation_360.json` | Fails the absolute ATE threshold and the mapping-interface stage; this is the honest measured result, not a software regression. |
| C7 | The three rotation numbers are different metrics, not disagreement: 0.483° RPE rotation RMSE (frame-to-frame, alignment-free), 2.185° ATE rotation RMSE (absolute, after global SO(3) fit, 80 frames), and 144.7° ATE rotation alignment (a removed global offset, not an error). The earlier 0.498° came from a 6-frame smoke run. | VALIDATED (explanation) | `prism/benchmarks/trajectory_evaluation.py:354-372`; `output/tum_rgbd_validation_xyz.json`; `output/tum_rgbd_validation_smoke.json` | ATE rotation grows with frame count; comparisons must state frames and alignment. |
| C8 | The mapping interface accepts vehicle-frame reference points and preserves unobserved cells as UNKNOWN/NaN. | VALIDATED (interface) | TUM report `mapping_terrain`; `layered_grid_unknown_preserved` check | Interface test only; traversability accuracy unmeasured. |
| C9 | COLMAP sparse models are loaded read-only in native units; exact pose matches require matching source-frame identity. | VALIDATED (software) | `tests/test_integrated_replay_adapters.py`, `tests/test_colmap_inspection.py` | Metric scale UNVERIFIED; no rerun, no redistribution. |
| C10 | Unavailable terrain evidence drives a cautious-stop advisory rather than a safe/clear decision. | VALIDATED | `tests/test_integrated_replay_adapters.py`, belief/doubt tests | Advisory only; no vehicle command. |
| C11 | No IMU stream or camera-IMU calibration exists for the replay clips or TUM data used here. | VALIDATED (fact) | `localization_backend_report()["sensor_inputs"]`; module docs | Does not imply the rover lacks an IMU; it says these datasets do. |
| C12 | Frozen files (`config.py`, `types.py`, `io_utils.py`, `viz_common.py`, `perception/geometry.py`) are unchanged. | VALIDATED | Modification-time check against pre-milestone state | Only mtime-verified; there is no usable Git history to diff. |

## B. Implemented, not validated

| ID | Claim | Status | Evidence | Limit |
| --- | --- | --- | --- | --- |
| C13 | A reusable provider contract exists for optional perception adapters. | IMPLEMENTED | `prism/integrations/provider_contract.py`; protocol test | Two providers implement it today. |
| C14 | A ROS-free multilayer terrain grid exists with elevation/support/step/slope/roughness/traversability/confidence/age layers. | EXPERIMENTAL | `prism/integrations/terrain_grid.py`; tests | No variance/normal layers; metric status may be `REFERENCE_INPUT_UNVERIFIED`. |
| C15 | YOLOP segmentation output fractions are recorded per frame (mean drivable fraction 0.124, lane 0.0005 over 27 frames). | EXPERIMENTAL | `output/yolop_bounded_evaluation.json` | Model statistics, not labels; two frames show 0.0 drivable fraction, which is plausible for off-road footage and not an error claim. |
| C16 | Detection accuracy for either detector on this clip is measurable by the existing evaluation harness once annotations are reviewed. | IMPLEMENTED, UNVALIDATED | `prism/detection_evaluation.py`; annotations remain `unreviewed` | No reviewed labels → no mAP/precision/recall claim. |
| C17 | Camera intrinsics can be produced from checkerboard images and recorded as a versioned artifact. | IMPLEMENTED | `prism/calibration.py`; tests | Intrinsics only; no camera-vehicle extrinsic, no metric scale. |

## C. Blocked

| ID | Claim | Status | Evidence | Limit |
| --- | --- | --- | --- | --- |
| C18 | Basalt visual-inertial odometry is blocked by missing IMU samples and camera-IMU calibration. | BLOCKED | `localization_backend_report()`; `docs/reuse_strategy.md` | Mono mode would not be VIO and would not supply metric scale. |
| C19 | RTAB-Map is blocked for this input (monocular RGB only); synthetic depth from COLMAP is prohibited. | BLOCKED | Existing localization report | Not evaluated further. |
| C20 | stella_vslam is blocked by the WSL/C++ build chain; no binary is present. | BLOCKED | `localization_backends.json`; WSL probe | Not attempted. |
| C21 | grid_map/elevation_mapping/traversability_estimation cannot run natively here (ROS, metric elevation). | BLOCKED | This register + integration manifest | Multilayer *concepts* were adapted; no ROS link. |
| C22 | TUM `rpy` and `360` archives are complete (410,268,381 and 417,954,230 bytes), extracted, and each ran 80 frames through the validation harness. | VALIDATED (availability + run) | `output/tum_rgbd_validation_{rpy,360}.json`; local SHA-256 `78103722…c4ac1` and `c4a4d631…a4e97` | Availability is not accuracy: rpy and 360 both fail at least one geometric check (C6b/C6c). |
| C23 | Visual odometry from `clips/clip_01.mp4` has metric scale. | BLOCKED | `visual_odometry_replay` tracking-only default; TUM note on monocular scale | No calibration + no metric depth. |
| C23b | The reference-input mapping interface succeeds on every sequence. | BLOCKED for `360` | `output/tum_rgbd_validation_360.json` timing stage `mapping_interface` | The support-plane fit is unavailable on the selected reference frame; the harness refuses rather than fabricating vehicle-frame geometry. |

## D. Explicitly rejected claims

| ID | Claim that is **not** made | Reason |
| --- | --- | --- |
| C24 | "YOLOP drivable-area output is off-road traversability." | BDD100K road-scene labels; no off-road evaluation; would confuse segmentation with terrain classification. |
| C25 | "YOLOP is now the default detector." | Disposition EVALUATE; LibreYOLO9t remains baseline. |
| C26 | "Two detection providers' counts are comparable." | Different taxonomies (BDD100K vehicle vs COCO-80) and different score semantics. |
| C27 | "SLAM/VIO is active or validated." | No sensor inputs; no build; no run. |
| C28 | "The terrain grid is a validated metric elevation map." | Metre units are inherited from the prototype grid, not verified; no terrain ground truth. |
| C29 | "Unknown cells may be treated as safe." | Contradicts the mapping contract; unknown is absence of measurement. |
| C30 | "Live-camera operation was tested." | Explicitly deferred. |
| C31 | "Planner, mapping, or supervisor behaviour was improved." | Not modified; no verified defect was found. |
| C32 | "Integrated replay passing means navigation readiness." | Software-path integration only. |
| C33 | "New non-YOLO neural architectures were introduced." | Constraint; YOLOP is YOLO-family and optional. |
| C34 | "IMU, poses, depth, or detections were synthesised to fill missing fields." | Constraint; blocked paths report blockers instead. |

## E. How to update

Add a row only with a path to evidence and a limit. When a blocked item becomes
runnable, record the sensor/calibration provenance first, then the measurement,
then flip the status. Never promote a software check to an accuracy claim.
