# Component inventory and execution paths

Evidence date: 2026-10-24. Baseline: 176 tests passing before this milestone's
additions; after them the full suite reports 188 passed (~100 s, CPU-only).

## 1. Execution paths that actually run

| Path | Entry point | What it does | What it does not do |
| --- | --- | --- | --- |
| Offline demo pipeline | `tools/run_all.py` → `prism/pipeline.py` + `prism/training/run_*_infer.py` | Runs cached perception stages per clip and renders demo videos. | No live input, no vehicle control, no metric certification. |
| Live navigator | `prism/runtime.py` (`PrismNavigator`) | Frame-in / velocity-out control loop wrapping segmentation, depth, traversability, odometry, mapping, planner, supervisor. | Not exercised in this milestone; planner/mapping/supervisor were not modified. |
| Integrated replay | `tools/run_prism_integrated_replay.py` | `clips/clip_01.mp4` → LibreYOLO detections → image tracking → separate ORB motion pass → COLMAP sparse diagnostics → terrain/localization status → Belief Ledger/Doubt Engine advisory. | No depth/pose fabrication; detection boxes stay 2-D; advisory does not command a vehicle. |
| Dataset validation | `tools/run_prism_dataset_validation.py` | TUM RGB-D bounded validation: RGB-only VO, RGB-D reference geometry, mapping-interface test, belief/doubt checks, terrain boundary. | No navigation-readiness claim; RGB-D inputs are benchmark reference inputs only. |
| YOLOP bounded evaluation | `tools/evaluate_yolop_bounded.py` | Same 27 frames through YOLOP ONNX and LibreYOLO baseline: latency, RSS, counts, scores, segmentation statistics. | No accuracy claim (no reviewed labels), no tracking, no prediction. |
| Observation replay | `prism/observation_replay.py`, `prism/video_observation_replay.py` | `Observation → RuntimeAdapter → BeliefLedger → DoubtEngine → advisory` and video-to-observation sequencing. | Provider must supply real evidence; decoding alone yields no measurements. |
| ROS 2 bridge (`ros2/`) | project-owned node sources | Wraps `PrismNavigator` for the rover/Gazebo. | Not run here; no ROS installation, no live camera. |

## 2. Capability inventory

| Capability | Modules | Status | Evidence | Boundary |
| --- | --- | --- | --- | --- |
| Video decode / deterministic replay | `prism/video_observation_replay.py`, `prism/observation_replay.py` | VALIDATED | `tests/test_video_observation_replay.py`, `tests/test_observation_replay.py` | Pixels and capture order only. |
| Object detection (baseline) | `prism/libreyolo_provider.py` | VALIDATED (integration) | `tests/test_libreyolo_provider.py`; integrated run report | Accuracy UNVALIDATED: no reviewed object annotations. |
| Object detection (YOLOP) | `prism/integrations/yolop_provider.py` | EXPERIMENTAL | `tests/test_reuse_integrations.py`; `output/yolop_bounded_evaluation.json` | Optional, local-only; BDD100K vehicle taxonomy; not default. |
| Detection accuracy evaluation | `prism/detection_evaluation.py` | IMPLEMENTED, UNVALIDATED on data | `tests/test_detection_evaluation.py`; annotations remain unreviewed | Refuses to invent labels. |
| Image-space tracking | `prism/integrations/tracking.py` | VALIDATED (deterministic IoU) | `tests/test_integrated_replay_adapters.py` | Norfair optional; tracks are image coordinates, not world trajectories. |
| Visual motion / VO | `prism/visual_odometry_replay.py` | IMPLEMENTED, TRACKING-ONLY without calibration | TUM validation report; integrated `camera_motion.csv` | Monocular relative motion; no metric scale claim. |
| Localization backends | `prism/integrations/localization.py` | BLOCKED for stella_vslam / RTAB-Map / Basalt; ACTIVE diagnostic path | `localization_backends.json`; `tests/test_integrated_replay_adapters.py`, `tests/test_reuse_integrations.py` | Selected path is OpenCV tracking-only + COLMAP sparse QA; relative-only. |
| COLMAP sparse inspection | `prism/colmap_inspection.py`, `prism/integrations/geometry.py` | VALIDATED (software) | `tests/test_colmap_inspection.py`, `tests/test_integrated_replay_adapters.py` | Native COLMAP units; metric scale UNVERIFIED. |
| Scale alignment | `prism/scale_alignment.py`, `tools/align_colmap_scale.py`, `docs/colmap_scale_reference_template.json` | IMPLEMENTED | `tests/test_scale_alignment.py` | Needs a reviewed physical distance; does not validate calibration or shape. |
| 2.5D map | `prism/perception/mapping.py`, `prism/types.py::BEVMap` | IMPLEMENTED; traversability accuracy UNVALIDATED | `tests/test_runtime.py`, mapping-interface section of TUM report | Unknown cells never become free; 6 cm cells from prototype config. |
| Layered terrain grid | `prism/integrations/terrain_grid.py`, `prism/integrations/terrain.py` | EXPERIMENTAL | `tests/test_reuse_integrations.py`; layered summary in validation report | Slope/roughness only where measured; no variance/normal layers; grid is not a terrain-accuracy claim. |
| Learned segmentation / traversability (legacy demo) | `prism/models/segmentation.py`, `seg_pidnet.py`, `traversability.py` | IMPLEMENTED for offline demos; UNVALIDATED on target rover | Offline pipeline caches | Non-YOLO legacy models; not extended in this milestone, not used as reuse candidates. |
| Learned depth (legacy demo) | `prism/models/depth.py` | IMPLEMENTED for offline demos; UNVALIDATED | Offline pipeline caches | Non-YOLO legacy model; excluded from new integration by project rule. |
| Uncertainty / world model | `prism/models/uncertainty.py`, `world_model.py` | IMPLEMENTED (demo) | Tests for runtime | Not integrated into replay reasoning. |
| IMU fusion | `prism/perception/imu_fusion.py` | IMPLEMENTED, BLOCKED in practice | `tests/test_pose_graph_imu.py` uses synthetic inputs | No IMU stream or camera-IMU calibration for clips; never called with fabricated samples. |
| Loop closure / pose graph | `prism/perception/loop_closure.py`, `pose_graph.py` | IMPLEMENTED (rover path) | `tests/test_pose_graph_imu.py` | Not exercised by replay. |
| Point-cloud utilities | `prism/perception/cloud_accum.py`, `lidarize.py`, `prism/integrations/geometry.py` | IMPLEMENTED / optional | tests and Open3D summary | COLMAP points stay in native units. |
| Planner / supervisor | `prism/nav/*` | NOT MODIFIED | `tests/test_dstar_lite.py`, `tests/test_goal_nav.py` | Out of scope for this milestone. |
| Reasoning boundary | `prism/observations.py`, `prism/runtime_adapter.py`, `prism/belief_ledger.py`, `prism/doubt_engine.py` | VALIDATED | `tests/test_observations.py`, `test_runtime_adapter.py`, `test_belief_ledger.py`, `test_doubt_engine.py` | Advisory only; unavailable evidence drives cautious recommendations. |
| Provider contract | `prism/integrations/provider_contract.py` | IMPLEMENTED | `tests/test_reuse_integrations.py` | Typed observation out; no control output. |
| Calibration | `prism/calibration.py` | IMPLEMENTED | `tests/test_calibration.py` | Camera intrinsics only; no camera-vehicle extrinsic or metric monocular scale. |

## 3. Connected vs unconnected

* Connected to the integrated replay: `libreyolo_provider`, `integrations.tracking`,
  `visual_odometry_replay`, `integrations.geometry`, `integrations.terrain`,
  `integrations.localization`, `integrations.mapping_diagnostic`,
  `observations`, `runtime_adapter`, `belief_ledger`, `doubt_engine`.
* Connected to the TUM validation harness: `benchmarks.tum_rgbd_adapter`,
  `benchmarks.reference_geometry`, `benchmarks.trajectory_evaluation`,
  `perception.mapping`, `integrations.terrain_grid` (new).
* Implemented but not wired into replay: `perception.loop_closure`,
  `perception.pose_graph`, `perception.imu_fusion`, `perception.dynamic_layer`,
  `models.*` (only the offline demo pipeline), `deploy.*`.
* New in this milestone and intentionally outside the live navigator:
  `integrations.provider_contract`, `integrations.yolop_provider`,
  `integrations.terrain_grid`. The live navigator consumes `BEVMap` directly; the
  replay path consumes observations. Keeping these separate avoids changing
  navigation behaviour to satisfy an offline evaluation.

## 4. Deliberate duplication (and why it stays)

| Duplication | Reason |
| --- | --- |
| LibreYOLO provider vs YOLOP provider | Both must emit the same typed observation; sharing an implementation would couple the baseline to an optional model. YOLOP stays optional. |
| IoU tracker vs Norfair tracker | Deterministic fallback when the optional dependency is missing; both are image-space only. |
| `terrain.py` (status gate) vs `terrain_grid.py` (layers) | One decides whether measuring is justified; the other is the measuring adapter. No algorithm duplication. |
| Offline cached pipeline vs live navigator | Reproducible rendering vs real-time control; pre-existing architecture, not introduced here. |

## 5. Gaps

1. No reviewed object annotations for `clips/clip_01.mp4` → detection accuracy
   cannot be measured; only latency/counts/scores.
2. No IMU stream and no camera-IMU calibration → Basalt/RTAB-Map VIO blocked;
   `imu_fusion` cannot be exercised on real data.
3. No metric depth or verified metric scale for the replay clip → terrain grids
   are metre-shaped but not metre-validated; COLMAP remains relative.
4. No per-cell variance or surface normals in `BEVMap` → those grid_map layers
   are omitted rather than approximated.
5. rpy/360 TUM archives are now complete (official sizes) and both ran 80 frames.
   Measured outcomes include failures: rpy fails only the path-fraction check
   (0.0298 m / 0.247 m), 360 exceeds the 0.1 m ATE limit (0.1057 m) and its
   reference-input mapping interface refuses a support-plane fit. These are
   reported per sequence, not averaged away.
6. No ROS installation → grid_map/elevation_mapping/traversability_estimation
   cannot be run natively.
7. No live-camera path tested (explicitly deferred).
8. YOLOP segmentation has no off-road labels → segmentation output statistics
   are not evaluated for accuracy and are not traversability.

## 6. Acceptance criteria used here

* New provider: deterministic decode-only tests, SHA pin enforcement, typed
  observation, no segmentation leak into detection fields, explicit
  `UNAVAILABLE`/error instead of fabricated output.
* New terrain layers: unknown cells preserved, NaN where neighbourhood is
  unmeasured, flat/ramp sanity checks, no variance/normal claims.
* Localization record: sensor availability stated as fact; blocked candidates
  carry concrete blockers; selected path still relative-only.
* Repository: frozen files unchanged; full test suite passing; both documented
  validation commands still supported.
