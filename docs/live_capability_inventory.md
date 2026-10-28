# Live capability inventory (audited 2026-10-25)

This inventory was produced by reading the implementations, not the READMEs. It
exists to answer one question before anything is put on screen: **what can this
repository actually compute today, on a live camera, with the inputs it has?**

Environment of the audit: `.venv` Python 3.11, `opencv-python 5.0.0.93`
(WIN32UI, native windows available), `torch 2.4.1+cpu`,
`libreyolo 1.6.0`, `norfair 2.3.0`, `open3d 0.20.0`, `transformers 4.57.6`,
`huggingface_hub 0.36.2`, CPU only. Re-audited 2026-10-25 after the approved
pretrained weights were installed under `model_assets/pretrained/` (see
`model_assets/pretrained/manifest.json` and
`docs/licenses_and_provenance.md`); the measured CPU costs in this table come
from `output/pretrained_smoke_tests/20261025T145051Z/smoke_report.json` and one
bounded headless viewer run with every socket blocked.

## 1. Readiness classes

- **LIVE** — runs per frame (or per decimated frame) on a live camera today.
- **OFFLINE** — runs today, but only on a captured sequence / completed reconstruction.
- **EXPERIMENTAL** — computes real numbers from real inputs, but the inputs are
  assumed-scale or the semantics are a prior rather than a measurement; it may be
  drawn only with that limitation on screen.
- **BLOCKED** — code exists, required input does not; must be reported as blocked, never drawn.

| Capability | Entry point actually used | Class | Inputs required | Measurement meaning | Cost (measured/reported) | What may be shown today |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection | `prism.libreyolo_provider.LibreYOLODetectionProvider.from_checkpoint` + `__call__(DecodedFrame)` | LIVE | local `model_assets/libreyolo/LibreYOLO9t.pt` (present) | typed 2-D image observations, scores are model scores | LibreYOLO baseline p50 408 ms, max 9 099 ms (`output/yolop_bounded_evaluation.json`) | boxes, labels, scores, counts, per-call latency |
| Image-space tracking | `prism.integrations.tracking.create_image_tracker` (`NorfairImageTracker`, `ImageSpaceIoUTracker`) | LIVE | detector's `ObjectDetection` tuple | image-space tracks; `predicted` flag marks extrapolation | negligible vs detector | track IDs, boxes, trails, observed/predicted counts |
| Camera motion | `prism.visual_odometry_replay._OrbFrontend`, `_mutual_ratio_match`, `estimate_fundamental_inliers`, `estimate_relative_motion`, `FrameOdometryRecord` | LIVE | none beyond frames; `CameraCalibration.missing()` is the default | without intrinsics: `TRACKING_ONLY` (features/matches/inliers/Sampson). No metric pose exists | ORB + MAGSAC per frame at 640x360 | match/inlier counts, inlier ratio, tracking status, loss reasons |
| Road-scene segmentation + detection (YOLOP) | `prism.integrations.yolop_provider.YOLOPDetectionProvider.from_onnx` (`run_detailed`, `segmentation_summary`) | LIVE (decimated) | local `tools/external/yolop/yolop-320-320.onnx`, SHA-256 verified | detections are 2-D; segmentation is returned as **statistics only** | p50 298 ms, p95 474 ms, max 2 827 ms (27 frames, CPU) | detections + mask statistics as numbers; masks are never terrain |
| Evidence and doubt | `prism.belief_ledger.BeliefLedger` (`record`, `advance_to`, `hypothesis_status`), `prism.doubt_engine.DoubtEngine.assess` | LIVE | observations from providers, caller-chosen `expires_at_ns` | evidence state per hypothesis (`CURRENT/STALE/EXPIRED/MISSING/UNAVAILABLE/INVALID/CONTRADICTORY`) and advisory `request_one_observation` / `wait_for_one_fresh_frame` / `cautious_stop` | sub-millisecond | real per-hypothesis states and advisory text; explicitly not a control decision |
| Runtime-geometry gate | `prism.runtime_adapter.RuntimeAdapter.to_packet` | LIVE (gate only) | an `Observation` | rejection reason when 2-D evidence cannot become geometry | negligible | the rejection string itself, e.g. detection-only observations are refused |
| Depth estimation | `prism.models.depth.DepthStage` (loads `model_assets/pretrained/depth_anything_v2_small_hf` through `prism.models.pretrained_paths`, `local_files_only=True`) | **LIVE** (decimated) | the approved HF directory, present on this host; overridable with `--depth-model` / `PRISM_DEPTH_MODEL_PATH` | relative inverse depth `q`; its `depth_m` comes from `fit_metric_ground` against the **assumed** `CFG.cam.height_above_ground_m = 0.12` | measured on CPU at 640x360: p50 5.7 s, max 5.8 s in the bounded run (up to 16 s under load) | colourised relative depth beside the live frame, valid fraction, fit `a`/`b`/residual/inliers, latency and result age - always labelled assumed scale. If the directory is missing the view goes back to a block card naming `tools/download_prism_pretrained.py` and draws nothing |
| Semantic segmentation | `prism.models.segmentation.SegStage` → `prism.models.seg_teacher.SegTeacher` (SegFormer-B0/ADE20K from `model_assets/pretrained/segformer_b0_ade_512_512`) | **LIVE** (decimated) | the approved HF directory, present; `--segmentation-model` / `PRISM_SEGMENTATION_MODEL_PATH` | prism-7 label map produced by the ADE20K→prism-7 mapping (unchanged), `IGNORE_INDEX` where the mapping abstains | measured on CPU: p50 4.1 s, max 6.4 s | label colours, overlay on the live frame, per-class counts, mean top-probability/entropy, drivable fraction - as pretrained semantics, never as traversability |
| Terrain grid | `prism.integrations.terrain.terrain_status`, `prism.integrations.terrain_grid.terrain_grid_from_bev` | **BLOCKED** for the metric grid; an experimental geometric grid is drawn alongside | `terrain_status` returns `BLOCKED_MISSING_INPUT` unless `physical_scale == "METRIC_SCALE_ALIGNED"` (`prism/integrations/terrain.py:24`); a metric grid needs a metric `BEVMap`, which needs metric depth/pose | the geometric grid's numbers are assumed-scale heights and a geometric traversability prior | grid build is cheap (fuse ≈ 6 ms, observe 53-1411 ms depending on cache state) | the gate's own status/reason text is still shown next to the grid, and `grid` stays `None`; the drawn grid is labelled `GEOMETRIC / ASSUMED SCALE / UNVALIDATED` and unobserved/low-confidence cells stay UNKNOWN |
| Learned traversability | `prism.models.traversability.TraversabilityStage` | **NOT USED BY THIS VIEWER** | never instantiated here. The checkpoint `checkpoints/trav_unc_shared.pt` is owned by the separate manual training task; it did not exist during this milestone and this milestone created no checkpoint, but that task has since written one (its own log, not our result) | n/a | n/a | `tools/run_prism_live_viewer.py:learned_traversability_note()` renders the real file state: `NOT TRAINED (trav_unc_shared.pt does not exist)` when absent, `NOT USED BY THIS VIEWER` when present. `learned_traversability_trained: False` in the report, `trav_source == "geometric"` on screen, and nothing was trained by this milestone |
| Mapping / occupancy update | `prism.perception.mapping.MappingStage`, `prism.nav.bev_utils.pack_state/render_state_bgr`, `prism.nav.goal_map.GlobalCostGrid` | **EXPERIMENTAL** live (camera-frame BEV) | `packet.depth.depth_m` (+ optional `packet.seg`); no `odom`, so no ego-motion warp | `trav_source == "geometric"` with the stage's `GEO_UNKNOWN_FLOOR`: a prior over assumed-scale geometry, not a learned classifier | measured on CPU: 68 ms per fuse in the bounded run | the height/traversability tiles with cell fractions and hit counts, labelled assumed-scale and unvalidated. `prism.integrations.mapping_diagnostic.mapping_diagnostic_status` still reports `DIAGNOSTIC_ONLY`, `active_map_updated: False`, and no cost grid is published to navigation |
| Point-cloud / LiDAR-like views | `prism.perception.cloud_accum.WorldCamera` + `render_points`, `prism.perception.lidarize.depth_to_cloud/render_cloud` | LIVE for **reconstructed** points and for a **single-frame** depth cloud | live depth for the unprojection; COLMAP `points3D` for `render_points` | arbitrary units unless a reviewed scale reference exists | splat render per refresh (36,039 points offered from one frame at stride 2) | the COLMAP sparse cloud, plus a coloured single-frame cloud labelled `SINGLE-FRAME / ASSUMED SCALE / NOT A RECONSTRUCTION` |
| 3-D reconstruction | `tools/colmap_sparse_probe.extract_frames` manifest schema + `tools/colmap_sparse_probe.run_colmap`, `prism.colmap_inspection.read_sparse_model/build_qa_report/write_ply/trajectory_rows`, `prism.integrations.geometry.sparse_geometry_report/open3d_sparse_inspection` | OFFLINE | frames dir + `frames_manifest.json` + `tools/external/colmap/4.2.1-nocuda/bin/colmap.exe` (present), CPU, 30-60 frames, minutes of runtime | `status: RELATIVE_ONLY`, `physical_scale: METRIC_SCALE_UNVERIFIED`, "COLMAP coordinates have arbitrary monocular scale" | minutes, bounded single worker | progress/status live; point cloud + PLY + QA after completion, labelled arbitrary-scale |
| Metric scale | `prism.scale_alignment.aligned_report` | **BLOCKED** | a `reviewed: true` two-landmark reference + calibration; `output/09_colmap_sparse/clip_01_repro/scale_alignment/scale_alignment_report.json` is `METRIC_SCALE_UNVERIFIED` with calibration `MISSING` | n/a | n/a | the unverified status |
| Camera calibration | `prism.calibration.calibrate_checkerboard`, `load_calibration_artifact`, `CameraCalibration.from_artifact` | LIVE workflow, **no artifact** | ≥3 (min 8) checkerboard images; docstring: "The repository does not ship a calibration asset." | validated intrinsics when produced | per-artifact load is trivial | if the user supplies an artifact, motion becomes `CALIBRATED`; otherwise `MISSING` and motion is tracking-only |
| Vehicle planning/control | `prism.runtime.PrismNavigator`, `prism.nav.planner`, `prism.nav.supervisor`, `ros2/**` (`Twist` on `/cmd_vel`) | **EXCLUDED BY DESIGN** | n/a | n/a | n/a | not imported by the viewer; the viewer is advisory-only |

## 2. Consequences for the live workbench

1. Drawn from live frames: detection, image tracking, ORB/fundamental motion
   diagnostics, YOLOP (decimated), the evidence/doubt advisory states, and - with
   the installed pretrained weights - the relative-depth view, the prism-7
   semantic segmentation view, the geometric BEV terrain grid and the
   single-frame depth point cloud. The first group is 2-D image evidence or
   scale-free geometry; the second group is real model output whose metre scale
   is assumed and whose terrain labels are a geometric prior.
2. Metric terrain, learned traversability, active mapping, metric scale and the
   camera-calibration-dependent motion classes keep their explicit block cards
   naming the missing artifact (calibration asset, `checkpoints/`,
   metric-scale reference). A missing pretrained directory returns the depth and
   segmentation views to the same treatment: `BLOCKED_MISSING_INPUT`, the
   installer command, and no image. No placeholder, gradient or heuristic image
   is ever drawn in place of a blocked capability.
3. 3-D is honest offline work: capture a frame sequence with the manifest schema
   `tools/colmap_sparse_probe.py` already uses, run the local COLMAP binary in one
   background worker, then inspect the sparse model with `prism.colmap_inspection`
   and render it through `prism.perception.cloud_accum.render_points` (plus an
   optional Open3D window). It is labelled arbitrary-scale; `Mapper.extract_colors 0`
   means point colours come from triangulation, and `write_scaled_exports` is only
   reached with a reviewed metric reference.
4. Results are always stamped with the frame they came from, so a 400 ms detector
   cannot be mistaken for the current frame.
5. `prism/runtime.py`, `prism/nav/planner.py`, `prism/nav/supervisor.py` and
   anything under `ros2/` are not imported. `AdvisoryDisposition` values
   (`no_action`, `cautious_stop`) are displayed as text only.
6. These rules are implemented, and now enforced by tests, in the native
   workbench: see `tools/run_prism_live_viewer.py`,
   `docs/live_viewer_workbench.md`, `tests/test_live_viewer.py` and
   `tests/test_live_viewer_pretrained.py` (the latter injects the stages, so it
   asserts the integration without loading torch), plus
   `tools/run_prism_pretrained_smoke.py` for the real-weights check. The
   inventory above is the input to that tool; if a capability's class changes,
   change the block card and the test together, not the picture.
