# P.R.I.S.M. native live vision workbench

`tools/run_prism_live_viewer.py` is a native Python/OpenCV desktop application.
It opens a real camera and shows the perception capabilities this repository can
actually execute on those frames. Everything else is shown as blocked, with the
name of the missing artifact. It is not a dashboard, web server, or Electron app,
and it never commands a vehicle.

Read `docs/live_capability_inventory.md` first: it is the audit this workbench is
built on. This document describes the tool, the keys, the outputs and the limits.

## Launch

Both sources use the same `.venv`.

```powershell
# PC webcam (device index 0 by default)
.\.venv\Scripts\python.exe tools\run_prism_live_viewer.py --source webcam --camera-index 0

# OnePlus phone running the IP Webcam app (plain HTTP MJPEG)
.\.venv\Scripts\python.exe tools\run_prism_live_viewer.py --source ipwebcam `
  --url "http://10.78.70.188:8080/video"
```

URL precedence for the phone stream is `--url` > `PRISM_IPWEBCAM_URL` > the
convenience default `http://10.78.70.188:8080/video`; `/video` is never doubled.
When the phone's IP changes, either pass a new `--url`, export a new
`PRISM_IPWEBCAM_URL`, or point `--url-file` at a one-line text file holding the
current URL: the file is re-read on every reconnect and on the `U` key, so no
source edit or restart is needed. The phone dashboard is
`http://10.78.70.188:8080/`; use `https://` only if the app was explicitly
configured for SSL.

Useful flags: `--headless` (no window, bounded audit run), `--max-frames N`
(stop after N processed frames), `--buffer-frames` (bounded hand-off, default 2),
`--detector-interval` / `--motion-interval` / `--yolop-interval`, `--confidence`,
`--weights`, `--with-yolop`, `--calibration <artifact>` or
`--allow-assumed-intrinsics`, `--recon-frames` / `--recon-interval` /
`--colmap-exe` / `--auto-reconstruct` / `--open3d-viewer`, `--record`,
`--output <dir>`. Run with `--help` for defaults. Weights are never downloaded:
an absent checkpoint makes detection `UNAVAILABLE` and draws no boxes.

## Pretrained vision models (depth, segmentation)

The depth and segmentation views run two downloaded ImageNet/ADE20K checkpoints
from disk, on CPU, with `local_files_only=True`. Nothing in the viewer can start
a download: the loaders take an explicit local directory and the process is
verified to work with every socket blocked.

```powershell
# one-time, idempotent installer (records model_assets\pretrained\manifest.json)
.\.venv\Scripts\python.exe tools\download_prism_pretrained.py

# then exactly the normal launch - the viewer finds the local directories itself
.\.venv\Scripts\python.exe tools\run_prism_live_viewer.py --source webcam --camera-index 0
```

`--depth-model` / `--segmentation-model` point at a different local directory, and
`PRISM_DEPTH_MODEL_PATH` / `PRISM_SEGMENTATION_MODEL_PATH` override resolution for
every entry point (viewer, training-label builder, ONNX export). `--no-depth` and
`--no-segmentation` skip a stage and report it as `DISABLED`. When a directory is
absent or incomplete the view reports `BLOCKED_MISSING_INPUT` and names the
installer command; it draws no substitute image.

These models are slow on a CPU (see the measured numbers below), so the stages are
sampled: `--depth-interval`, `--seg-interval` and `--map-interval` run each stage
every N processed frames (default 5, first frame always runs) and the views keep
showing the last real result with its age. `--cloud-stride` subsamples the
point-cloud view.

## Keys

| Key | Effect |
| --- | --- |
| `1` | live frame + LibreYOLO 2-D detections |
| `2` | image-space tracking: IDs, boxes, trails, observed/predicted |
| `3` | depth: live frame + colourised relative inverse depth + assumed-scale metres |
| `4` | segmentation: per-pixel prism-7 labels from the pretrained SegFormer and the overlay on the live frame |
| `5` | terrain: geometric BEV grid (height + traversability) labelled `GEOMETRIC / ASSUMED SCALE / UNVALIDATED`, plus a learned-traversability line generated from real file state (`NOT TRAINED …` when `checkpoints/trav_unc_shared.pt` is absent, `NOT USED BY THIS VIEWER` when the separate training task has produced it) |
| `6` | camera motion: ORB matches and fundamental-matrix inliers, arbitrary scale |
| `7` | offline COLMAP reconstruction (sparse cloud through the repo renderer) |
| `8` | single-frame point cloud unprojected from the live depth, labelled assumed scale |
| `9` | evidence: BeliefLedger states, DoubtEngine advisory, adapter gate, module table |
| `0` | overview tiles of the above |
| `R` | toggle recording of the composed view (`recording.mp4`) |
| `S` | snapshot: current view PNG + JSON sidecar with the live diagnostics |
| `M` | reset the motion reference, the trails and the perception stages (depth smoothing, BEV) |
| `H` | limits overlay (the advisory-only statements and blocked capabilities) |
| `C` / `K` / `V` / `O` | capture a new sequence / run COLMAP / open the native Open3D window / orbit preset |
| `U` | re-resolve the stream URL (new phone IP) and reopen the source |
| `P` | pause inference: frames are dropped, never queued |
| `Q` / `Esc` | clean stop, flush the report |

## What each view is allowed to show

| View | Real implementation used | Class | Meaning of the numbers |
| --- | --- | --- | --- |
| live, tracking | `LibreYOLODetectionProvider`, `create_image_tracker` | LIVE | 2-D image observations, model scores, image-space IDs; a predicted track is labelled as extrapolation |
| motion | `_OrbFrontend`, `_mutual_ratio_match`, `estimate_fundamental_inliers`, `estimate_relative_motion` | LIVE | without calibration `tracking_only: True` (features/matches/inliers/Sampson only); with assumed intrinsics the status is `unvalidated_motion`; never a metric pose |
| depth | `prism/models/depth.py:DepthStage` (Depth Anything V2-Small, local CPU) + `fit_metric_ground` | LIVE / ASSUMED SCALE | colourised relative inverse depth, valid fraction, per-run latency and result age, and the ground-fit `scale`/`shift`/residual/inliers anchored on the **assumed** 0.12 m camera height. The metre column is explicitly not a calibrated measurement |
| segmentation | `prism/models/segmentation.py:SegStage` -> `SegTeacher` (SegFormer-B0 ADE20K, local CPU) | LIVE / PRETRAINED SEMANTICS | prism-7 label map, overlay on the live frame, per-class pixel counts, mean top-probability and entropy, drivable fraction, latency and age; always labelled pretrained ADE20K semantics mapped to prism-7, never ground-truth traversability. `IGNORE_INDEX` (255) abstentions stay unlabelled |
| terrain | `prism/perception/mapping.py:MappingStage` geometric fallback, `cell_stats` | EXPERIMENTAL | BEV height/traversability built from assumed-scale depth plus the semantics above, drawn as `GEOMETRIC / ASSUMED SCALE / UNVALIDATED`. Unknown/stale/low-confidence cells stay UNKNOWN; the learned-traversability line printed next to it is generated from real file state and the viewer never loads `checkpoints/trav_unc_shared.pt`; the metric gate `terrain_status` stays `BLOCKED_MISSING_INPUT` |
| cloud | `geometry.unproject` + `cloud_accum.render_points` on the live depth | LIVE / ASSUMED SCALE | one frame's coloured points, labelled `SINGLE-FRAME / ASSUMED SCALE / NOT A RECONSTRUCTION`; the offline COLMAP workflow (view 7) is the only 3-D geometry pipeline and is never called real-time |
| reconstruction | `tools/colmap_sparse_probe.run_colmap`, `read_sparse_model`, `sparse_geometry_report`, `render_points` | OFFLINE | a background worker over a captured sequence; reported as `RELATIVE_ONLY` / `METRIC_SCALE_UNVERIFIED`; explicitly not a real-time model |
| evidence | `BeliefLedger`, `DoubtEngine`, `RuntimeAdapter.to_packet` | LIVE | real per-hypothesis states, advisory kinds only (`request_one_observation`, `wait_for_one_fresh_frame`, `cautious_stop`), and the adapter's real rejection string |
| mapping | `mapping_diagnostic_status` | BLOCKED | reported as diagnostic-only with `active_map_updated: False`; no occupancy grid is drawn |

The status rail always reports measured values: source (redacted), frame size,
frames read, processed FPS, stale frames dropped, detector p50 and current
latency, detection/track counts, result age in milliseconds, module states and
the latest error. A view whose module is offline-only, missing prerequisites,
unavailable or failed says so in that view; no static or fabricated visual is
ever presented as live output. `prism/runtime.py`, `prism/nav/planner.py`,
`prism/nav/supervisor.py` and `ros2/` are not imported.

## Outputs

Every run gets a unique directory under `output/live_viewer_runs/<UTC>-<token>/`
(the token is random and collides only with an existing directory, because the
Windows monotonic clock is too coarse to distinguish two runs in the same
second):

- `run_config.json` - resolved configuration, URL redacted;
- `run_report.json` - session, capture, processing, detection, tracking, motion,
  depth, segmentation, terrain, mapping, cloud, YOLOP, evidence, per-module
  statuses, reconstruction, latency percentiles, errors and the `honesty` block;
- `events.jsonl`, `detections.jsonl`, `tracks.jsonl`, `motion.jsonl`,
  `evidence.jsonl`, `status.jsonl` - per-frame audit rows;
- `last_perception.npz` - the last real depth (`rel_inv`, `depth_m`, `depth_valid`),
  segmentation (`seg_label`) and BEV (`bev_height`, `bev_trav`, `bev_conf`) arrays,
  so a view's numbers can be re-checked after the run;
- `session.log`, `snapshots/*.png` + `*.json`, `recording.mp4` (when `R`);
- `reconstruction/take_NNN/frames/*.png`, `frames_manifest.json`, COLMAP outputs
  and `sparse_points_native.ply` after a run.

The `honesty` block records `live_camera_connected` (this process decoded frames
from the configured source), `frames_read_from_configured_source`, that no
vehicle runtime was imported, that no metric depth/terrain/map/metric trajectory
was reported (`depth_metric_reported: False`,
`terrain_or_traversability_reported: False`), that the reported depth is the
relative model output with `depth_scale_basis` naming the assumed camera height,
that the segmentation is `segmentation_is_pretrained_ade20k_not_traversability`,
that `learned_traversability_trained: False` and that the point cloud
`cloud_is_single_frame_assumed_scale`, that no IMU input path exists, that the
reconstruction is not real time, and `fabricated_visuals_drawn: 0`, plus the list
of unverified assumptions.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | finished, and at least one frame was read from the configured source |
| 2 | invalid configuration (bad flag or value); nothing was started |
| 3 | the run never decoded a single frame from the configured source |
| 4 | a native window was expected but unavailable, so the run degraded to headless |
| 130 | interrupted by the operator |

## Verification status and limits

- 114 focused tests cover this workbench: `tests/test_live_viewer.py` (96, plus
  the 19 IP-webcam runner tests) for URL precedence and `--url-file`
  re-resolution, the bounded newest-wins hand-off, provider errors fabricating
  nothing, real ledger/doubt/adapter states, motion calibration classes, the
  reconstruction manifest and gates, keys, reports and exit codes; and
  `tests/test_live_viewer_pretrained.py` (18) for the integrated depth,
  segmentation, geometric-terrain and cloud views, per-stage error isolation, the
  run artifacts and the local-only loading rule. Capture, the GUI and the torch
  stages are injected or mocked: those suites open no camera, no window and no
  socket, and load no checkpoint.
- `tools/run_prism_pretrained_smoke.py` runs the real downloaded weights on real
  clip frames and is the evidence for the numbers below; it writes PNG previews
  and `smoke_report.json` under `output/pretrained_smoke_tests/<UTC>/` and exits
  non-zero if any check fails.
- The viewer was also run bounded and headless with the real models and every
  socket in the process blocked (`socket.connect`/`create_connection`/
  `getaddrinfo`), to prove startup needs no network: depth and segmentation both
  reached `ACTIVE` from `model_assets/pretrained/...` with
  `local_files_only: true`, the terrain view reached `EXPERIMENTAL_GEOMETRIC` and
  the report kept `depth_metric_reported: False`.
- **Measured CPU latency, not a frame rate claim.** On this host (CPU-only torch,
  640x360 inference) a single depth pass took p50 5.7 s (max 5.8 s in that run,
  up to 16 s under load), segmentation p50 4.1 s (max 6.4 s), detection p50 2.8 s,
  motion p50 2.5 s and the geometric BEV fuse 68 ms. The heavyweight stages are
  therefore sampled (`--depth-interval` / `--seg-interval` / `--map-interval`) and
  the window keeps painting the last real result with its age; nothing here runs
  at 30 FPS with the models enabled.
- **No physical camera was connected while building or testing this workbench.**
  Neither the phone stream nor a PC webcam was opened by the agent; live results
  were exercised through mocked capture and through clip frames replayed through
  the real pipeline. Treat `live_camera_connected` as evidence about a run you
  performed, not about this repository's testing.
- Detection latency is dominated by the CPU LibreYOLO baseline (p50 408 ms in the
  recorded evaluation, 2.8 s for these clip frames), so annotations lag the live
  frame; the views and the report always show that age instead of pretending
  otherwise.
- A failed optional module (YOLOP, Norfair, COLMAP, Open3D, the checkpoint, a
  missing pretrained directory) is recorded as an error and makes that capability
  unavailable; it never aborts the session and never draws a substitute.
- Advisory only: nothing in this tool sends a command to a drone, flight
  controller, motor or vehicle, and no hazard score is invented from detections.
