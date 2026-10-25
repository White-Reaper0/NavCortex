# DRISHTI build contract

Every module in this repo is written against this file. Read it fully before writing code.
**Do not modify `drishti/config.py`, `drishti/types.py`, `drishti/io_utils.py`,
`drishti/viz_common.py`, or `drishti/perception/geometry.py`** — they are the shared
contract and other agents depend on them byte-for-byte. If you believe one needs a
change, add your own helper in your own file instead, and say so in your final report.

## What DRISHTI is

Camera-only navigation perception for an off-road UGV, demonstrated offline on five
10-second clips cut from a POV RC-car video. The point of the demo is the contrast with
a YOLO-style detector: instead of "what object is this?", DRISHTI answers
"can the vehicle drive here, how high is the terrain, how confident are we, and what
happens if we take this path?"

Context: this reproduces the perception/planning stack described in the team's speaker
packet (`context/context.pdf`), whose baseline is LARIAD Offroad-Nav
(Marsal et al., arXiv:2604.03096) with inverse-depth rescaling from arXiv:2412.14103.
Depth Anything V2-Small, PIDNet-S, ORB-SLAM3-style VO are the named components.

**Honesty rules — these matter and will be checked.**
- Never label anything "measured on a UGV" or "real-time on CPU" unless this repo
  actually measured it. Report what was measured, on what hardware.
- Metric scale comes from an *assumed* camera height (`CFG.cam.height_above_ground_m`).
  Anywhere metric numbers are shown to a viewer, they must be traceable to that
  assumption. Don't present them as calibrated ground truth.
- Confidence maps are not calibrated collision probabilities. Label them as confidence.
- The models here are trained on pseudo-labels derived from this footage plus an
  ADE20K-pretrained teacher — **not** on RELLIS-3D / GOOSE / ORFD ground truth, because
  those datasets are not available offline in this environment. Any place you name a
  dataset, name what was actually used. Write "GOOSE-style taxonomy", not "trained on GOOSE".

## Environment

- Windows 11, Python 3.10, `torch 2.11.0+cu128`, CUDA available (RTX 4050 Laptop, 6 GB).
- `cv2 4.13`, `numpy 2.2`, `transformers 5.9`, `onnxruntime 1.23`, `stable_baselines3 2.7`,
  `gymnasium 1.2`, `scipy`, `sklearn`, `matplotlib`. **`timm` and `einops` are NOT installed** —
  do not import them. Do not pip install anything without saying so in your report.
- Offline HF cache already contains:
  `depth-anything/Depth-Anything-V2-Small-hf`, `nvidia/segformer-b0-finetuned-ade-512-512`,
  and torchvision `mobilenet_v3_small` ImageNet weights. Network is available but slow;
  prefer these.
- Working dir is `D:\HARSHIT\UVG`. Run python as `python -m drishti.<module>` or
  `python tools/<script>.py` from that directory.
- **GPU is shared between agents.** Keep VRAM under ~1.5 GB, use batch size 1–8, and
  free tensors. If you hit CUDA OOM, retry on CPU rather than fighting for memory.

## Data

- `clips/clip_01..clip_05.mp4` — 1280x720, 30 fps, 300 frames each. `clips/clips.json`
  has per-clip scene / lighting / stress notes.
  - clip_01, clip_02: daylight gravel park trails with grass banks (off-road)
  - clip_03: dusk earth path past a brick wall
  - clip_04, clip_05: low light, parked vehicles, kerbs, a pedestrian (stress cases)
- The RC chassis fills the bottom of every frame and a channel watermark sits top-left.
  Use `io_utils.ego_mask()` and never treat those pixels as scene geometry.
- Canonical processing resolution is `PROC_W x PROC_H` = 640x360 for every dense map.

## Frames and conventions

- camera frame: X right, Y **down**, Z forward (OpenCV).
- vehicle frame: X right, Y **forward**, Z up, origin on the ground under the camera.
- BEV grid: `CFG.bev` -> 128 rows x 192 cols at 0.06 m/cell = 7.68 m forward, ±5.76 m
  lateral. **Row 0 is the farthest forward cell, row 127 is at the vehicle. Column 96 is
  the centreline.** Helpers: `geometry.bev_indices`, `bev_to_veh`, `veh_to_bev_px`.
- Depth is optical-axis depth in metres. Relative inverse depth from the network is `q`.
  `geometry.fit_metric_ground(...)` -> `GroundFit(a, b, normal, height, ...)`;
  `geometry.depth_from_q(q, fit)` -> `(depth_m, valid)`.

## Taxonomies (fixed)

- Terrain, DRISHTI-7, `CFG` `TERRAIN_CLASSES`:
  `0 sky, 1 trail, 2 grass, 3 rough_veg, 4 obstacle, 5 water, 6 dynamic`; 255 = ignore.
- Traversability, 4 classes: `0 safe, 1 risky, 2 obstacle, 3 unknown`.
- Decisions: `GO, SLOW, REROUTE, STOP`. Actions: `FORWARD, LEFT, RIGHT, SLOW, REROUTE, STOP`.

## Module interfaces you must honour

All dataclasses live in `drishti/types.py`. Populate the fields you own; leave the rest.

| Stage | Module you write | Produces |
|---|---|---|
| depth | `drishti/models/depth.py` (owned by the lead, already specified) | `DepthResult` |
| terrain | `drishti/models/seg_pidnet.py`, `seg_teacher.py` | `SegResult` |
| traversability | `drishti/models/traversability.py` | `TraversabilityResult` |
| uncertainty | `drishti/models/uncertainty.py` | `UncertaintyResult` |
| odometry | `drishti/perception/odometry.py` | `OdometryResult` |
| place recog | `drishti/models/vpr.py` | `PlaceResult` |
| mapping | `drishti/perception/mapping.py` | `BEVMap` |
| lidarize | `drishti/perception/lidarize.py` | point cloud / ring-scan arrays |
| world model | `drishti/models/world_model.py` | `WorldModelPrediction` |
| planner | `drishti/nav/planner.py`, `supervisor.py` | `Trajectory`, `Decision` |
| RL | `drishti/nav/rl_env.py`, `train_rl.py` | PPO policy checkpoint |

Each stage exposes a class with this shape:

```python
class XxxStage:
    def __init__(self, device: str = "cuda", **kw): ...
    def reset(self) -> None: ...                 # clear temporal state between clips
    def __call__(self, packet: FramePacket) -> FramePacket: ...   # fills its own field
```

`reset()` is mandatory for anything with temporal state (VO, mapping, VPR, uncertainty).

## Renderers

Each stage owns a renderer in `drishti/render/r_<name>.py` exposing:

```python
def render(packet: FramePacket, state: dict) -> np.ndarray   # returns BGR 1280x720
```

`state` is a per-clip mutable dict you may use for history (trajectories, sparklines).
Use **only** `drishti/viz_common.py` primitives for chrome, colour and type so all eleven
output videos read as one product: `canvas, panel, blit, text, legend, colorbar,
bar_meter, badge, decision_badge, sparkline, header, footer, colorize_*, draw_grid_dots,
rounded_note`. Palette is dark (`BG/PANEL/EDGE/TEXT/ACCENT`).

Every stage video must be **self-explanatory**: a title, what the model is, what the
colours mean (legend or colorbar), the numbers that matter, and one short "why this
matters for navigation" note. Assume a judge who has never seen the system.

## Output layout (hard requirement)

```
output/<stage_dir>/clip_01.mp4 ... clip_05.mp4     # 5 clips in EVERY folder
```
Stage dirs are exactly `CFG` `OUTPUT_STAGES` (01_depth_anything_v2 ... 11_final_dashboard).
All output videos are 1280x720 @ 30 fps, 300 frames, H.264, written with
`io_utils.VideoWriter`.

## Caching

Perception runs once; renderers read from cache. Use
`io_utils.save_stage(clip_id, stage, **arrays)` / `load_stage(clip_id, stage)`.
Stage names: `depth, seg, geom, trav, unc, odom, vpr, bev, wm, plan`.
Store per-frame arrays stacked along axis 0, float16 where precision allows, so a clip
stays well under ~1 GB.

## Efficiency requirements (must be real, and measured)

The pitch is CPU-friendly deployment. So: shared MobileNetV3-Small trunk where possible,
knowledge distillation from the heavy teacher into the small student, low-resolution
inputs, FP16 on GPU, and an ONNX INT8 export path with a measured CPU latency table
written to `output/benchmarks.json`. **Measure, don't assert.** Report the hardware.

## Deliverable per agent

1. Working code, no placeholder/TODO paths left in the execution path.
2. A `if __name__ == "__main__":` self-test that runs standalone and prints something
   verifiable (shapes, ranges, timings).
3. A short report back: what you built, what you measured, what is faked or approximated
   and why, and anything the lead must wire up.

## Navigation additions (SIH26126 presentation)

Added after the perception build, on top of the contract above. None of them edit
the frozen modules: goal-layer settings live in `drishti/nav/nav_config.py`, and
vehicle profiles are applied to the live `CFG` at start-up by `drishti/vehicles.py`.

| Concern | Module | Produces / does |
|---|---|---|
| fit validity | `drishti/perception/fit_validity.py` | mask ANDed into `DepthResult.valid` where the ground fit is unreliable |
| movers | `drishti/perception/dynamic_layer.py` | `DynamicUpdate`; inflated cells that expire in < 1 s, written into the planner state |
| Point B | `drishti/nav/goal_planner.py`, `goal_map.py` | `GoalStatus` (mode, steer point, D* Lite path, recalled state) |
| replanning | `drishti/nav/dstar_lite.py` | incremental shortest path on the world-fixed cost grid |
| rover IMU | `drishti/perception/imu_fusion.py` | scale/heading-corrected `d_trans`, `d_yaw` |
| loops | `drishti/perception/pose_graph.py`, `loop_closure.py` | verified loop edges, optimised keyframe poses |
| live loop | `drishti/runtime.py` | `DrishtiNavigator.step(bgr, t) -> NavOutput(v, w, decision, ...)` |
| ROS 2 | `ros2/drishti_ros/` | `drishti_node` (Image in, Twist out), Gazebo Harmonic world, mission monitor |

Same honesty rules apply. The Point A -> B numbers come from `tools/sim_goal_nav.py`,
which renders the 2.5-D state from a ground-truth map with no perception networks in
the loop; report them as decision-level results, never as perception or field results.
Goal-level stops use rule ids `G0` (arrived), `G1` (no route) and `G2` (search still
running) alongside the supervisor's `R1`-`R8`.
