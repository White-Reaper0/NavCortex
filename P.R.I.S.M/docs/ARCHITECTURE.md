# DRISHTI — technical architecture

A walkthrough of what actually happens between a video frame arriving and a motion
decision leaving, with the coordinate conventions, the metric-scale derivation, the
safety rule table and the training recipes.

This document uses the packet's own discipline: **Planned / Running / Measured**.
Everything below is *Running* (code in this repository that executes) or *Measured*
(a number this repository produced). The one exception is marked where it appears: the
ROS 2 / Gazebo Harmonic package (§4.19) is written but has not been executed in this
repository's build environment. Where a component of the design in the SIH26126
presentation is
absent, it is named as absent.

Contract: [`CONTRACT.md`](../CONTRACT.md). Shared types: [`drishti/types.py`](../drishti/types.py).
Configuration: [`drishti/config.py`](../drishti/config.py).

---

## 1. Data flow: frame to decision

```
clip frame 1280x720                                     io_utils.read_frames
   └─ resize to PROC 640x360, BGR uint8                 FramePacket.rgb
        │
        ├─(1) DepthStage ─────────────► relative inverse depth q  (518x518 ViT)
        │        └─ fit_metric_ground(q, valid, seg.label)  ──► GroundFit(a, b, n, h)
        │        └─ depth_from_q                             ──► DepthResult.depth_m
        │        └─ FitValidityTracker                       ──► unreliable fit/tiles -> valid=False
        │
        ├─(2) SegStage (PIDNet-S) ────► SegResult.label / prob_max / entropy
        │
        ├─(3) geometry ───────────────► points_cam, points_veh, height, slope, roughness
        │
        ├─(4) TraversabilityStage ────► TraversabilityResult.prob / label / risk
        │        (shared MobileNetV3-Small trunk, 17-channel input stack)
        ├─(5) UncertaintyStage ───────► UncertaintyResult.depth_conf / seg_conf / fused_conf
        │        (same trunk forward, reused features + MC-dropout head)
        │
        ├─(6) OdometryStage (ORB) ────► OdometryResult.pose / d_trans / d_yaw / tracking_ok
        │        └─ ImuFusion (rover) ───────────────────► scale- and heading-corrected motion
        ├─(7) VPRStage (GeM) ─────────► PlaceResult.descriptor / is_revisit
        │        └─ LoopClosureManager (rover) ──────────► PnP-verified loop -> pose graph
        │
        ├─(8) MappingStage ───────────► BEVMap (height, trav_prob, conf, age, hits, terrain)
        ├─(9) LidarizeStage ──────────► PointCloud + RingScan (visualisation product)
        │
        ├─(9b) DynamicObstacleLayer ──► inflated, sub-second-expiry cells for movers
        ├─(9c) GoalPlanner ───────────► open ground: steer for Point B
        │        └─ GlobalCostGrid + D* Lite ────────────► boxed in: steer along the route
        │
        ├─(10) WorldModelStage ───────► 6 actions x 6 steps of latent rollout (optional)
        ├─(11) Planner ───────────────► 17 candidate Trajectory objects, scored
        ├─(12) PolicyStage (PPO) ─────► one preferred action (optional, advisory)
        └─(13) Supervisor ────────────► Decision(kind, action, speed_mps, reason, rule)
```

Stages 1–9 are *perception*: they answer "what is out there and how much do we believe
it". Stages 10–13 are *decision*: they answer "what should the vehicle do and is that
allowed". The split matters because the supervisor can veto the policy without any
perception rerun.

**Execution model.** Offline, perception runs once per clip and is cached
(`io_utils.save_stage` → `work/cache/<clip>/<stage>.npz`); every renderer reads from
cache. This is why the eleven output videos are cheap to regenerate and why the
benchmark numbers in [`output/benchmarks.json`](../output/benchmarks.json) are
per-stage rather than per-video. **Live**, `drishti/runtime.py` (`DrishtiNavigator`)
runs the same stages frame by frame in the order above and returns a velocity command
and is wrapped as a ROS 2 node (both in §4.19).

---

## 2. Coordinate frames and conventions

| Frame | Axes | Origin |
|---|---|---|
| image | u right, v down | top-left pixel |
| camera | X right, **Y down**, Z forward (OpenCV) | optical centre |
| vehicle | X right, **Y forward**, Z up | on the ground directly under the camera |
| BEV grid | row 0 = farthest forward, row 127 = at the vehicle, col 96 = centreline | — |

`geometry.vehicle_basis(normal)` builds the camera→vehicle rotation from the fitted
ground normal (up = −n, forward = the camera Z axis projected onto the ground plane,
right = forward × up), and `geometry.to_vehicle` then lifts the origin from the camera
to the ground by adding `fit.height`.

**BEV grid** (`CFG.bev`): 128 rows × 192 columns at 0.06 m/cell = **7.68 m forward,
±5.76 m lateral**. The row order is inverted on purpose — row 0 is the far field — so
the grid can be drawn directly as a top-down image without flipping.
`geometry.bev_indices` maps vehicle-frame points to `(row, col, keep)`;
`geometry.bev_to_veh` inverts it at cell centres; `geometry.veh_to_bev_px` maps metres
into a drawn panel rectangle.

**Intrinsics** are a documented assumption, not a calibration: 92° horizontal FOV at
640×360 gives `fx = fy = (640/2)/tan(46°) = 309.0 px`, `cx, cy = 320, 180`.

**Ego mask.** The RC chassis fills the bottom 18.5 % of every source frame and a channel
watermark sits top-left. `io_utils.ego_mask()` returns True only where the pixel shows
the world. Those pixels are excluded from depth back-projection, the ground fit, VO
feature extraction, BEV accumulation and every reported statistic.

---

## 3. Metric scale: the derivation, and why it is only *almost* identifiable

This is the most interesting result in the geometry code and it is worth stating
precisely. The full argument lives in the module docstring of
[`drishti/perception/geometry.py`](../drishti/perception/geometry.py).

Depth Anything V2 emits an **affine-invariant relative inverse depth** `q`. It is not
metres. Following the inverse-depth rescaling of Marsal et al. (arXiv:2412.14103),
true optical-axis depth `D` satisfies

```
1 / D = a·q + b
```

The paper's reference for `(a, b)` is sparse metric landmarks from a **visual-inertial**
estimator. This footage has no IMU and no calibration file, so DRISHTI substitutes the
**ground plane**: the drivable surface ahead is a plane at a known distance below the
camera — known because we *assume* `CFG.cam.height_above_ground_m = 0.12 m`.

For a pixel with normalised ray `m = K⁻¹[u v 1]ᵀ` (so `m_z ≡ 1`), the 3-D point is
`P = D·m`. A ground point lies on the plane `n·P = h` with `|n| = 1`, hence

```
n·m / h = 1/D = a·q + b        ⇒        u·m − a·q − b = 0,    u := n/h
```

### The identifiability argument

On a **single** plane, `1/D` is an exact affine function of `(m_x, m_y)`. The data
therefore supply only **three** independent numbers — the coefficients of `q` regressed
on `m_x`, `m_y`, `1` — against **four** unknowns: two plane angles, `a`, and `b`. One
plane at a known height cannot separate scale from shift.

Worse, the naive design matrix `[m_x, m_y, m_z, −q, −1]` is not merely under-determined,
it is **exactly rank-deficient**: `m_z ≡ 1`, so the third column duplicates the constant
column. Any solver that stacks it will return numerical noise, not a poorly conditioned
estimate. This is a structural property of the parameterisation, not a data problem, and
it does not go away with more pixels.

### How DRISHTI resolves it

By adopting the **scale-only model `b = 0`**, i.e. treating the network output as scaled
disparity where `q → 0` means range → ∞. That leaves the well-posed homogeneous system

```
[m_x, m_y, 1, −q] · [u_x, u_y, u_z, a]ᵀ = 0
```

whose one-dimensional null space is the solution, fixed absolutely by `|u| = 1/h_cam`.
It is solved by SVD over up to 6000 candidate ground pixels (lower half of the image,
restricted to `trail`/`grass` when a segmentation label is available) and refined by six
iterations of Cauchy IRLS with scale `2.5·1.4826·MAD`.

Sanity gates before a fit is accepted: plane tilt ≤ 45° from vertical-down, ≥ 120
inliers, `a > 0`. A rejected fit falls back to the previous frame's fit. Accepted fits
are temporally smoothed with `w_new = 0.35` on `a`, `b` and the normal.

### What this costs, stated plainly

Two assumptions are baked into **every** metric number downstream:

1. **`h_cam = 0.12 m`.** All distances, heights, clearances, step heights and speeds are
   *directly proportional* to this. Halve it and every metre in the demo halves. It is
   an assumption about an uncalibrated action-cam POV, not a measurement.
2. **`b = 0`.** This mainly costs accuracy at long range, where monocular depth is least
   trustworthy anyway. `GroundFit.residual` (RMS of `u·m − a·q` in 1/m over inliers)
   reports how well the recovered plane actually explains the observed relative depth,
   and is surfaced in the depth renderer.

`depth_from_q` additionally clamps to `0.15 m < D < 25 m` and marks everything outside
invalid, so a degenerate fit produces *unknown*, never a confident wrong number.

---

## 4. Modules: inputs, outputs, state

Every stage follows the contract shape: `__init__(device=...)`, `reset()`, `__call__(packet) -> packet`.

### 4.1 Depth — `drishti/models/depth.py`

* **Model**: Depth Anything V2-Small (ViT-S/14 DINOv2 + DPT head), 24.8 M parameters,
  **pretrained, downloaded weights** (`depth-anything/Depth-Anything-V2-Small-hf`).
  Not trained or fine-tuned here.
* **In**: `packet.rgb` (360, 640, 3) uint8 BGR. Resized to 518×518, ImageNet-normalised.
* **Out**: `DepthResult(rel_inv (360,640) f32, depth_m f32, scale=a, shift=b,
  align_residual, align_inliers, valid bool)`.
* **State**: previous `GroundFit` (used as a temporal prior) and previous `q`
  (EMA, `temporal_alpha = 0.35`, because per-frame relative depth flickers on 30 fps
  handheld-style footage and that shakes the metric fit). `reset()` clears both.
* **Cache** `depth`: `q`, `depth` f16, `valid` u8, `scale`, `shift`, `residual`,
  `inliers`, `normal` (300,3), `ms`.
* **Fit validity** (`drishti/perception/fit_validity.py`): where the ground fit is
  unreliable, the region is marked invalid and becomes UNKNOWN downstream.
  *Frame level*: a failed fit may reuse the last good one for ≤ 15 frames (0.5 s); after
  that, or with no good fit yet, the whole frame is invalid (previously a nominal plane
  was used and its depths published as valid). *Region level*: the lower half is cut
  into 8×4 tiles; in a tile with ≥ 150 trail/grass pixels, a median relative
  inverse-depth disagreement with the fitted plane above 0.30 invalidates the tile.
  The cached videos predate this gate; the live runtime applies it.

### 4.2 Terrain — `drishti/models/segmentation.py`, `seg_pidnet.py`, `seg_teacher.py`

* **Student**: PIDNet-S written from scratch in plain PyTorch (no `timm`, no `einops`),
  m=2 n=3 planes=32 ppm=96 head=128, PAPPM + Light_Bag — 7.6 M parameters. **Trained in
  this repository** by distillation; there are no ImageNet-pretrained PIDNet weights
  offline, so it starts from random init.
* **Teacher**: SegFormer-B0 finetuned on ADE20K (`nvidia/segformer-b0-finetuned-ade-512-512`),
  **pretrained, downloaded**. Its 150 ADE classes are projected to DRISHTI-7 through an
  explicit, row-normalised 150×7 matrix built by class *name*, with documented soft
  splits (tree → 0.60 rough_veg + 0.40 obstacle; car → 0.60 obstacle + 0.40 dynamic; …).
  Pixels whose retained mass < 0.35 abstain (label 255).
* **In**: `packet.rgb`. Student input 512×288 (`CFG.SEG_INPUT`), FP16 on GPU.
* **Out**: `SegResult(label (360,640) u8, prob_max f32, entropy f32 normalised by log 7,
  logits optional f16)`. Ego pixels forced to `label=255, prob_max=0, entropy=1`.
* **State**: EMA over the (7, H, W) probability tensor, `ema = 0.6`. `reset()` clears it.
* **Cache** `seg`: `label`, `prob_max`, `entropy`, `shares` (300,7), `drivable`, `ms`, `meta`.

### 4.3 Geometry — `drishti/perception/geometry.py` (frozen contract module)

`unproject` → `to_vehicle` → `height_slope_roughness`, producing
`GeometryResult(points_cam, points_veh, height_above_ground, slope_deg, roughness,
ground_normal, ground_d, ground_inlier_frac)`. Roughness is the local height standard
deviation in a 9×9 window computed with box filters over a validity mask; slope is
`atan(|∇z|)` in metric vehicle coordinates, median-filtered. Classical, no learning.

### 4.4 Traversability — `drishti/models/traversability.py`

* **Model**: `SharedPerceptionTrunk` = torchvision MobileNetV3-Small (**ImageNet
  weights**, cached locally) with the stem inflated from 3 to **17 input channels**, plus
  a light top-down FPN to stride 4, followed by a thin head (3×3 → BN → Hardswish → two
  1×1 convs). The trunk is **shared with uncertainty**: one backbone forward serves both
  heads. That sharing is the whole CPU-efficiency argument, so it is measured as one
  graph in `checkpoints/onnx/trav_unc_shared.onnx`.
* **In**: the documented 17-channel stack at 320×180 (= PROC/2):
  0–2 RGB (ImageNet-normalised, *absolute* not per-frame, so a dark frame stays dark and
  the uncertainty head can see the illumination it is meant to distrust);
  3 inverse depth `clip(0.5/D, 0, 1)`; 4 height above ground `clip(h,−0.45,0.90)/0.90`;
  5 slope `clip(deg,0,45)/45`; 6 roughness `clip(m,0,0.10)/0.10`; 7 geometry-valid mask;
  8–14 DRISHTI-7 one-hot; 15 seg max-probability; 16 seg normalised entropy.
  One-hot and mask channels are resized NEAREST, the rest INTER_AREA.
* **Out**: `TraversabilityResult(prob (4,360,640) f32, label u8, risk f32)`; ego pixels
  forced to `UNKNOWN`, risk 0.
* **State**: previous `GroundFit`; cached trunk features + `(clip_id, idx)` key so the
  uncertainty stage can reuse them. `reset()` clears all of it.
* **Supervision**: geometric + semantic **pseudo-labels**, never ground truth. Every
  accuracy number for this stage is *agreement with those pseudo-labels*.

### 4.5 Uncertainty — `drishti/models/uncertainty.py`

A **learned** confidence estimator, not softmax entropy re-badged. Two self-supervised
targets generated offline:

* **Depth error** — temporal geometric + photometric inconsistency. Frame `t−1`'s metric
  depth is unprojected, moved into frame `t` with the VO relative pose, reprojected and
  z-buffer splatted:
  `e_depth = 0.7·|D_t − warp(D_{t−1})| / max(D_t, 0.2) + 0.3·|I_t − warp(I_{t−1})| / (mean + 12)`.
* **Terrain error** — `e_seg = 0.6·blur(student ≠ teacher) + 0.4·teacher_entropy`.
  (Measured student/teacher disagreement: 0.17 on daylight clips, 0.29 on low-light.)

At inference the head sees **only the current frame's feature stack** — no `t−1`, no
teacher, no odometry. It has to have learned what "about to be wrong" looks like.

`conf = clip(exp(−(K_ERR·clip(err,0,3) + K_MC·clip(σ,0,1))), 0, 1)` with `K_ERR = 3.0`,
`K_MC = 6.0`. `σ` is the standard deviation over `n_mc = 8` Monte-Carlo dropout passes
**of the head only** — the trunk runs once, so the epistemic channel costs a few hundred
microseconds. Fusion is a geometric mean, `fused = depth_conf^0.5 · seg_conf^0.5`.
Temporal EMA `0.55`.

**These are confidence maps, not calibrated collision probabilities.** Low confidence
maps to SLOW / REROUTE in the supervisor; it never maps to a confidently wrong answer.

### 4.6 Visual odometry — `drishti/perception/odometry.py`

Classical, no learning: CLAHE(2.5, 8×8) → grid-bucketed ORB (5×4 cells, starved cells
re-detected with relaxed FAST) → BFMatcher Hamming with Lowe ratio 0.78 in both
directions plus mutual cross-check → `findEssentialMat` with USAC_MAGSAC (1.5 px,
conf 0.9999) → `recoverPose` → Sampson residual and median-parallax gates.

**Scale** is the interesting part, because a monocular essential matrix gives direction
only. Two estimators run: a robust median-ratio fit between triangulated and metric
depth (6 Cauchy IRLS iterations, 0.20–8.0 m band), and a 1-D weighted least-squares
reprojection fit for the baseline. The reprojection fit is accepted when
`s > 0 ∧ SE < 0.45 ∧ n ≥ 25`, otherwise the ratio fit is used; the result is clipped to
0.30 m/frame, passed through a 7-sample running median, then an EMA (`α = 0.25`).
Sources are reported explicitly: `SCALE_DEPTH / SMOOTHED / CONST / NONE`.

Three track states — `OK / COAST / LOST`. LOST (matches < 28, E-inliers < 22, parallax
< 0.08°, |yaw rate| > 150 °/s) **holds** the pose and sets `tracking_ok = False`, which
the supervisor turns into an unconditional STOP (rule R1).

* **Out**: `OdometryResult(pose, d_trans, d_yaw, speed_mps, n_matches, n_inliers,
  tracking_ok, track_quality, keypoints, flow, trajectory)`.

> This is **depth-anchored monocular VO**, not ORB-SLAM3. There is no IMU in the
> footage, so there is no monocular-inertial initialisation, no loop-closing back end and
> no bundle adjustment. Calling it "ORB-SLAM3-style front end" is the strongest honest
> description. For the rover, IMU fusion (§4.17) and a pose-graph loop-closing back end
> (§4.18) sit after this stage; neither is used on the footage.

### 4.7 Place recognition — `drishti/models/vpr.py`

MobileNetV3-Small truncated at `features[:9]` (48 ch, stride 16), **frozen ImageNet
weights**, → 1×1 conv 48→256 + BN + ReLU → GeM pooling (learnable `p`, clamped 1–10,
init 3.0) → 256×256 learned whitening → L2 normalise. ~0.4 M parameters of which
~78 k are trained. Input 320×180.

Retrieval: cosine similarity against the visited database, excluding the last 45 frames
of the same clip, top-12 re-scored by a SeqSLAM-style aligned 5-frame mean; a revisit is
declared when the sequence score ≥ 0.86 **and** three consecutive frames have agreed.

* **Out**: `PlaceResult(descriptor (256,), best_match_idx, best_score, is_revisit,
  loop_gap_frames, db_size)`.
* **State**: the growing database. `reset(keep_db=cross_clip)`.

### 4.8 Mapping — `drishti/perception/mapping.py`

Rolling 2.5-D BEV. Per frame:

1. **Roll** — affine warp of the previous grid into the new vehicle frame from
   `(d_trans, d_yaw)`. Height and support carry a validity channel so bilinear
   interpolation cannot invent a surface; age warps NEAREST. Then `conf *= 0.965`
   (× 0.90 extra when VO is lost, with the motion zeroed so the map does not smear),
   `hits *= 0.965`, `age += 1`, stale cells retired (age > 45 or conf < 1e-3).
2. **Observe** — unproject depth → vehicle frame → cell indices; per-cell **order
   statistics** via one packed radix sort: `TOP_Q = 0.95` for the surface height,
   `SUPPORT_Q = 0.20` for support. The high quantile matters physically: a 0.12 m camera
   looking at a 0.30 m box sees only its *front face*, so the tallest thing the sensor
   can report is the top of that face (measured 0.285 m vs 0.30 m truth on the synthetic
   scene). A mean would report 0.15 m and drive the vehicle into the box.
   Per-pixel weight `= clip(fused_conf, 0.02, 1) · 1/(1 + (range/4)²)`.
3. **Fuse** — confidence-weighted mixtures for height/support/trav; noisy-OR for
   confidence, `conf ← clip(conf + w(1 − conf))`; terrain votes accumulate.

Unknown is explicit: `unknown = ¬observed ∨ conf < CFG.safety.conf_unknown (0.35)`, and
unknown cells get `trav = UNKNOWN`, one-hot UNKNOWN probability and `height = NaN`.

Pure helpers used by the planner: `height_step_map` — a **white top-hat**
(`height − morphological opening` over a 0.9 m window). A raw height map cannot be
thresholded against a 4.5 cm clearance because it carries the per-frame bias of the
monocular ground fit *and* the terrain's own slope, both far larger than the clearance.
The top-hat erases anything wider than the window (a slope) and keeps anything narrower
(a kerb, a wall, a rock) — which is exactly the distinction between "steep" and "a step".

* **Out**: `BEVMap(height (128,192) f32 NaN=unobserved, trav_prob (4,128,192) f32,
  trav u8, conf f32, age f32, hits f32, terrain u8)` plus ride-along `trav_source`
  (`"learned"` when a traversability head fed it, `"geometric"` when it fell back).

### 4.9 LiDAR-like reconstruction — `drishti/perception/lidarize.py`

Two products from the same metric depth: a budgeted point cloud (26 000 points, sampled
through a fixed RNG permutation so the cloud does not boil between frames) and a
synthetic 32×360 ring scan with non-uniform beam elevations (+6° to −26°), first-hit-wins
per (ring, azimuth), and an explicit `fov_mask` marking the ±46° the camera can actually
see. It is a **visualisation and reconstruction product, not a LiDAR**: blind sectors are
drawn as dead zones precisely so nobody reads it as a 360° sensor.

### 4.10 World model — `drishti/models/world_model.py`

`BEVEncoder` (8, 128, 192) → 96-D latent → `LatentDynamics` (2-layer GRU, hidden 192,
**residual** step `s ← tanh(s + W·h)` so "nothing changes" is easy to express) →
`Heads` decoding a 16×16 obstacle occupancy, a mean traversability and a collision risk.
728 610 parameters total.

The point of a *latent* world model rather than generated future video: the planner does
not need pixels of the future, it needs where obstacles will be and how likely a
collision is. Rolling out all six actions for six steps (`WM_DT = 2.0/6 = 0.333 s`, so
six steps span exactly the 2 s safety horizon) costs well under a millisecond on GPU,
which is what makes it usable both as the RL simulator and as a per-frame risk oracle.

### 4.11 Planner — `drishti/nav/planner.py`

A deterministic small-fan sampling rollout planner (explicitly **not** MPPI).
`dt = 2.0/12 = 0.1667 s`, 12 steps, unicycle `arc_poses`. Six base actions × a steering
fan (5 + 3 + 3 + 3 + 2 + 1) give **17 candidates**: 16 moving arcs and standing still.

Cost, exactly:

```
cost = 2.4·wm_risk[a] + 2.0·max(per_step_risk) + 1.3·unknown_frac
     + 0.8·(1 − mean_conf) + 0.55·inv_clearance + 1.1·step_norm + 0.18·|ω|
     − 1.6·progress                         (+100 if infeasible)
```

with `step_norm = clip(max_step / 0.045, 0, 2)`, `inv_clearance = 1/max(min_clearance, 0.08)`
when `min_clearance < 2 m` else 0.

*Progress* has two forms:

* **No goal** (the recorded footage): `progress = xy[-1]·(−sin ψ_goal, cos ψ_goal)` with
  `ψ_goal = 0`, i.e. "keep going forward along the trail".
* **With a goal point** `g` from the goal layer (§4.14), Point B itself or a D* Lite
  look-ahead point: `progress = |g| − min_t |xy_t − g|`, the reduction in distance to
  `g` at closest approach, so overshooting earns nothing. Two more terms apply only here.
  `+2.0` (`idle`) for standing still, because without it a goal *behind* the vehicle
  deadlocks it (every turning arc earns little progress and pays turn and clearance
  terms). And `+1.5·(1 − cos Δψ)/2` (`heading`) for an arc that does not pass through
  `g` and ends facing away from it. Both are costs, not constraints: infeasible arcs
  stay rejected and every supervisor rule still applies.

Hard feasibility gates, first match writes `reject_reason`:

| id | condition | meaning |
|---|---|---|
| R-P1 | `max_step > 0.045 m` | step exceeds chassis clearance |
| R-P2 | `obstacle_frac > 0.5` | footprint mostly on obstacle cells |
| R-P3 | `unknown_frac > 0.62` | driving into unobserved space |
| R-P4 | `v > 0.05 ∧ free_distance < braking_distance(v)` | cannot stop in the space seen |
| R-P5 | `off_map_frac > 0.55` | arc leaves the local map |

`braking_distance(v) = 0.45·(v/1.2)²`.

### 4.12 RL policy — `drishti/nav/rl_env.py`, `train_rl.py`

A PPO policy trained **inside the world model**, never on the vehicle. Observation is
109 floats: 96-D latent, 3 context (mean confidence, unknown fraction, initial
traversability), 3 dynamics (predicted traversability, predicted risk, normalised yaw),
6 last-action one-hot, 1 normalised episode time. Action space `Discrete(6)`.

Reward: `+3.0·step_dist − 2.2·risk − 0.8·(1 − trav) − 0.06·[a ≠ a_prev]
− 0.40·[v < 0.10] − 0.25·|yaw|`, terminal `−8.0` on collision (`risk ≥ 0.78`).
Episodes are 32 steps (10.7 s).

The policy is **advisory**: `PolicyStage` writes `packet._policy_action`, and the
supervisor is free to override it. That is by design — see R1/R2 below.

### 4.13 Safety supervisor — `drishti/nav/supervisor.py`

First-match-wins cascade over the planner's scored candidates. `MIN_USEFUL_SPEED = 0.12`,
`SLOW_FACTOR = 0.42`. Every branch clamps speed to `[0, 1.2]` m/s.

| Rule | Condition | Decision | Action / speed | Threshold source |
|---|---|---|---|---|
| **R1** | `tracking_ok == False` | STOP | STOP, 0.0 m/s, risk 1.0, always overrides | `odom.tracking_ok` |
| **R2** | no feasible candidate | STOP | STOP, 0.0, quotes the lowest-cost candidate's `reject_reason` | — |
| **R2** | best candidate *is* STOP | STOP | STOP, 0.0 — "the supervisor restricts motion, it never forces it" | — |
| **R3** | `max_step > clearance` | STOP | STOP, 0.0, overrides | `CFG.ugv.clearance_m` 0.045 |
| **R3** | a blocked arc exists ∧ chosen ∈ {LEFT, RIGHT, REROUTE} | REROUTE | keep action, speed ≤ 0.72 | `CFG.ugv.clearance_m` |
| **R4** | `risk ≥ 0.78` | STOP | STOP, 0.0, overrides | `CFG.safety.risk_stop` |
| **R4** | `risk ≥ 0.55` | REROUTE | lowest-cost feasible arc with risk < 0.55, speed ≤ 0.84 | `CFG.safety.risk_reroute` |
| **R4** | `risk ≥ 0.30` | SLOW | same action, speed × 0.42 | `CFG.safety.risk_slow` |
| **R5** | `unknown_frac ≥ 0.62` | STOP | STOP, 0.0, overrides | `CFG.safety.unknown_frac_stop` |
| **R5** | `unknown_frac ≥ 0.30` | SLOW | same action, speed × 0.42 | `CFG.safety.unknown_frac_slow` |
| **R6** | `conf < 0.35` ∧ a higher-confidence arc exists | REROUTE | new action, speed ≤ 0.60 | `conf_unknown` 0.35 / `conf_slow` 0.55 |
| **R6** | `conf < 0.35`, no alternative | SLOW | crawl, speed ≤ 0.30 | `CFG.safety.conf_unknown` |
| **R6** | `conf < 0.55` | SLOW | same action, speed × 0.42 | `CFG.safety.conf_slow` |
| **R7** | `braking_distance(v) > free` ∧ safe speed < 0.12 | STOP | STOP, 0.0, overrides | `brake_distance_m` 0.45, `max_speed_mps` 1.2 |
| **R7** | `braking_distance(v) > free` ∧ safe speed ≥ 0.12 | SLOW | speed ← the speed it *can* stop from | same |
| **R8** | all gates clear | GO / SLOW / REROUTE per chosen action | `min(ACTION_CMD[a].v, 1.2)` | — |

`_free_ahead` re-integrates the chosen arc out to the full 7.68 m BEV range rather than
stopping at the 2 s horizon, so R7 is not blind past the planning horizon.

The reported `Decision` carries `rule` (`"R1"`…`"R8"`), a human-readable `reason`, and
`policy_source` ∈ `{"rl", "supervisor", "rl+supervisor"}` so the dashboard can always say
who decided.

**Goal-level stops** are applied by the runtime in front of the supervisor's verdict
(they can only turn a command into STOP, never the reverse):

| Rule | Condition | Decision |
|---|---|---|
| **G0** | within `arrive_tol_m` (0.25 m) of Point B | STOP: arrived |
| **G1** | D* Lite's open list ran dry: no route over the ground seen so far | STOP and hold until something changes (a mover leaves, a cell expires) |
| **G2** | D* Lite hit its per-cycle expansion cap (6,000) | STOP this cycle; the search resumes next cycle |


### 4.14 Goal layer — `drishti/nav/goal_planner.py`, `goal_map.py`, `nav_config.py`

Point A → Point B over a world-fixed grid of "the ground seen so far".

* **GlobalCostGrid**: 60 m × 60 m at 0.12 m (2×2 BEV cells), centred on the start pose,
  in the odometry frame. Every cycle the 8-channel BEV state is folded in (worst
  evidence per cell per frame wins, newest frame wins over older ones). Cost per metre:
  seen-safe 1.0, risky 2.5, **never observed / low confidence 4.0** (finite: unseen
  ground costs, it is never free and never a wall), obstacle or height step above
  clearance ∞. Blocked cells are dilated by 0.16 m (vehicle half-width + margin), then a
  0.5 m soft band raises cost up to 4× towards them, as in a Nav2 inflation layer, so
  routes stay centred in corridors. Evidence older than 30 s reverts to unknown, so a
  drifted old obstacle cannot wall the vehicle in forever.
* **Open ground**: the straight corridor to B (clipped to the local map) is ≥ 70%
  observed with no blocked cell, and the global segment is unblocked. The arcs score
  progress towards B itself.
* **Boxed in**: D* Lite (§4.15) from B to the vehicle, and the arcs steer for the route
  point 1.6 m ahead. Blocking changes (finite ↔ ∞) are pushed into the search at once;
  cost refinements on cells that stay passable are batched once per second, which keeps
  the median cycle near 30 ms.
* **Recall**: the rolling map forgets a cell 1.5 s after it leaves the camera cone. For
  cells that are unobserved *locally*, `recall_into_state` writes back remembered
  evidence (≤ 15 s old) at confidence 0.5, below `conf_slow`. The vehicle can then turn
  round over ground it has seen, but it slows while relying on memory. Live observations
  are never overwritten.
* **Measured** (`tools/sim_goal_nav.py`, 6 scenarios × 3 seeds, ground-truth map rendered
  through the view cone, no perception networks in the loop): **18/18 reached, 0
  collisions, 0 interventions** → `logs/sim_goal_nav.json`.

### 4.15 D* Lite — `drishti/nav/dstar_lite.py`

Koenig & Likhachev (2002), with the `k_m` key modifier and a lazy-deletion heap.
8-connected; edge cost = step length × mean of the two cells' costs. A diagonal step is
refused if it would cut a blocked corner. The heuristic is octile distance × the cheapest
cell cost (admissible, consistent). Keys are compared with a 1e-9 tolerance, because
float ties otherwise ended the repair one vertex early (found by the Dijkstra
cross-check). `compute(max_expansions)` is resumable: `converged` and `exhausted` tell
"still searching" from "provably unreachable". Verified against a from-scratch
Dijkstra through random cost changes and start moves (`tests/test_dstar_lite.py`).

### 4.16 Dynamic layer — `drishti/perception/dynamic_layer.py`

DRISHTI-7 class 6 (person, animal, vehicle) pixels with valid depth are projected into
BEV cells, dilated by the vehicle half-width + 0.30 m, and given an expiry of
`t + 0.8 s` (under one second). The layer rolls with ego-motion like the terrain map.
`apply()` writes active cells into the planner state as certain OBSTACLE. The same
cells are blocked in the global grid until they expire, so D* Lite replans when a mover
appears and again when it leaves. A cell is cleared only by its own expiry, never by a
frame in which segmentation happened to miss the person.

### 4.17 Rover IMU fusion — `drishti/perception/imu_fusion.py`

Not used on the footage (no IMU). On the rover:

* **Heading**: complementary filter. The bias-corrected gyro carries the short term and
  VO the long term; gyro bias is learned from the slow gyro-minus-VO yaw-rate residual
  and during confirmed stops. During VO dropouts the gyro carries heading alone.
* **Scale**: over a 2 s window, changes in VO forward speed are regressed against
  changes in IMU-integrated speed (`s = Σ dv_imu·dv_vo / Σ dv_vo²`). Changes cancel the
  IMU's velocity offset. Windows without acceleration are skipped, because scale is
  unobservable at constant speed. The estimate is EMA-smoothed and clipped to
  [0.4, 2.5].
* **Zero-velocity updates** need the IMU (|a| ≈ g, |ω| ≈ 0) **and** VO (< 0.03 m/s) to
  agree for 6 consecutive frames, or a commanded stop from the runtime. Biases are
  learned only on a commanded stop or after 1.5 s quiet. An earlier version learned
  accelerometer bias during slow acceleration from rest and biased the scale low; the
  gating is the fix.
* **Measured** (synthetic, `python -m drishti.perception.imu_fusion`): a deliberately
  wrong 2× VO scale is recovered to 1.98; heading drift under a 0.01 rad/s gyro bias
  falls from 18° (VO) to 11°.

### 4.18 Pose graph and loop closure — `drishti/perception/pose_graph.py`, `loop_closure.py`

Keyframes every 0.5 m or 0.35 rad, chained by odometry edges. When VPR reports a revisit,
the nearest keyframe to the matched frame is verified by ORB matching against that
keyframe's stored metric depth plus PnP-RANSAC (≥ 25 inliers). A loop edge must then pass
a **drift-consistency gate**: χ² (3 dof, 99.9% → 16.27) of its residual under the
covariance propagated along the odometry chain, inflated ×3 because VO drift is
bias-like. A Cauchy kernel was tried first and rejected, because at the first iteration
a true loop after metres of drift has the same residual as a false one. Backends: GTSAM
(`BetweenFactorPose2`, Levenberg–Marquardt) when importable, else a built-in sparse
Gauss–Newton. Both are tested. **Measured** on a synthetic 16 m square: end-point error
0.71 → 0.01 m (built-in) and 0.87 → 0.03 m (GTSAM), with a planted false loop rejected
by both.

### 4.19 Runtime and ROS 2 — `drishti/runtime.py`, `ros2/drishti_ros/`

`DrishtiNavigator.step(bgr, t)` runs, in order: terrain, depth (+ fit validity),
traversability and trust, odometry, IMU fusion, VPR, 2.5-D map, loop closure, dynamic
layer, goal layer, optional world model / policy, 17 arcs, supervisor. It returns
`(v, w, decision, rule, goal status, pose, timings)`. The vehicle profile is applied
before any stage is constructed. Perception can be injected, which is how the
closed-loop simulator and the tests drive the navigation half without weights.

The ROS 2 Jazzy package wraps it as `drishti_node` (Image in, Imu optional, PoseStamped
goal; Twist out plus a JSON decision topic; 0.5 s camera watchdog). It also provides a
Gazebo Harmonic world matching the `wave_rover` profile, a `ros_gz_bridge` map, a
`mission_monitor` that scores success, collisions and interventions from simulator
ground truth only, and a WAVE ROVER serial bridge with a 0.3 s motor watchdog.
**Written, not yet executed in this repository's environment**; only the pure-Python
conversions are unit-tested.

### 4.20 Vehicle profiles — `drishti/vehicles.py`, `configs/vehicles/*.json`

Camera height, FOV and pitch, plus footprint, clearance, step limit, slope, speed and
braking distance, are written into the live `CFG.cam` / `CFG.ugv` (the frozen
`config.py` is not edited). No model is retrained. `rc_pov` reproduces every cached
result; `wave_rover` is the target; `example_large_ugv` is illustrative and shows the
same 10 cm step judged blocking for one chassis and drivable for the other
(`tests/test_fit_validity_vehicles.py`).

---

## 5. Training recipes

No human annotation exists anywhere in this repository, and RELLIS-3D / GOOSE / ORFD are
not available offline. Every "label" below is a pseudo-label; every accuracy number is
agreement with a teacher or with pseudo-labels.

### 5.1 Terrain distillation — `training/make_seg_dataset.py` + `distill_seg.py`

* **Dataset**: every 2nd frame of the five clips (750) plus 300 extra frames seeked from
  `video/input.mp4` in two ranges (`day` 60–340 s, `lowlight` 420–900 s).
  **Measured totals: 1350 samples, 1181 train / 169 val.** Targets are the teacher's
  soft (7, 72, 128) distribution at stride 4 plus a (144, 256) hard label with 255 ignore.
  Teacher runs in `short512` mode with horizontal-flip TTA.
  Pixel balance: sky 8.98 %, trail 26.33 %, grass 5.19 %, rough_veg 15.20 %,
  obstacle 18.94 %, water **0.040 %**, dynamic 0.54 %, ignore 24.78 %.
* **Optimiser**: AdamW, lr 6e-4, weight decay 0.01, linear warm-up 2 epochs then cosine
  to 1e-3 of base, AMP fp16, grad-norm clip 5.0, batch 8, 40 epochs.
* **Loss**: `2.0·KD + 1.0·OHEM-CE + 0.4·aux(P-branch) + 20.0·weighted-BCE(D-branch vs
  Canny boundary) + 1.0·boundary-aware CE`. KD is temperature-softened KL at `T = 2.0`
  scaled by `T²`, evaluated at the student's stride-8 grid; the hard losses run at
  144×256. OHEM keeps pixels below 0.9 confidence, minimum 1/16 of the batch. Class
  weights are clipped inverse-sqrt frequency, `[0.5, 3.0]`.
* **Measured result** (best epoch 30): teacher-agreement **mIoU 0.6548**, pixel accuracy
  0.8914. Per class: sky 0.851, trail 0.898, grass 0.708, rough_veg 0.748,
  obstacle 0.742, water **0.0043**, dynamic 0.632. Water is effectively unlearned — it
  is 0.04 % of the pixels and the honest reading is that this taxonomy slot is empty in
  this footage.

### 5.2 Traversability + uncertainty — `make_trav_labels.py` + `train_trav_unc.py`

* **Pseudo-labels**, all thresholds derived from `CFG.ugv` / `CFG.bev` and recorded in
  `meta.json`: OBSTACLE when patch-median height > 0.030 m sustained over ≥ 0.0374 m²
  (11 metric BEV cells) or `z > 0.24 m` or `z < −0.45 m` or slope > 30.8° or terrain ∈
  {obstacle, dynamic} not vetoed by geometry; RISKY when `h > 0.015`, `z < −0.060`,
  slope > 13.2°, roughness > 0.015, terrain ∈ {rough_veg, water}, or within 0.16 m of an
  OBSTACLE cell; SAFE when terrain ∈ {trail, grass} and `|h| ≤ 0.015` and slope ≤ 13.2°
  and roughness ≤ 0.015; UNKNOWN for invalid depth, sky, `prob_max < 0.35`,
  entropy > 0.80, depth > 8 m, failed ground fit, or the ego mask.
  **Measured dataset: 248 samples** over 6 sources; class pixel fractions safe 14.19 %,
  risky 18.03 %, obstacle 29.29 %, unknown 37.51 %.
* **Training**: one trunk, two heads, one optimiser. AdamW with two param groups — trunk
  at 7.5e-5 (0.25× multiplier), heads at 3e-4 — weight decay 1e-4, cosine to 0.02× base,
  AMP fp16, clip 5.0, batch 8, 30 epochs, validation = the trailing 22 % of each clip
  (a temporal block, not a random split). Horizontal flip on the whole 17-channel stack.
* **Loss**: `CE(class-weighted, ignore 255) + 0.5·softDice + 0.6·risk + 1.0·unc`, where
  risk is `0.5·L1(σ(r̂), r) + 0.5·BCEWithLogits`, and unc is `SmoothL1(softplus(raw), u,
  β = 0.05)` weighted per pixel.

### 5.3 World model — `training/train_world_model.py`

AdamW, OneCycleLR (`pct_start = 0.15`), grad clip 1.0, rollout unrolled 6 steps with
gradients through the whole unroll. Loss
`1.0·latent + 1.0·occupancy + 0.5·traversability + 1.5·risk + 0.5·ground`, where the
latent term is MSE against a stop-gradient encoding of the true future map, occupancy is
`BCEWithLogits(pos_weight = 1.5) + 2.0·L1`, and risk is BCE against the geometric
footprint sweep. Data are real cached BEV transitions at stride 10 frames (0.333 s)
mixed with explicitly-labelled kinematic warp augmentation for all six actions.
**Measured run**: 400 steps, batch 12, lr 3e-4, real fraction 0.4, 750 anchors /
600 real multi-step transitions, 123 s on GPU. The recorded action histogram is
dominated by STOP (578 of 600) because the source footage is a forward-driving POV with
almost no commanded turning — a real limitation of the demo data, not of the model.

### 5.4 VPR — `models/vpr.py --train`

Self-supervised InfoNCE on the frozen trunk's head only. AdamW lr 3e-3, weight decay
1e-4, cosine schedule, AMP, 400 steps, batch 16, temperature 0.07, symmetric
cross-entropy with false-negative masking (same source and |Δt| ≤ 1.5 s are excluded
from the negatives). Positives are two augmentations of the same frame or a frame within
0.4 s. The only honest metric available is Recall@1 of "an augmented query retrieves a
temporally-near frame of the same source", and that is what is reported.

### 5.5 PPO — `nav/train_rl.py`

stable-baselines3 PPO, MlpPolicy `[64,64]`/`[64,64]`, CPU, seed 1337, `n_steps 256`,
batch 256, 8 epochs, γ 0.98, GAE λ 0.95, clip 0.2, entropy 0.008, lr 3e-4, 8 vectorised
environments. **Measured run: 30 000 steps, 79 s.** Held-out evaluation against
always-forward / random / always-stop: PPO return −14.10 ± 2.79 vs −26.24 / −30.00 /
−35.10; progress 0.88 m vs 10.67 / 1.65 / 0.00 m. **Collision rate is 100 % for every
policy including the baselines** — the initial-state bank came from a single clip's real
BEV maps (48 train / 12 held out) and every state in it eventually contains an obstacle
within the 32-step horizon. The policy is better than the baselines on return; it is not
a collision-free policy and must not be presented as one.

---

## 6. Deployment path

See [`drishti/deploy/`](../drishti/deploy). Four scripts, each independently re-runnable:

* `export_onnx.py` — one fixed-shape ONNX graph per network, each **verified** by
  comparing ONNX Runtime against PyTorch on real clip frames and reporting max absolute
  and max relative error. A graph that does not match numerically is reported as failed.
  Two workarounds live here and are documented at their definitions: a static,
  exactly-equivalent replacement for `AdaptiveAvgPool2d` when the output size is not a
  divisor of the input (the world-model encoder pools 8×12 → 3×4), and a
  three-attempt exporter chain (TorchScript → TorchScript with that rewrite → dynamo).
* `quantize.py` — FP16, dynamic INT8 and static INT8 (QDQ, per-channel weights, uint8
  activations) with a calibration reader that streams a few hundred **genuine frames
  sampled across all five clips**, daylight and low-light both, and an accuracy pass that
  measures the cost of each variant against the FP32 graph on real frames.
* `prune.py` — structured channel pruning of block interiors with an L1 filter-norm
  criterion, a sparsity sweep and a short self-distillation fine-tune.
* `benchmark.py` — the latency table, including the non-network stages, written to
  `output/benchmarks.json` and rendered to `output/benchmarks.png`.

The measured numbers, the methodology, and what is *not* claimed are in the README's
benchmark section and in `output/benchmarks.json`.

---

## 7. What this architecture is not

* Not ORB-SLAM3. On the footage: depth-anchored monocular VO with no IMU, no
  loop-closing back end, no bundle adjustment. On the rover, an IMU complementary
  filter and a 2-D pose graph are added (§4.17–4.18); there is still no bundle adjustment.
* Not trained on RELLIS-3D, GOOSE or ORFD. GOOSE-style *taxonomy*; ADE20K teacher;
  this footage.
* Not a LiDAR. A reconstruction from monocular depth with the camera's blind sectors
  drawn explicitly.
* Not calibrated. Intrinsics are assumed from a nominal FOV; scale is anchored on an
  assumed camera height.
* Not physical autonomy, yet. Perception is offline video. Closed-loop control is
  exercised only in a 2-D decision-level simulator with a ground-truth map (§4.14). The
  Gazebo Harmonic and rover paths are written but not yet run.
