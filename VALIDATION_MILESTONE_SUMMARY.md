# P.R.I.S.M. TUM RGB-D Validation Milestone Summary

## Completed Objectives

### 1. Orientation Metric Discrepancy Resolved

**Issue**: Three different rotation values appeared in reports:
- Progress-log: 0.498°
- Final ATE rotation RMSE: 2.185°
- RPE rotation RMSE: 0.483°

**Resolution**: These are **different metrics on different frame counts**, not a discrepancy:

| Metric | Value | Description |
|--------|-------|-------------|
| ATE rotation RMSE | 2.185° | Absolute orientation error after removing global offset (80 frames) |
| RPE rotation RMSE | 0.483° | Relative frame-to-frame rotation error, alignment-free (80 frames) |
| ATE rotation alignment | 144.69° | Global offset removed (not an error) |
| Progress-log ATE rot | 0.498° | ATE rotation from 6-frame smoke run (fewer frames = less drift) |

**Key Insight**: ATE rotation accumulates drift over more frames, while RPE rotation measures local consistency and is frame-count independent.

**Documentation**: Added comprehensive comments in `prism/benchmarks/trajectory_evaluation.py:348-390` explaining:
- Metric definitions and distinctions
- Alignment conventions (orientation vs position)
- Frame-count effects
- Units and coordinate frames

### 2. Multi-Sequence Infrastructure Added

**Files Modified**:
- `prism/benchmarks/tum_rgbd_adapter.py`:
  - Added constants for freiburg1_rpy and freiburg1_360 sequences
  - Refactored `ensure_freiburg1_xyz` into general `ensure_sequence` function
  - Added `ensure_freiburg1_rpy` and `ensure_freiburg1_360` wrappers
  - Fixed `discover_sequence_root` bug (was returning wrong sequence)

- `tools/run_prism_dataset_validation.py`:
  - Added `--sequences` CLI argument (comma-separated: xyz,rpy,360)
  - Modified `run_validation` to accept `sequence_name` parameter
  - Added `_build_multi_sequence_summary` for aggregating results
  - Per-sequence output files: `tum_rgbd_validation_{sequence}.json`

**Usage**:
```bash
# Single sequence (backward compatible)
python tools/run_prism_dataset_validation.py --sequences xyz --max-frames 80

# Multiple sequences
python tools/run_prism_dataset_validation.py --sequences xyz,rpy,360 --max-frames 80
```

### 3. Focused Regression Tests Added

**New Tests** (3 tests in `tests/test_tum_rgbd_benchmark.py`):

1. `test_ate_rotation_increases_with_frame_count_due_to_drift`:
   - Verifies ATE rotation accumulates drift over more frames
   - Confirms RPE rotation is frame-count independent

2. `test_rpe_rotation_is_alignment_free`:
   - Verifies RPE rotation measures local consistency without global alignment
   - Confirms ATE rotation alignment removes global offset correctly

3. `test_orientation_metrics_handle_missing_rotations`:
   - Verifies orientation metrics are NaN when rotations not provided
   - Tests graceful handling of missing data

**Test Results**: All 176 tests pass (23 TUM benchmark tests + 153 other tests)

### 4. Validation Report Structure

**Current Report** (`output/tum_rgbd_validation_xyz.json`):
- Format: `prism-tum-rgbd-validation/v2`
- Software checks: 8/8 passed
- Geometric checks: 15/15 passed
- RGB-only tracking success: 96.2%
- ATE RMSE: 0.0316 m (after Sim(3) alignment)
- ATE rotation RMSE: 2.185°
- RPE rotation RMSE: 0.483°
- Sim(3) scale: 0.0159 (monocular scale unobservable)

**Evidence Separation**: Report clearly separates:
- Ground truth (used for evaluation only, not fed to VO)
- Measured depth (RGB-D reference checks)
- Detection scores (unmeasured)
- Predictions (unmeasured)
- Traversability (unvalidated)

### 5. Known Limitations

1. **Additional Sequences**: the initial rpy/360 download attempt timed out.
   Both archives were completed on 2026-10-24 (official sizes 410,268,381 and
   417,954,230 bytes) and validated on 80 frames each; measured results are in
   the addendum below and are not all passing.

2. **Monocular Scale**: RGB-only VO has no metric scale; Sim(3) alignment is post-hoc evaluation only.

3. **No Navigation Readiness**: Results do not establish GPS-denied navigation readiness.

4. **Unmeasured Metrics**:
   - Terrain traversability accuracy (no labels)
   - Prediction accuracy (no future events)
   - Detector accuracy (out of scope)

## Commands Executed

```bash
# Run full test suite
.venv/Scripts/python.exe -m pytest tests/ -x -q
# Result: 176 passed in 124.19s

# Run validation on xyz sequence
.venv/Scripts/python.exe tools/run_prism_dataset_validation.py \
  --sequences xyz --max-frames 80 --no-download
# Result: All checks passed
```

## Files Changed

1. `prism/benchmarks/tum_rgbd_adapter.py`:
   - Lines 55-68: Added rpy/360 constants
   - Lines 303-316: Fixed discover_sequence_root bug
   - Lines 694-752: Refactored ensure_sequence
   - Lines 754-799: Added ensure_freiburg1_rpy/360 wrappers

2. `tools/run_prism_dataset_validation.py`:
   - Lines 52-63: Added imports for new ensure functions
   - Lines 666-729: Modified run_validation signature
   - Lines 1021-1070: Updated main() with --sequences argument
   - Lines 1020-1068: Added _build_multi_sequence_summary

3. `tests/test_tum_rgbd_benchmark.py`:
   - Lines 213-270: Added 3 new orientation metric tests

## Next Highest-Value Validation Milestone

**Recommendation**: Live camera testing (final milestone)

**Prerequisites**:
1. Download freiburg1_rpy and freiburg1_360 manually (or wait for network improvement)
2. Run multi-sequence validation to verify rotational motion handling
3. Document per-sequence metrics comparison

**Alternative**: If live camera testing is not yet appropriate:
- Add cross-sequence metric comparison tests
- Validate multi-sequence report aggregation
- Test edge cases (tracking loss, degenerate motion)

## Conclusion

All objectives completed successfully:
- ✅ Orientation metric discrepancy resolved and documented
- ✅ Multi-sequence infrastructure added
- ✅ Focused regression tests added (176 tests passing)
- ✅ Bounded validation command executed
- ✅ Evidence separation preserved
- ✅ No false claims about navigation readiness

The system is ready for the next validation milestone with clear documentation of what has been measured, what remains unmeasured, and what the metrics actually mean.

## Addendum 2026-10-24: multi-sequence runs completed

Two verified defects blocked the earlier multi-sequence attempt and were fixed:

1. `_provenance` hardcoded the XYZ archive name/URLs/size, so rpy/360 archives
   were reported as size mismatches.
2. `extract_archive` searched for the XYZ sequence root, so a non-XYZ extraction
   could never locate its own directory.
3. `_build_multi_sequence_summary` compared status against `"pass"` while the
   reports emit `"passed"`.

Command run:

```bash
.venv/Scripts/python.exe tools/run_prism_dataset_validation.py \
  --sequences xyz,rpy,360 --max-frames 80
```

Per-sequence measured results (RGB-only ORB, post-hoc Sim(3) evaluation):

| Sequence | Tracking | Losses | ATE RMSE (m) | RPE trans (m) | RPE rot (°) | Path ratio | Sim(3) scale | Geometric check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xyz | 96.2 % (76/79) | 3 | 0.0316 | 0.0108 | 0.483 | 1.374 | 0.0159 | passed |
| rpy | 87.3 % (69/79) | 10 | 0.0298 | 0.0053 | 1.023 | 1.387 | 0.0050 | failed `ate_path_fraction` (0.0298 m / 0.247 m = 0.121 > 0.10) |
| 360 | 82.3 % (65/79) | 14 | 0.1057 | 0.0169 | 0.945 | 1.595 | 0.0169 | failed `ate_rmse` (> 0.1 m) and path fraction; mapping interface refused (no support-plane fit) |

These are measured outcomes, not tuning targets: the absolute ATE threshold is
failed only by 360, while rpy fails a path-fraction check that is
ill-conditioned on a 0.25 m reference path. No threshold was changed to make
the aggregate look green.

Three regression tests were added for the fixed defects (sequence-specific
provenance, extraction root selection, `"passed"` aggregation). Full suite
after the additions: **188 passed in 99.68 s**.

## Addendum 2026-10-25: IP Webcam live probe

The environment was recreated (Python 3.11, `torch 2.4.1+cpu`,
`libreyolo 1.6.0`, `opencv-python 5.0.0.93`, `numpy 1.26.4`,
`transformers 4.57.6`, `open3d 0.20.0`, `norfair 2.3.0`).
`tools/run_prism_ipwebcam.py` adds a bounded advisory-only MJPEG probe with
mocked-capture tests. Measured in this environment:

- `tests/test_ipwebcam_runner.py`: **19 passed in 2.33 s**.
- Full suite: **207 passed in 69.99 s**.

The tests mock video capture and TCP probes; no live phone connection was
established for this run, so physical-camera behaviour remains unvalidated.

## Addendum 2026-10-25: native live vision workbench

`tools/run_prism_live_viewer.py` is a native Python/OpenCV desktop application
(no browser dashboard, web server, React or Electron). It opens a PC webcam or
the IP Webcam MJPEG stream and displays, per view, only what this repository can
really run: LibreYOLO 2-D detections, Norfair/IoU image-space tracks, the ORB +
fundamental-matrix motion diagnostics reused from
`prism/visual_odometry_replay.py`, optional YOLOP statistics as numbers, and real
`BeliefLedger` / `DoubtEngine` / `RuntimeAdapter` states. Depth, terrain /
traversability and active mapping are BLOCKED cards that blit the real camera
frame and print the blocking gate's own missing-artifact list; 3-D is the
existing offline COLMAP path driven by one bounded background worker and
reported as `RELATIVE_ONLY` / `METRIC_SCALE_UNVERIFIED`. Each run writes a unique
directory under `output/live_viewer_runs/` with `run_config.json`,
`run_report.json` (including an `honesty` block and `fabricated_visuals_drawn: 0`)
and per-frame JSONL audit rows. Exit codes: 0 normal, 2 invalid config, 3 no
frame ever read, 4 window expected but unavailable, 130 interrupted.

Measured in this environment:

- `tests/test_live_viewer.py`: **96 passed in ~21 s** (mocked capture, mocked
  window, no model load, no socket; the suite also asserts that a headless run
  never triggers the TCP probe).
- `tests/test_live_viewer.py` + `tests/test_ipwebcam_runner.py`: **115 passed in
  21.10 s**.
- Full suite: **303 passed in 98.15 s** (previous 207 plus the 96 new tests).

Three defects were found and fixed by those tests: session output directories
could collide because the Windows monotonic clock repeats inside a tight loop
(the token is now random and existence-checked), `_print_summary` raised
`TypeError` when no detector latency existed, and an early reconstruction
`finish` abandoned frames that were already queued instead of writing them
before closing the manifest. A fourth test-driven fix records the composed
view's frame size in the report even after the video writer closes. One test
expectation was corrected instead of the code: the blocked cards legitimately
carry different gate labels (`BLOCKED_MISSING_INPUT` for depth and terrain,
`DIAGNOSTIC_ONLY` for active mapping), so the assertion checks those real states
rather than one blanket prefix.

No physical camera was connected while building or testing the workbench: every
capture path in the tests and in the bounded verification runs is a mocked
`cv2.VideoCapture` stand-in, so `docs/claims_register.md` C30 (live-camera
operation tested) stays rejected and C36 is recorded as IMPLEMENTED,
UNVALIDATED (physical camera).

## Addendum 2026-10-25: pretrained weights installed and the vision views made real

The operator approved exactly three pretrained assets for local CPU use
(Depth Anything V2-Small, SegFormer-B0/ADE20K, the torchvision MobileNetV3-Small
ImageNet weights); every other non-YOLO architecture stays excluded.
`tools/download_prism_pretrained.py` installed them idempotently and recorded
`model_assets/pretrained/manifest.json`:

- `model_assets/pretrained/depth_anything_v2_small_hf/` —
  `depth-anything/Depth-Anything-V2-Small-hf` @ `5426e4f0f36572d16453bbda7a8389317b1bef99`,
  `model.safetensors` 99,173,660 B, SHA-256 `3152477ce0d8d697…49a16b70`,
  24,785,089 params, loaded on CPU in 8.26 s with `local_files_only=True`.
- `model_assets/pretrained/segformer_b0_ade_512_512/` —
  `nvidia/segformer-b0-finetuned-ade-512-512` @ `489d5cd81a0b59fab9b7ea758d3548ebe99677da`,
  `model.safetensors` 15,036,944 B, SHA-256 `6ae39addd01de6b1…51fee23f9`,
  3,752,694 params, loaded in 0.66 s. Its card declares `license: other` and
  links the (non-commercial) NVIDIA Source Code License; the weights are
  gitignored and not redistributed.
- `~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` — 10,306,551 B,
  SHA-256 `047dcff4addef86e…f4e1f`, loaded in 1.22 s. Initialisation only: the
  custom 17-channel terrain architecture was not replaced and **this milestone
  trained nothing**. `checkpoints/trav_unc_shared.pt` did not exist during this
  work and was not created by it; the separate manual training task has since
  produced one (2026-10-25, its own log: 30 epochs, val mIoU 0.721 against
  geometric pseudo-labels - not ground truth and not our result). The viewer
  never loads that checkpoint, and its on-screen learned-traversability line is
  generated from real file state instead of a hardcoded claim.

`tools/run_prism_pretrained_smoke.py` then ran both real models on two frames of
`clips/clip_01.mp4` on CPU and wrote
`output/pretrained_smoke_tests/20261025T145051Z/` (15 PNG previews +
`smoke_report.json`): **11/11 checks PASS**. Depth came out at 640×360 with
63.7 %/64.1 % valid, `rel_inv ∈ [0, 1.02]`, an assumed-scale ground fit of
a≈9.27/9.59 with 5,513/5,589 inliers and residual 0.037/0.053, and median
assumed-scale distance 0.80/0.74 m. Segmentation produced 7/6 distinct prism-7
classes with 44,338 `IGNORE_INDEX` abstentions (19 %), mean top-probability 0.52
and drivable fraction 0.34/0.39. The real `MappingStage` observed 5,664 BEV cells
and labelled 1,956 of them (92.0 % of the grid stays UNKNOWN), and the
single-frame cloud carried 36,039 finite points. Wall-clock CPU cost per pass:
depth 16.2 s then 4.8 s, segmentation 19.1 s then 3.3 s.

Re-run on 2026-10-25 after the learned-traversability wording became
state-derived: `output/pretrained_smoke_tests/20261025T162948Z/`, again
**11/11 PASS**, with `learned_traversability_absent` replaced by
`traversability_is_the_geometric_fallback` (`mapping.trav_source == "geometric"`)
plus an explicit `learned_traversability_checkpoint` record of the file's presence
and `used_by_this_run: false`. Asserting that the checkpoint is absent was the
wrong test once a separate task was entitled to create it; asserting that this
run uses only the geometric fallback is the claim that stays true either way.

The workbench itself was run bounded and headless through the real pipeline with
clip frames and **every socket in the process blocked** (`socket.connect`,
`connect_ex`, `create_connection`, `getaddrinfo`) to prove startup needs no
network: exit 0, `output/live_viewer_runs/20261025T150233Z-13865dbf`, depth and
segmentation `ACTIVE` from the local directories
(`model_source.local_files_only: true`), terrain `EXPERIMENTAL_GEOMETRIC`,
measured p50 latency depth 5.70 s / segmentation 4.06 s / detection 2.81 s /
motion 2.47 s / BEV fuse 0.068 s, 6 frames read, 2 processed, 4 stale frames
dropped by the bounded hand-off. Its report keeps
`depth_metric_reported: False`, `terrain_or_traversability_reported: False`,
`learned_traversability_trained: False` and `fabricated_visuals_drawn: 0`.

Tests and defects:

- Final visual verification (2026-10-25): the six real views were re-rendered from
  `clips/clip_01.mp4` frames through `compose_view` with every socket blocked, into
  `output/pretrained_viewer_views/20261025T173825Z/` (plus `smoke/` from
  `tools/run_prism_pretrained_smoke.py --frames 2`, 11/11 PASS), and each PNG was
  read back and inspected. Two wording defects were found only by looking at the
  images: the live view guessed "detector has not returned for any frame yet (model
  loading?)" while `detection` was actually `ACTIVE` and simply unsampled at the
  run's cadence (now derived from `statuses.detection.state`), and the segmentation
  panel truncated prism-7 class names mid-word (`obsta=`, `dynam=`), which reads as
  a rendering bug (now full names). The terrain panel and stamp report the
  checkpoint's real presence without implying it is loaded or validated.
- Focused reruns used a temp directory on the free drive (`TMPDIR`/`TEMP`/`TMP` →
  `F:\tmp_pytest`): an earlier identical run failed with
  `OSError: [Errno 28] No space left on device` because the system drive had ~4.4 GB
  free, not because of any code change.

- `tests/test_live_viewer_pretrained.py`: **22 passed in 6.5 s** (stages
  injected, so torch never loads; one test drives the real numpy-only
  `MappingStage`, and one asserts the local-only rule by making
  `snapshot_download` raise). Three of them now pin the two captions found by
  reading the rendered PNGs: the detector line must follow
  `statuses.detection.state`, and no prism-7 class name may be cut short.
- `tests/test_live_viewer.py` + `tests/test_live_viewer_pretrained.py`:
  **118 passed in 26.4 s**. The full suite was not rerun for this verification
  pass (322/322 had already passed before it).
- Fixed while integrating: `reset_perception_state` referenced a
  `segmentation_stage` attribute that does not exist; the segmentation overlay
  crashed on a `cv2` size mismatch between the 640×360 label map and the display
  frame; the overview grid still placed tiles in 3 columns after it was widened
  to 4, which broke a broadcast; the depth/segmentation/BEV cadence was keyed on
  the absolute frame sequence so a short run never fired a stage;
  `pretrained_paths._is_complete_model_dir` used `any(generator-of-generators)`,
  which is always truthy, so a half-written download was accepted as a complete
  model; `colorize_height` emitted an invalid-value warning on NaN BEV cells;
  and the smoke check "segmentation covers the frame" wrongly treated the
  project's own `IGNORE_INDEX` abstentions as missing pixels. The last flaky
  assertion (a report test that read the depth fit before the pipeline had
  processed a frame) now waits for the result like its neighbours.
- Still not claimed: metric depth, metric or learned traversability, real-time
  3-D, live-camera behaviour, or any accuracy number for these pretrained models
  (no reviewed labels exist).
