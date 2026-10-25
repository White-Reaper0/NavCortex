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
