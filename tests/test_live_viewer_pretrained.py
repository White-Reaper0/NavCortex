"""Focused tests for the pretrained vision path inside the native live viewer.

The two downloaded models are never loaded here: the stages are replaced by the
small fakes below, so these tests exercise the viewer's integration (cadence,
labels, error isolation, reporting) without torch or a network. One test drives
the real ``MappingStage`` because it is numpy-only and it is what proves the
geometric terrain fallback is the project's own, not a viewer-side invention.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from test_live_viewer import (FRAME, FakeCapture, FakeDetector, FakeTracker, as_dict,
                              finish, make_config, make_session, started)
from prism.types import BEVMap, DepthResult, SegResult

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "run_prism_live_viewer" not in sys.modules:
    SPEC = importlib.util.spec_from_file_location(
        "run_prism_live_viewer", ROOT / "tools" / "run_prism_live_viewer.py")
    lv = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = lv
    SPEC.loader.exec_module(lv)
else:
    lv = sys.modules["run_prism_live_viewer"]

STAGE_SIZE = (36, 48)          # (h, w) for the fake stages - small but not degenerate


class FakeDepthStage:
    """Stands in for DepthStage: publishes a DepthResult for the frame it is given."""

    model_source = "fake://depth_anything_v2_small_hf"

    def __init__(self, *, fail: bool = False, size=STAGE_SIZE, depth_m: float = 1.5) -> None:
        self.fail = fail
        self.size = size
        self.depth_m = float(depth_m)
        self.calls = 0
        self.resets = 0

    def __call__(self, packet):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated depth crash")
        h, w = self.size
        packet.depth = DepthResult(
            rel_inv=np.full((h, w), 0.66, np.float32),
            depth_m=np.full((h, w), self.depth_m, np.float32),
            scale=0.9, shift=0.05, align_residual=0.012, align_inliers=42,
            valid=np.ones((h, w), bool))
        packet.timings_ms["depth"] = 12.5
        return packet

    def reset(self) -> None:
        self.resets += 1


class FakeSegStage:
    """Stands in for SegStage running the SegFormer-B0/ADE20K teacher."""

    model_name = "SegFormer-B0 (ADE20K) -> prism-7"
    backend = "teacher"
    model_source = "fake://segformer_b0_ade_512_512"

    def __init__(self, *, fail: bool = False, size=STAGE_SIZE) -> None:
        self.fail = fail
        self.size = size
        self.calls = 0
        self.resets = 0

    def result(self, bgr):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated segmentation crash")
        h, w = self.size
        return SegResult(label=np.full((h, w), 1, np.uint8),
                         prob_max=np.full((h, w), 0.8, np.float32),
                         entropy=np.full((h, w), 0.2, np.float32))

    def class_shares(self, label):
        shares = np.zeros(7, np.float32)
        shares[1] = 0.9
        return shares

    def drivable_fraction(self, label):
        return 0.9

    def reset(self) -> None:
        self.resets += 1


class FakeMappingStage:
    """Stands in for MappingStage when only the plumbing is under test."""

    def __init__(self, *, bev: BEVMap | None = None) -> None:
        self.bev = BEVMap.empty() if bev is None else bev
        self.trav_source = "geometric"
        self.last_stats: dict = {}
        self.calls = 0
        self.resets = 0

    def __call__(self, packet):
        self.calls += 1
        packet.bev = self.bev
        packet.timings_ms["bev"] = 4.0
        return packet

    def reset(self) -> None:
        self.resets += 1


def integrated_session(tmp_path, *, depth=None, segmentation=None, mapping=None,
                       capture=None, config=None):
    depth = depth if depth is not None else FakeDepthStage()
    segmentation = segmentation if segmentation is not None else FakeSegStage()
    mapping = mapping if mapping is not None else FakeMappingStage()
    base = as_dict(config or make_config())
    base.update(with_depth=True, with_segmentation=True)
    cfg = lv.ViewerConfig(**base)
    return started(tmp_path, capture=capture or FakeCapture(40, fps=300.0), config=cfg,
                   depth=depth, segmentation=segmentation, mapping=mapping)


class TestDepthAndSegmentationViews:
    def test_depth_view_blits_the_model_output_not_only_the_camera_feed(self, tmp_path):
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.depth_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            result = session.pipeline.depth_result
            assert result is not None and result["status"] == "AVAILABLE"
            assert np.asarray(result["rel_inv"]).shape == STAGE_SIZE
            assert result["valid_fraction"] > 0.0
            blitted = []
            real_blit = lv._blit

            def spy(img, source, *args, **kwargs):
                blitted.append(np.asarray(source))
                return real_blit(img, source, *args, **kwargs)

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(lv, "_blit", spy)
            try:
                snap = session.snapshot()
                lv.compose_view(lv.VIEW_DEPTH, snap, session, lv.DISPLAY_SIZE)
            finally:
                monkeypatch.undo()
            frame = np.asarray(snap["frame"])
            assert any(item.shape != frame.shape for item in blitted), (
                "the depth view shows no depth imagery")
        finally:
            finish(session)

    def test_depth_valid_outline_is_placed_on_the_drawn_frame(self, tmp_path):
        """The mask is in frame pixels: it must be moved onto the letterboxed copy."""
        session = integrated_session(tmp_path, depth=FakeDepthStage(size=FRAME.shape[:2]))
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.depth_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            snap = session.snapshot()
            blits = []
            real_blit = lv._blit

            def spy(img, source, *args, **kwargs):
                placed = real_blit(img, source, *args, **kwargs)
                blits.append((source, placed))
                return placed

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(lv, "_blit", spy)
            try:
                image = lv.compose_view(lv.VIEW_DEPTH, snap, session, lv.DISPLAY_SIZE)
            finally:
                monkeypatch.undo()
            # identify the camera tile by the object actually handed to _blit: with a
            # frame-sized fake depth output, shape alone cannot tell the two tiles apart
            frame_blit = next(placed for source, placed in blits if source is snap["frame"])
            outline = np.all(np.asarray(image) == np.array((90, 200, 90), np.uint8), axis=-1)
            ys, xs = np.nonzero(outline)
            assert xs.size, "the depth view drew no valid-region outline"
            assert xs.min() >= frame_blit.x and xs.max() < frame_blit.x + frame_blit.w
            assert ys.min() >= frame_blit.y and ys.max() < frame_blit.y + frame_blit.h
        finally:
            finish(session)

    def test_depth_result_reports_the_assumed_scale_fit(self, tmp_path):
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.depth_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            report = session._report_section(session.snapshot())
            fit = report["depth"]["last"]["fit"]
            assert fit["scale_a"] == 0.9 and fit["inliers"] == 42
            assert "assumed" in report["depth"]["semantics"].lower()
            assert report["depth"]["model_source"] == FakeDepthStage.model_source
            assert session.snapshot()["statuses"]["depth"]["state"] == "ACTIVE"
        finally:
            finish(session)

    def test_segmentation_reports_pretrained_semantics_not_traversability(self, tmp_path):
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.seg_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            result = session.pipeline.seg_result
            assert result["status"] == "AVAILABLE"
            assert np.asarray(result["label"]).dtype == np.uint8
            assert result["class_shares"][1] == pytest.approx(0.9)
            assert "ADE20K" in result["pretraining"]
            report = session._report_section(session.snapshot())
            assert "not ground-truth traversability" in report["segmentation"]["semantics"]
            lv.compose_view(lv.VIEW_SEGMENTATION, session.snapshot(), session,
                            lv.DISPLAY_SIZE)
        finally:
            finish(session)

    def test_inference_runs_on_a_sampled_cadence(self, tmp_path):
        depth, segmentation = FakeDepthStage(), FakeSegStage()
        session = integrated_session(
            tmp_path, depth=depth, segmentation=segmentation,
            config=make_config(depth_interval=5, seg_interval=5, map_interval=5,
                               max_frames=12))
        try:
            deadline = time.monotonic() + 10.0
            while session.pipeline.processed < 12 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert depth.calls == segmentation.calls
            assert 1 <= depth.calls <= 3, "a heavyweight model must not run every frame"
        finally:
            finish(session)

    def test_stage_failure_is_isolated_from_the_other_views(self, tmp_path):
        depth, segmentation = FakeDepthStage(fail=True), FakeSegStage()
        session = integrated_session(tmp_path, depth=depth, segmentation=segmentation)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.depth_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert session.pipeline.depth_result["status"] == "STAGE_ERROR"
            assert session.pipeline.depth_errors >= 1
            assert session.pipeline.seg_result["status"] == "AVAILABLE"
            assert session.pipeline.detection["status"] == "AVAILABLE"
            snap = session.snapshot()
            for view in (lv.VIEW_DEPTH, lv.VIEW_TERRAIN, lv.VIEW_CLOUD, lv.VIEW_OVERVIEW):
                lv.compose_view(view, snap, session, lv.DISPLAY_SIZE)
        finally:
            finish(session)

    def test_operator_reset_clears_temporal_state(self, tmp_path):
        depth, segmentation, mapping = FakeDepthStage(), FakeSegStage(), FakeMappingStage()
        session = integrated_session(tmp_path, depth=depth, segmentation=segmentation,
                                     mapping=mapping)
        try:
            session.paused = True
            time.sleep(0.2)               # let any in-flight inference finish
            session.handle_key(ord("m"))
            assert depth.resets >= 1 and segmentation.resets >= 1 and mapping.resets >= 1
            assert session.pipeline._last_depth_obj is None
        finally:
            finish(session)


class TestGeometricTerrainGrid:
    def test_mapping_waits_for_depth_before_publishing_a_grid(self):
        mapping = FakeMappingStage()
        pipeline = lv.LivePipeline(
            make_config(depth_interval=1, map_interval=1), source_label="test:0",
            detector=FakeDetector(), tracker=FakeTracker(),
            motion=lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing()),
            mapping_stage=mapping)
        pipeline.process(lv.CapturedFrame(1, 2_000_000_000, FRAME.copy()))
        assert pipeline.bev_result["status"] == "WAITING_DEPTH"
        assert mapping.calls == 0
        pipeline.depth_stage = FakeDepthStage()
        pipeline.process(lv.CapturedFrame(2, 3_000_000_000, FRAME.copy()))
        assert pipeline.bev_result["status"] == "AVAILABLE"
        assert pipeline.bev_result["label"] == "GEOMETRIC / ASSUMED SCALE / UNVALIDATED"
        assert mapping.calls >= 1

    def test_real_mapping_stage_produces_a_geometric_grid_from_depth(self):
        """The project's own geometric fallback, driven with a model-shaped depth result."""
        from prism.perception.mapping import MappingStage
        stage = MappingStage(device="cpu")
        size = (120, 160)
        depth = FakeDepthStage(size=size, depth_m=1.2)
        packet = lv.FramePacket(clip_id="test", idx=1, t=0.03,
                                rgb=np.zeros((*size, 3), np.uint8))
        depth(packet)
        packet.seg = FakeSegStage(size=size).result(packet.rgb)
        stage(packet)
        assert stage.trav_source == "geometric"
        bev = packet.bev
        assert bev.height.shape == (lv.CFG.bev.H, lv.CFG.bev.W)
        assert np.isfinite(bev.height).any(), "no cell was observed from the depth output"
        stats = lv.cell_stats(bev)
        assert 0.0 < stats["frac_safe"] + stats["frac_risky"] + stats["frac_obstacle"] < 1.0
        assert stats["frac_unknown"] > 0.0, "unobserved cells must stay UNKNOWN"

    def test_learned_traversability_stays_untrained_and_separate(self, tmp_path):
        """The viewer must never present a learned traversability model as running here.

        The checkpoint belongs to the separate manual training task, so its presence in the
        repository is that task's state, not this viewer's: the wording has to stay true either
        way, which is why both states are exercised.
        """
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while (session.pipeline.bev_result or {}).get("status") != "AVAILABLE" \
                    and time.monotonic() < deadline:
                time.sleep(0.02)
            report = session._report_section(session.snapshot())
            assert "trav_unc_shared.pt" in report["terrain"]["learned_traversability"]
            assert report["terrain"]["trav_source"] == "geometric"
            assert report["terrain"]["label"] == "GEOMETRIC / ASSUMED SCALE / UNVALIDATED"
            assert report["terrain"]["live_gate"]["status"] == "BLOCKED_MISSING_INPUT"
            assert report["terrain"]["live_gate"]["grid"] is None
            snap = session.snapshot()
            status_note = snap["statuses"]["terrain"]["learned_traversability"]
            assert status_note == lv.learned_traversability_note()
            assert ("NOT TRAINED" in status_note or "NOT USED" in status_note), status_note
            # checkpoint absent: the plain "nothing is trained" wording
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(lv, "TRAV_CHECKPOINT", tmp_path / "absent.pt")
            try:
                assert lv.learned_traversability_note().startswith("NOT TRAINED")
                absent = session._report_section(session.snapshot())
                assert absent["terrain"]["learned_traversability"].startswith("NOT TRAINED")
            finally:
                monkeypatch.undo()
            # checkpoint present: still no learned model running in this viewer
            trained = tmp_path / "trav_unc_shared.pt"
            trained.write_bytes(b"not loaded by the viewer")
            monkeypatch.setattr(lv, "TRAV_CHECKPOINT", trained)
            try:
                note = lv.learned_traversability_note()
                assert "NOT USED BY THIS VIEWER" in note and trained.name in note
                assert "does not exist" not in note
                check = session._report_section(session.snapshot())
                assert check["terrain"]["learned_traversability"] == note
            finally:
                monkeypatch.undo()
        finally:
            finish(session)

    def test_status_rail_and_help_report_the_new_modules(self, tmp_path):
        session = integrated_session(tmp_path)
        try:
            snap = session.snapshot()
            assert snap["statuses"]["segmentation"]["state"] == "ACTIVE"
            assert snap["statuses"]["terrain"]["learned_traversability"] == \
                lv.learned_traversability_note()
            assert snap["statuses"]["terrain"]["metric_grid_gate"].startswith(
                "BLOCKED_MISSING_INPUT")
            lv.compose_view(lv.VIEW_OVERVIEW, snap, session, lv.DISPLAY_SIZE)
            session.help = True
            lv.compose_view(lv.VIEW_LIVE, snap, session, lv.DISPLAY_SIZE)
        finally:
            finish(session)


class TestPanelWordingFollowsState:
    """A caption that guesses is a defect: it must come from what the run actually knows."""

    @staticmethod
    def _drawn_lines(session, view, monkeypatch, mutate=None) -> str:
        captured: list[str] = []
        real_text = lv._text

        def spy(img, org, line, *args, **kwargs):
            captured.append(str(line))
            return real_text(img, org, line, *args, **kwargs)

        monkeypatch.setattr(lv, "_text", spy)
        snap = session.snapshot()
        if mutate is not None:
            mutate(snap)
        try:
            lv.compose_view(view, snap, session, lv.DISPLAY_SIZE)
        finally:
            monkeypatch.undo()
        return "\n".join(captured)

    def test_live_view_says_the_detector_is_loaded_when_it_is(self, tmp_path, monkeypatch):
        session = integrated_session(tmp_path)
        try:
            assert session.snapshot()["statuses"]["detection"]["state"] == "ACTIVE"
            text = self._drawn_lines(session, lv.VIEW_LIVE, monkeypatch,
                                     lambda snap: snap.pop("detection", None))
            assert "model loading?" not in text
            assert "detector loaded, no frame sampled yet" in text
            assert str(session.config.detector_interval) in text
        finally:
            finish(session)

    def test_live_view_says_the_detector_is_unavailable_when_it_is(self, tmp_path, monkeypatch):
        session = integrated_session(tmp_path)
        try:
            def mutate(snap):
                snap["detection"] = None
                snap["statuses"] = dict(snap["statuses"], detection={
                    "state": "UNAVAILABLE", "drawn": False})

            text = self._drawn_lines(session, lv.VIEW_LIVE, monkeypatch, mutate)
            assert "detector unavailable: no boxes can be drawn" in text
        finally:
            finish(session)

    def test_segmentation_panel_prints_every_prism_class_name_in_full(self, tmp_path,
                                                                      monkeypatch):
        from prism.config import TERRAIN_CLASSES
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.seg_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            text = self._drawn_lines(session, lv.VIEW_SEGMENTATION, monkeypatch)
            assert "class shares:" in text
            truncated = [name for name in TERRAIN_CLASSES if f"{name}=" not in text]
            assert not truncated, f"class names cut short: {truncated}"
        finally:
            finish(session)


class TestCloudPreview:
    def test_cloud_is_unprojected_from_the_actual_depth(self, tmp_path):
        session = integrated_session(tmp_path)
        try:
            deadline = time.monotonic() + 6.0
            while session.pipeline.depth_result is None and time.monotonic() < deadline:
                time.sleep(0.02)
            result = session.pipeline.depth_result
            cloud = lv._live_depth_cloud(result, np.asarray(session.snapshot()["frame"]), 1)
            assert cloud is not None
            points, colours = cloud
            assert points.shape[0] == colours.shape[0] > 0
            assert np.isfinite(points).all()
            assert np.abs(points[:, 1]).max() <= 12.0
            lv.compose_view(lv.VIEW_CLOUD, session.snapshot(), session, lv.DISPLAY_SIZE)
        finally:
            finish(session)

    def test_cloud_is_empty_honest_when_depth_is_out_of_range(self):
        stage = FakeDepthStage(depth_m=90.0)      # beyond the 12 m window
        packet = lv.FramePacket(clip_id="test", idx=1, t=0.0, rgb=FRAME.copy())
        stage(packet)
        result = {"depth_m": packet.depth.depth_m, "valid": packet.depth.valid}
        assert lv._live_depth_cloud(result, FRAME, 1) is None


class TestRunArtifacts:
    def test_finalize_writes_a_run_report_and_a_numpy_artifact(self, tmp_path):
        session = integrated_session(tmp_path, config=make_config(max_frames=4))
        assert session.run() == 0
        report = json.loads((session.output / "run_report.json").read_text(encoding="utf-8"))
        assert report["depth"]["drawn"] is True
        assert report["depth"]["runs"] >= 1
        assert report["latency_ms"]["depth"]["p50_ms"] is not None
        assert report["honesty"]["depth_relative_model_operational"] is True
        assert report["honesty"]["depth_metric_reported"] is False
        assert report["honesty"]["segmentation_is_pretrained_ade20k_not_traversability"] \
            is True
        assert report["honesty"]["learned_traversability_trained"] is False
        arrays = np.load(session.output / "last_perception.npz")
        assert {"rel_inv", "depth_m", "depth_valid", "seg_label", "bev_height"} \
            <= set(arrays.files)


class TestLocalOnlyLoading:
    def test_missing_weights_report_the_installer_and_download_nothing(self, tmp_path,
                                                                       monkeypatch):
        import huggingface_hub
        import prism.models.pretrained_paths as paths

        def never(*args, **kwargs):
            raise AssertionError("the viewer must never download weights")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", never)
        # simulate a checkout where the approved weights were never installed
        monkeypatch.setattr(paths, "local_model_dir", lambda *a, **k: None)
        config = make_config(with_depth=True, with_segmentation=True,
                             depth_model=tmp_path / "no-depth",
                             segmentation_model=tmp_path / "no-seg")
        session = started(tmp_path, capture=FakeCapture(20, fps=300.0), config=config,
                          timeout_s=15.0)
        try:
            statuses = session.snapshot()["statuses"]
            assert statuses["depth"]["state"] == "BLOCKED_MISSING_INPUT"
            assert statuses["segmentation"]["state"] == "UNAVAILABLE"
            assert any("download_prism_pretrained.py" in item
                       for item in statuses["depth"]["missing"])
            assert session.pipeline.depth_stage is None
            assert session.pipeline.seg_stage is None
            assert session.pipeline.mapping_stage is None
        finally:
            finish(session)

    def test_environment_override_selects_the_local_directory(self, tmp_path):
        from prism.models.pretrained_paths import (DEPTH_ENV_VAR, DEPTH_MODEL_ID,
                                                   local_model_dir)
        model_dir = tmp_path / "depth-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        resolved = local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR,
                                   environ={DEPTH_ENV_VAR: str(model_dir)})
        assert resolved == model_dir

    def test_incomplete_directory_is_not_treated_as_a_model(self, tmp_path, monkeypatch):
        from prism.models.pretrained_paths import (DEPTH_ENV_VAR, DEPTH_MODEL_ID,
                                                   local_model_dir)
        # ignore the repository default dir: this test is about directory validation
        monkeypatch.setattr("prism.models.pretrained_paths._DEFAULT_DIRS", {})
        empty = tmp_path / "half-a-model"
        empty.mkdir()
        (empty / "config.json").write_text("{}", encoding="utf-8")
        assert local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR,
                               environ={DEPTH_ENV_VAR: str(empty)}) is None

    def test_complete_directory_is_accepted_without_network(self, tmp_path, monkeypatch):
        from prism.models.pretrained_paths import (DEPTH_ENV_VAR, DEPTH_MODEL_ID,
                                                   local_model_dir)
        monkeypatch.setattr("prism.models.pretrained_paths._DEFAULT_DIRS", {})
        model_dir = tmp_path / "depth-local"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        assert local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR,
                               environ={DEPTH_ENV_VAR: str(model_dir)}) == model_dir
        assert local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR, environ={}) is None

    def test_disabled_by_flag_is_reported_as_disabled_not_blocked(self, tmp_path):
        session = make_session(tmp_path, config=make_config())
        statuses = session.snapshot()["statuses"]
        assert statuses["depth"]["state"] == "DISABLED"
        assert statuses["segmentation"]["state"] == "DISABLED"
        session.reconstruction.shutdown()
