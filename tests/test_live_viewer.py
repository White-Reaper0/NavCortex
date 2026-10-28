"""Model-free, window-free tests for the native live vision workbench.

Capture, the GUI and the detector are always mocked: nothing here opens a
camera, opens a window, touches the network or loads a neural network. The
tests exist to keep the workbench honest - a blocked capability must stay
reported as blocked and must never paint a synthetic image.
"""
from __future__ import annotations

from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_prism_live_viewer", ROOT / "tools" / "run_prism_live_viewer.py")
lv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lv
SPEC.loader.exec_module(lv)

from prism.belief_ledger import EvidenceRelation  # noqa: E402
from prism.integrations.tracking import TrackRecord  # noqa: E402
from prism.observations import (  # noqa: E402
    CoordinateFrame,
    EvidencePayload,
    ImageCoordinateConvention,
    ObjectDetection,
    ObjectDetectionEvidence,
    Observation,
    Validity,
)
from prism.video_observation_replay import DecodedFrame, VideoReplayInput  # noqa: E402

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
FRAME[:, :, 1] = 90


class FakeCapture:
    """Stand-in for cv2.VideoCapture; yields scripted read() outcomes."""

    def __init__(self, frames: int, *, fps: float = 200.0, opened: bool = True,
                 fail_reads: bool = False, forever: bool = False) -> None:
        self._frames = int(frames)
        self._forever = bool(forever)
        self._delay = 1.0 / max(1.0, float(fps))
        self._opened = opened
        self.fail_reads = fail_reads
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return self._opened

    def get(self, property_id: int) -> float:
        return 0.0

    def set(self, property_id: int, value: float) -> bool:
        return True

    def read(self):
        self.reads += 1
        if self.fail_reads:
            return False, None
        if self._frames <= 0 and not self._forever:
            return False, None
        if not self._forever:
            self._frames -= 1
        time.sleep(self._delay)
        return True, FRAME.copy()

    def release(self) -> None:
        self.released = True


class FakeDetector:
    """Returns real ObjectDetection evidence, or raises when told to fail."""

    def __init__(self, *, fail: bool = False, detections: int = 1) -> None:
        self.fail = fail
        self.detections = int(detections)
        self.calls = 0
        self.config = {"implementation": "FakeDetector", "weights_sha256": None}

    def __call__(self, frame: DecodedFrame) -> VideoReplayInput:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated provider crash")
        items = tuple(
            ObjectDetection(class_id=index, taxonomy="coco-2017@80",
                            score=min(0.99, 0.5 + 0.1 * index),
                            bbox_xyxy=(2.0 + 4.0 * index, 3.0, 6.0 + 4.0 * index, 9.0))
            for index in range(self.detections))
        payload = ObjectDetectionEvidence(
            "coco-2017@80", frame.width, frame.height,
            ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE, items)
        observation = Observation(
            observation_id=f"fake:frame-{frame.frame_index}:{frame.capture_at_ns}",
            source="fake", captured_at_ns=frame.capture_at_ns,
            coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.VALID,
            provenance={"detector": {"implementation": "FakeDetector"}},
            evidence=EvidencePayload(object_detections=payload))
        return VideoReplayInput(observation, lv.HYPOTHESIS_DETECTION,
                                EvidenceRelation.SUPPORTS, frame.capture_at_ns + 1_000_000_000)


class FakeTracker:
    backend = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def update(self, detections, *, frame_index: int, capture_at_ns: int):
        self.calls += 1
        return tuple(
            TrackRecord(track_id=index, frame_index=frame_index, capture_at_ns=capture_at_ns,
                        class_id=item.class_id, taxonomy=item.taxonomy,
                        bbox_xyxy=item.bbox_xyxy, score=item.score,
                        observed=True, predicted=False, age_frames=0)
            for index, item in enumerate(detections))


def make_config(**overrides):
    base = dict(source=lv.SOURCE_WEBCAM, headless=True, max_frames=3, open_retries=0,
                read_retries=0, retry_delay_s=0.0, read_retry_delay_s=0.0,
                recon_frames=8, recon_interval=1, buffer_frames=2, motion_interval=2,
                with_depth=False, with_segmentation=False)
    base.update(overrides)
    return lv.ViewerConfig(**base)


def as_dict(config) -> dict:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def make_session(tmp_path, *, config=None, capture=None, detector=None, opener=None,
                 with_detector=True, depth=None, segmentation=None, mapping=None):
    config = config or make_config()
    if opener is None:
        chosen = lambda cfg: capture  # noqa: E731
    else:
        chosen = opener
    factory = None
    if with_detector:
        chosen_detector = detector if detector is not None else FakeDetector()
        factory = lambda weights, confidence, source: chosen_detector  # noqa: E731

    def _injected(stage):
        return None if stage is None else (lambda cfg: stage)

    return lv.ViewerSession(config, output_dir=tmp_path / "run", capture_opener=chosen,
                            gui=lv.NullGui(), detector_factory=factory,
                            depth_factory=_injected(depth),
                            segmentation_factory=_injected(segmentation),
                            mapping_factory=_injected(mapping))


def started(tmp_path, *, config=None, capture=None, detector=None, depth=None,
            segmentation=None, mapping=None, timeout_s=6.0):
    """A session whose inference thread has built its modules."""
    session = make_session(tmp_path, config=config, capture=capture, detector=detector,
                           depth=depth, segmentation=segmentation, mapping=mapping)
    session.start()
    deadline = time.monotonic() + timeout_s
    while session.pipeline is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session.pipeline is not None, "modules were never prepared"
    return session


def finish(session) -> None:
    session.stop("test teardown")
    session.join(timeout_s=3.0)
    session.reconstruction.shutdown()


# --------------------------------------------------------------------- config
class TestConfiguration:
    def test_webcam_is_the_default_source(self):
        assert lv.resolve_source(None, None, {}) == lv.SOURCE_WEBCAM

    def test_explicit_url_implies_ipwebcam(self):
        assert lv.resolve_source(None, "http://10.0.0.9:8080", {}) == lv.SOURCE_IPWEBCAM

    def test_environment_url_implies_ipwebcam(self):
        env = {lv.URL_ENV_VAR: "http://10.0.0.9:8080"}
        assert lv.resolve_source(None, None, env) == lv.SOURCE_IPWEBCAM

    def test_cli_url_wins_over_environment(self):
        env = {lv.URL_ENV_VAR: "http://10.0.0.9:8080"}
        url, source = lv.RUNNER.resolve_url_with_source("http://10.78.70.188:8080/video", env)
        assert url == "http://10.78.70.188:8080/video"
        assert source == "cli"

    def test_environment_is_used_when_no_cli_url(self):
        env = {lv.URL_ENV_VAR: "http://192.168.0.150:8080"}
        url, source = lv.RUNNER.resolve_url_with_source(None, env)
        assert url == "http://192.168.0.150:8080/video"
        assert source == "environment"

    def test_convenience_default_is_the_documented_phone(self):
        url, source = lv.RUNNER.resolve_url_with_source(None, {})
        assert url == "http://10.78.70.188:8080/video"
        assert source == "default"

    def test_video_suffix_is_never_doubled(self):
        assert lv.RUNNER.normalize_stream_url("http://h:8080/video") == "http://h:8080/video"
        assert lv.RUNNER.normalize_stream_url("http://h:8080") == "http://h:8080/video"

    def test_source_label_reflects_the_selection(self):
        assert make_config().source_label == "webcam:0"
        phone = lv.ViewerConfig(source=lv.SOURCE_IPWEBCAM, url="http://10.0.0.5:8080/video")
        assert phone.source_label == "ipwebcam:http://10.0.0.5:8080/video"

    @pytest.mark.parametrize("overrides", [
        {"source": "dslr"}, {"camera_index": -1}, {"buffer_frames": 0},
        {"confidence": 1.7}, {"detector_interval": 0}, {"recon_frames": 2},
        {"evidence_ttl_ns": 0}, {"display_fps": float("nan")}, {"url": "  "}])
    def test_invalid_values_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            lv.ViewerConfig(**overrides)

    def test_paths_are_normalised_to_path_objects(self, tmp_path):
        config = make_config(output=str(tmp_path / "run"), weights=str(tmp_path / "w.pt"))
        assert isinstance(config.output, Path) and isinstance(config.weights, Path)

    def test_output_directories_are_unique_per_run(self, tmp_path):
        config = make_config(output_root=tmp_path / "runs")
        sessions = [lv.ViewerSession(config, capture_opener=lambda cfg: FakeCapture(0),
                                     gui=lv.NullGui()) for _ in range(2)]
        assert sessions[0].output != sessions[1].output
        for session in sessions:
            assert str(session.output).startswith(str(tmp_path / "runs"))
            session.reconstruction.shutdown()

    def test_main_returns_two_on_invalid_config(self, capsys):
        assert lv.main(["--confidence", "5", "--headless"]) == 2
        assert "invalid configuration" in capsys.readouterr().err

    def test_parser_defaults_match_the_documented_launch(self):
        args = lv.build_parser().parse_args(
            ["--source", "ipwebcam", "--url", "http://10.78.70.188:8080/video"])
        config = lv.config_from_args(args)
        assert config.source == lv.SOURCE_IPWEBCAM
        assert config.url == "http://10.78.70.188:8080/video"
        assert config.url_source == "cli"

    def test_weights_never_default_to_a_download(self):
        args = lv.build_parser().parse_args([])
        config = lv.config_from_args(args)
        assert str(config.weights).endswith("LibreYOLO9t.pt")
        assert Path(config.weights).is_absolute()


class TestLatestFrameBuffer:
    def test_newest_frame_wins_and_obsolete_ones_are_counted(self):
        buffer = lv.LatestFrameBuffer(2)
        for index in range(5):
            buffer.publish(lv.CapturedFrame(index, 1000 + index, FRAME))
        assert buffer.published == 5
        frame = buffer.consume(timeout_s=0.1)
        assert frame is not None and frame.sequence == 4
        assert buffer.stale_dropped >= 3
        assert buffer.consumed == 1

    def test_close_releases_a_waiting_consumer(self):
        buffer = lv.LatestFrameBuffer(1)
        result = {}

        def waiter():
            result["frame"] = buffer.consume(timeout_s=2.0)

        thread = threading.Thread(target=waiter)
        thread.start()
        buffer.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert result["frame"] is None

    def test_capacity_is_never_zero(self):
        assert lv.LatestFrameBuffer(0).capacity == 1


# ------------------------------------------------------------------ the blocks
class TestBlockedCapabilitiesStayBlocked:
    @pytest.mark.parametrize("block", [lv.DEPTH_BLOCK, lv.TERRAIN_BLOCK, lv.MAPPING_BLOCK])
    def test_no_blocked_capability_draws_anything(self, block):
        assert block["drawn"] is False
        assert block["missing"], "a blocked capability must name what is missing"
        assert block["state"] in ("BLOCKED_MISSING_INPUT", "BLOCKED_NO_IMPLEMENTATION",
                                  "DIAGNOSTIC_ONLY", "OFFLINE_ONLY")
        assert block["note"]

    def test_depth_and_terrain_views_blit_only_the_real_rgb_frame(self, tmp_path):
        session = make_session(tmp_path)
        frame = FRAME.copy()
        snap = {"frame": frame, "frame_index": 1, "frame_capture_ns": time.monotonic_ns(),
                "processed": 1, "detection": None, "motion": {}, "yolop": None, "trails": {},
                "advisory": {"disposition": None}, "adapter_gate": {"attempted": False},
                "statuses": session.snapshot()["statuses"], "errors": [], "recon": {}}
        blitted = []
        real_blit = lv._blit

        def spy(img, source, *args, **kwargs):
            blitted.append(source)
            return real_blit(img, source, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(lv, "_blit", spy)
        try:
            for view in (lv.VIEW_DEPTH, lv.VIEW_TERRAIN):
                blitted.clear()
                image = lv.compose_view(view, snap, session, lv.DISPLAY_SIZE)
                assert image.shape[:2] == (lv.DISPLAY_SIZE[1], lv.DISPLAY_SIZE[0])
                assert blitted, f"the {view} view shows no imagery at all"
                assert all(item is frame for item in blitted), (
                    f"the {view} view drew something that is not the live camera frame")
        finally:
            monkeypatch.undo()

    def test_blocked_capabilities_are_not_measurements_in_the_report(self, tmp_path):
        session = make_session(tmp_path)
        session.frames_read = 3
        report = session.finalize()
        assert report["depth"]["drawn"] is False
        assert report["depth"]["runs"] in (0, None)
        assert report["segmentation"]["drawn"] is False
        assert report["terrain"]["drawn"] is False
        assert report["terrain"]["live_gate"]["status"] == "BLOCKED_MISSING_INPUT"
        assert report["terrain"]["live_gate"]["grid"] is None
        assert report["mapping"]["drawn"] is False
        session.reconstruction.shutdown()

    def test_honesty_block_is_written_every_run(self, tmp_path):
        session = make_session(tmp_path)
        honesty = session.finalize()["honesty"]
        assert honesty["live_camera_connected"] is False
        assert honesty["frames_read_from_configured_source"] == 0
        assert honesty["depth_metric_reported"] is False
        assert honesty["terrain_or_traversability_reported"] is False
        assert honesty["active_map_updated"] is False
        assert honesty["metric_trajectory_reported"] is False
        assert honesty["imu_input_exists"] is False
        assert honesty["reconstruction_is_real_time"] is False
        assert honesty["fabricated_visuals_drawn"] == 0
        assert honesty["advisory_only_no_vehicle_command"] is True
        assert honesty["runtime_planner_supervisor_ros2_imported"] is False
        assert len(honesty["unverified_assumptions"]) >= 3
        assert honesty["limits"] == list(lv.LIMITS)
        session.reconstruction.shutdown()

    def test_vehicle_control_is_declared_as_not_imported(self, tmp_path):
        session = make_session(tmp_path)
        assert session.run() == 3
        assert session.report["module_statuses"]["vehicle_control"]["state"] == "NOT_IMPORTED"

    def test_the_tool_never_imports_the_command_path(self):
        source = (ROOT / "tools" / "run_prism_live_viewer.py").read_text(encoding="utf-8")
        for forbidden in ("prism.runtime", "prism.nav.planner", "nav/supervisor", "rclpy",
                          "mavlink", "serial"):
            for verb in ("import ", "from "):
                assert f"{verb}{forbidden} " not in source, forbidden


# -------------------------------------------------------------------- pipeline
class TestDetectionAndTracking:
    def build(self, detector, *, detector_interval=1, motion_interval=5):
        config = make_config(detector_interval=detector_interval,
                             motion_interval=motion_interval)
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing())
        return lv.LivePipeline(config, source_label="test:0", detector=detector,
                               tracker=FakeTracker(), motion=motion)

    def frame(self, sequence=1):
        return lv.CapturedFrame(sequence, 1_000_000_000 + sequence * 1_000_000, FRAME.copy())

    def test_detector_boxes_are_displayed_verbatim(self):
        pipeline = self.build(FakeDetector(detections=2))
        pipeline.process(self.frame())
        detection = pipeline.detection
        assert detection["status"] == "AVAILABLE"
        assert detection["detection_count"] == 2 and detection["track_count"] == 2
        assert tuple(item.bbox_xyxy for item in detection["detections"])[0] == (2.0, 3.0, 6.0, 9.0)
        assert detection["latency_ms"] >= 0.0
        assert pipeline.detections_log[0]["provenance"]["detector"]["implementation"] == \
            "FakeDetector"

    def test_provider_error_fabricates_no_detections(self):
        pipeline = self.build(FakeDetector(fail=True))
        pipeline.process(self.frame())
        detection = pipeline.detection
        assert detection["status"] == "PROVIDER_ERROR"
        assert detection["detections"] == () and detection["tracks"] == ()
        assert detection["detection_count"] == 0 and detection["track_count"] == 0
        assert "simulated provider crash" in detection["error"]
        assert pipeline.detector_errors == 1 and pipeline.detector_runs == 0
        assert any("simulated provider crash" in error for error in pipeline.errors)

    def test_an_empty_result_stays_empty(self):
        pipeline = self.build(FakeDetector(detections=0))
        pipeline.process(self.frame())
        assert pipeline.detection["status"] == "AVAILABLE"
        assert pipeline.detection["detection_count"] == 0
        assert pipeline.trails == {}

    def test_no_detector_means_no_boxes(self):
        pipeline = self.build(None)
        pipeline.process(self.frame())
        assert pipeline.detection is None
        assert pipeline.statuses["detection"]["state"] == "UNAVAILABLE"
        assert pipeline.statuses["detection"]["missing"]

    def test_evidence_goes_through_the_real_ledger_and_doubt_engine(self):
        from prism.belief_ledger import HypothesisStatus
        pipeline = self.build(FakeDetector())
        pipeline.process(self.frame())
        at_ns = pipeline.detection["capture_at_ns"]
        assert pipeline.ledger.hypothesis_status(lv.HYPOTHESIS_DETECTION, at_ns=at_ns) \
            is HypothesisStatus.SUPPORTED
        assert pipeline.advisory["disposition"] in (
            "no_action", "request_evidence", "cautious_stop")
        assert pipeline.advisory["note"].startswith("advisory only")
        for item in pipeline.advisory["recommendations"]:
            assert item["kind"] in ("request_one_observation", "wait_for_one_fresh_frame",
                                    "cautious_stop")

    def test_terrain_evidence_is_recorded_as_unavailable_not_as_free_space(self):
        from prism.belief_ledger import EvidenceState, HypothesisStatus
        pipeline = self.build(FakeDetector())
        pipeline.process(self.frame())
        records = pipeline.ledger.evidence_for(lv.HYPOTHESIS_TERRAIN)
        assert len(records) == 1
        observation = records[0].observation
        assert observation.validity is Validity.UNAVAILABLE
        assert "no metric spatial input" in str(observation.provenance)
        state = pipeline.ledger.evidence_state(records[0],
                                               at_ns=pipeline.detection["capture_at_ns"])
        assert state is EvidenceState.UNAVAILABLE
        assert pipeline.ledger.hypothesis_status(
            lv.HYPOTHESIS_TERRAIN, at_ns=pipeline.detection["capture_at_ns"]) \
            is HypothesisStatus.UNKNOWN
        assert pipeline.statuses["terrain"]["drawn"] is False

    def test_runtime_adapter_gate_reports_its_real_rejection(self):
        pipeline = self.build(FakeDetector())
        pipeline.process(self.frame())
        gate = pipeline.adapter_gate
        assert gate["attempted"] is True
        assert gate["accepted"] is False
        assert gate["reason"]

    def test_intervals_throttle_the_modules(self):
        pipeline = self.build(FakeDetector(), detector_interval=3)
        for sequence in range(1, 7):
            pipeline.process(self.frame(sequence))
        assert pipeline.detector_runs == 2

    def test_tracking_stays_in_image_space(self):
        pipeline = self.build(FakeDetector())
        pipeline.process(self.frame())
        assert pipeline.trails
        assert all(len(point) == 2 for trail in pipeline.trails.values() for point in trail)
        status = pipeline.statuses["tracking"]
        assert status["coordinate_frame"] == "image"
        assert "not world identities" in status["note"]

    def test_missing_checkpoint_is_reported_not_faked(self, tmp_path):
        config = make_config(weights=tmp_path / "absent.pt",
                             colmap_exe=tmp_path / "absent-colmap.exe")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run", gui=lv.NullGui(),
                                   capture_opener=lambda cfg: FakeCapture(0))
        session._prepare_modules()
        assert session.pipeline.detector is None
        assert session.pipeline.statuses["detection"]["state"] == "UNAVAILABLE"
        assert any("checkpoint not found" in error for error in session.errors)

    def test_detector_init_failure_is_isolated(self, tmp_path):
        def broken(weights, confidence, source):
            raise RuntimeError("SimDeviceError: backend unavailable")

        config = make_config(colmap_exe=tmp_path / "absent-colmap.exe")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run", gui=lv.NullGui(),
                                   capture_opener=lambda cfg: FakeCapture(0),
                                   detector_factory=broken)
        session._prepare_modules()
        assert session.pipeline.detector is None
        assert "SimDeviceError: backend unavailable" in \
            session.pipeline.statuses["detection"]["init_error"]
        assert session.pipeline.statuses["camera_motion"]["state"] == "ACTIVE_TRACKING_ONLY"

    def test_tracker_init_failure_only_removes_tracks(self, tmp_path, monkeypatch):
        def broken():
            raise ImportError("norfair is not installed")

        monkeypatch.setattr(lv, "create_image_tracker", broken)
        config = make_config(colmap_exe=tmp_path / "absent-colmap.exe")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run", gui=lv.NullGui(),
                                   capture_opener=lambda cfg: FakeCapture(0),
                                   detector_factory=lambda *a: FakeDetector())
        session._prepare_modules()
        assert isinstance(session.pipeline.tracker, lv._NullTracker)
        session.pipeline.process(lv.CapturedFrame(1, time.monotonic_ns(), FRAME.copy()))
        assert session.pipeline.detection["detection_count"] == 1
        assert session.pipeline.detection["track_count"] == 0
        assert session.pipeline.statuses["tracking"]["backend"] == "unavailable"

    def test_yolop_init_failure_is_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lv, "load_yolop",
                            lambda path, source: (_ for _ in ()).throw(
                                FileNotFoundError("no onnx present")))
        config = make_config(with_yolop=True, colmap_exe=tmp_path / "absent-colmap.exe")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run", gui=lv.NullGui(),
                                   capture_opener=lambda cfg: FakeCapture(0),
                                   detector_factory=lambda *a: FakeDetector())
        session._prepare_modules()
        assert session.pipeline.yolop is None
        assert session.pipeline.statuses["yolop"]["state"] == "NOT_REQUESTED"
        assert any("YOLOP initialisation failed" in error for error in session.errors)

    def test_yolop_error_is_counted_not_faked(self):
        class BrokenYolop:
            def run_detailed(self, frame):
                raise RuntimeError("onnx runtime missing")

        pipeline = self.build(FakeDetector())
        pipeline.yolop = BrokenYolop()
        pipeline.config = dataclass_replace(pipeline.config, yolop_interval=1)
        pipeline.process(self.frame())
        assert pipeline.yolop_result["status"] == "PROVIDER_ERROR"
        assert pipeline.yolop_errors == 1
        assert pipeline.yolop_result.get("segmentation") is None


def dataclass_replace(config, **overrides):
    values = as_dict(config)
    values.update(overrides)
    return lv.ViewerConfig(**values)


class TestMotionDiagnostics:
    def textured(self, seed: int, *, shift: int = 0):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, (lv.PROC_H, lv.PROC_W, 3), dtype=np.uint8)
        return np.roll(image, shift, axis=1)

    def test_without_calibration_motion_is_tracking_only(self):
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing())
        first = motion.update(self.textured(1), 0, 1_000_000)
        second = motion.update(self.textured(1, shift=12), 1, 2_000_000)
        assert first["status"] == lv.MotionStatus.FIRST_FRAME.value
        assert second["tracking_only"] is True
        assert second["scale_note"] == lv.ARBITRARY_SCALE_NOTE
        assert second["calibration"]["status"] == "missing"
        assert "fundamental-matrix" in second["geometry_note"]
        assert second["translation_unit"] in (None, "arbitrary")

    def test_assumed_intrinsics_are_flagged_unvalidated(self):
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.assumed_from_config())
        motion.update(self.textured(2), 0, 1_000_000)
        result = motion.update(self.textured(2, shift=10), 1, 2_000_000)
        assert result["tracking_only"] is False
        assert motion.calibration.usable_intrinsics and not motion.calibration.calibrated
        assert result["status"] != lv.MotionStatus.TRACKING_ONLY.value
        assert result["calibration"]["status"] == "assumed"
        assert result["scale_note"] is None

    def test_a_broken_frame_never_raises_out_of_the_module(self):
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing())
        result = motion.update(np.zeros((8, 8), np.uint8), 0, 1)
        assert result["status"] == lv.MotionStatus.FRAME_ERROR.value
        assert result["proc_bgr"] is None
        assert "not a 3-channel image" in result["reason"]

    def test_reset_forgets_the_previous_view(self):
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing())
        motion.update(self.textured(3), 0, 1)
        motion.reset()
        assert motion.update(self.textured(3), 1, 2)["status"] == \
            lv.MotionStatus.FIRST_FRAME.value

    def test_no_calibration_is_the_default(self):
        assert lv.build_calibration(make_config()).usable_intrinsics is False

    def test_assumed_intrinsics_need_an_explicit_flag(self):
        assert lv.build_calibration(make_config(allow_assumed_intrinsics=True)).usable_intrinsics

    def test_an_unreadable_calibration_artifact_is_rejected(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text("{ not json", encoding="utf-8")
        calibration = lv.build_calibration(make_config(calibration_path=path))
        assert calibration.usable_intrinsics is False
        assert "calibration artifact rejected" in calibration.provenance

    def test_view_reports_the_scale_note_to_the_operator(self, tmp_path):
        motion = lv.MotionDiagnostics(calibration=lv.CameraCalibration.missing())
        motion.update(self.textured(4), 0, 1)
        session = make_session(tmp_path)
        session.motion = motion
        snap = session.snapshot()
        assert snap["motion"]["tracking_only"] is True
        image = lv.compose_view(lv.VIEW_MOTION, snap, session, lv.DISPLAY_SIZE)
        assert image is not None
        session.reconstruction.shutdown()


# --------------------------------------------------------------- 3-D workflow
class TestReconstructionWorkflow:
    def manager(self, tmp_path, *, exe=None, target=4, stride=1, probe=None):
        return lv.ReconstructionManager(
            tmp_path / "recon", target_frames=target, stride=stride,
            colmap_exe=exe or (tmp_path / "colmap.exe"), threads=1, timeout_s=30,
            source_label="test:0", probe=probe)

    def test_missing_binary_is_stated_without_vendoring(self, tmp_path):
        manager = self.manager(tmp_path)
        assert manager.begin() is False
        assert manager.state == manager.FAILED
        assert "COLMAP executable not found" in manager.detail
        assert "does not vendor one" in manager.detail

    def test_sequence_is_written_with_the_probe_manifest_schema(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"not a real binary")
        manager = self.manager(tmp_path, exe=exe, target=4, stride=2)
        assert manager.begin() is True
        for sequence in range(6):
            manager.submit(FRAME, 1_000_000_000 + sequence * 10_000_000, sequence)
        manifest = manager.finish_capture()
        assert manager.state == manager.CAPTURED
        assert manifest["source_video"] == "live:test:0"
        assert manifest["source_sha256"] is None
        assert manifest["capture_mode"].startswith("live frames")
        assert "no hardware timestamp" in manifest["timestamp_note"]
        assert manifest["extracted_frames"] == 3 and manifest["requested_frames"] == 4
        assert manifest["resize_interpolation"].startswith("none")
        for row in manifest["frames"]:
            assert set(row) == {"file", "source_frame_index", "source_time_s", "capture_at_ns",
                                "sha256", "width", "height"}
            assert row["source_time_s"] is None
            assert len(row["sha256"]) == 64
            assert (manager.frames_dir / row["file"]).is_file()
        assert (manager.directory / "frames_manifest.json").is_file()
        manager.shutdown()

    def test_stride_selects_submissions_and_overflow_is_counted(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"x")
        manager = self.manager(tmp_path, exe=exe, target=2, stride=3)
        manager.begin()
        assert manager.submit(FRAME, 1, 0) == "skipped"
        assert manager.submit(FRAME, 2, 1) == "skipped"
        assert manager.submit(FRAME, 3, 2) == "queued"
        manager.shutdown()

    def test_submits_outside_a_capture_are_ignored(self, tmp_path):
        manager = self.manager(tmp_path)
        assert manager.submit(FRAME, 1, 0) == "idle"

    def test_short_sequence_is_refused_before_colmap_runs(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"x")
        manager = self.manager(tmp_path, exe=exe, target=4)
        manager.begin()
        manager.submit(FRAME, 1, 0)
        manager.submit(FRAME, 2, 1)
        manager.finish_capture()
        result = manager.run()
        assert "at least 3 frames" in result["status"]
        assert manager.state == manager.FAILED
        manager.shutdown()

    def test_run_uses_the_supported_probe_call(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"x")
        calls = {}

        class Probe:
            def run_colmap(self, **kwargs):
                calls.update(kwargs)
                raise RuntimeError("stop before the real binary")

        manager = self.manager(tmp_path, exe=exe, target=3, probe=Probe())
        manager.begin()
        for sequence in range(3):
            manager.submit(FRAME, 1_000_000_000 + sequence, sequence)
        manager.finish_capture()
        result = manager.run()
        assert calls["colmap_exe"] == exe
        assert calls["frames_dir"] == manager.frames_dir
        assert calls["overwrite"] is False
        assert calls["free_focal_length"] is False
        assert calls["camera_model"] == "SIMPLE_RADIAL"
        assert manager.state == manager.FAILED
        assert result["status"].startswith("COLMAP run failed")
        manager.shutdown()

    def test_a_run_without_a_sparse_model_is_a_failure_not_an_empty_scene(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"x")

        class Probe:
            def run_colmap(self, **kwargs):
                Path(kwargs["output"]).mkdir(parents=True, exist_ok=True)

        manager = self.manager(tmp_path, exe=exe, target=3, probe=Probe())
        manager.begin()
        for sequence in range(3):
            manager.submit(FRAME, 1_000_000_000 + sequence, sequence)
        manager.finish_capture()
        assert "no sparse model directory" in manager.run()["status"]
        manager.shutdown()

    def test_a_ready_model_reuses_its_cached_summary(self, tmp_path):
        manager = self.manager(tmp_path)
        manager.state = manager.READY
        manager.summary = {"status": "SPARSE_RECONSTRUCTION", "point_count": 7}
        assert manager.run() is manager.summary

    def test_status_reports_the_binary_and_captured_frames(self, tmp_path):
        manager = self.manager(tmp_path)
        status = manager.status()
        assert status["colmap_exe_present"] is False
        assert status["state"] == manager.IDLE
        assert status["captured_frames"] == 0
        assert status["target_frames"] == 4

    def test_no_point_cloud_exists_before_a_model_does(self, tmp_path):
        manager = self.manager(tmp_path)
        assert manager.view((320, 240)) is None

    def test_the_native_viewer_needs_an_exported_ply(self, tmp_path):
        manager = self.manager(tmp_path)
        result = manager.launch_open3d_viewer()
        assert result["status"] == "UNAVAILABLE"
        assert "no exported PLY" in result["reason"]

    def test_begin_starts_a_new_take_and_resets_state(self, tmp_path):
        exe = tmp_path / "colmap.exe"
        exe.write_bytes(b"x")
        manager = self.manager(tmp_path, exe=exe, target=2)
        manager.begin()
        manager.submit(FRAME, 1, 0)
        manager.submit(FRAME, 2, 1)
        manager.finish_capture()
        first = manager.directory
        assert manager.begin() is True
        assert manager.captured == 0 and manager.summary == {}
        assert manager.directory != first and manager.directory.name == "take_002"
        manager.shutdown()

    def test_session_submits_live_frames_to_the_offline_sequence(self, tmp_path):
        config = make_config(recon_frames=4, recon_interval=1,
                             colmap_exe=tmp_path / "absent-colmap.exe")
        session = make_session(tmp_path, config=config, capture=FakeCapture(10, fps=300.0))
        assert session.run() == 0
        recon = session.report["reconstruction"]
        assert recon["colmap_exe_present"] is False
        assert recon["state"] == lv.ReconstructionManager.FAILED
        assert "does not vendor one" in recon["detail"]

    def test_reconstruction_is_never_labelled_real_time(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(20, fps=400.0))
        assert session.run() == 0
        status = session.report["module_statuses"]["reconstruction"]
        assert status["state"] == "OFFLINE_WORKFLOW"
        assert "not a real-time model" in status["note"]


# -------------------------------------------------------------------- session
class TestSessionRuns:
    def test_headless_run_writes_the_whole_audit_trail(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(12, fps=300.0),
                               config=make_config(max_frames=3))
        assert session.run() == 0
        for name in ("run_config.json", "run_report.json", "session.log", "events.jsonl",
                     "detections.jsonl", "tracks.jsonl", "motion.jsonl", "evidence.jsonl",
                     "status.jsonl"):
            assert (session.output / name).is_file(), name
        report = json.loads((session.output / "run_report.json").read_text(encoding="utf-8"))
        assert report["schema"] == "prism-live-viewer/v1"
        assert report["capture"]["frames_read"] >= 3
        assert report["processing"]["frames_processed"] >= 3
        assert report["detection"]["status"] == "AVAILABLE"
        assert report["tracking"]["count"] == 1
        assert report["honesty"]["live_camera_connected"] is True
        assert report["honesty"]["frames_read_from_configured_source"] >= 3

    def test_reports_never_claim_a_camera_that_was_not_read(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(0, fail_reads=True))
        assert session.run() == 3
        report = session.report
        assert report["capture"]["frames_read"] == 0
        assert report["honesty"]["live_camera_connected"] is False
        assert report["honesty"]["frames_read_from_configured_source"] == 0

    def test_no_network_probe_happens_when_headless(self, tmp_path, monkeypatch):
        def failing(config, url=None):
            raise lv.CaptureError("simulated: nothing answered")

        def never(*args, **kwargs):
            raise AssertionError("headless runs must not probe the network")

        monkeypatch.setattr(lv.RUNNER, "_diagnose_failure", never)
        config = make_config(source=lv.SOURCE_IPWEBCAM, url="http://10.0.0.5:8080/video")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run",
                                   capture_opener=failing, gui=lv.NullGui())
        assert session.run() == 3
        assert session.connection_diagnostics is None
        assert "simulated: nothing answered" in session.waiting_message()

    def test_exit_four_when_a_window_was_expected(self, tmp_path):
        config = make_config(headless=False, max_frames=2)
        session = lv.ViewerSession(config, output_dir=tmp_path / "run",
                                   capture_opener=lambda cfg: FakeCapture(8, fps=300.0),
                                   gui=lv.NullGui())
        assert session.run() == 4
        assert session.report["gui"]["available"] is False

    def test_open_retries_are_bounded(self, tmp_path):
        attempts = []

        def opener(config):
            attempts.append(1)
            raise lv.CaptureError("no handle")

        config = make_config(open_retries=1, max_frames=1)
        session = lv.ViewerSession(config, output_dir=tmp_path / "run",
                                   capture_opener=opener, gui=lv.NullGui())
        assert session.run() == 3
        assert len(attempts) == 2
        assert any("giving up" in error for error in session.errors)

    def test_the_stream_is_reopened_after_it_dies(self, tmp_path):
        captures = []

        def opener(config, url=None):
            capture = FakeCapture(4, fps=400.0)
            captures.append(capture)
            return capture

        config = make_config(source=lv.SOURCE_IPWEBCAM, url="http://10.0.0.5:8080/video",
                             max_frames=6, open_retries=2)
        session = lv.ViewerSession(config, output_dir=tmp_path / "run", capture_opener=opener,
                                   gui=lv.NullGui(),
                                   detector_factory=lambda *a: FakeDetector())
        assert session.run() == 0
        assert len(captures) >= 2
        assert captures[0].released is True
        assert any("reopened" in error for error in session.errors)

    def test_obsolete_frames_are_dropped_not_queued(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(40, fps=2000.0),
                               config=make_config(max_frames=2, buffer_frames=1))
        assert session.run() == 0
        assert session.buffer.stale_dropped > 0
        assert session.report["capture"]["stale_frames_dropped"] == session.buffer.stale_dropped

    def test_timestamps_are_strictly_increasing(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(20, fps=4000.0),
                               config=make_config(max_frames=4))
        session.start()
        deadline = time.monotonic() + 10.0
        while session.frames_read < 12 and time.monotonic() < deadline:
            time.sleep(0.02)
        finish(session)
        stamps = [row["capture_at_ns"] for row in session.status_log]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_result_age_is_visible_so_slow_output_cannot_look_current(self, tmp_path):
        session = started(tmp_path, capture=FakeCapture(20, fps=300.0),
                          detector=FakeDetector())
        pipeline = session.pipeline
        detection_ns = pipeline.detection["capture_at_ns"] if pipeline.detection else None
        assert detection_ns is None or session.age_ms(detection_ns) >= 0.0
        assert session.frame_age_ms() >= 0.0
        assert session.detection_now_ms is None or session.detection_now_ms >= 0.0
        finish(session)

    def test_run_config_is_written_before_any_frame(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(0, fail_reads=True))
        session.start()
        session.stop("test")
        session.join(timeout_s=3.0)
        config = json.loads((session.output / "run_config.json").read_text(encoding="utf-8"))
        assert config["source"] == lv.SOURCE_WEBCAM
        assert config["headless"] is True

    def test_credentials_never_reach_the_run_config(self, tmp_path):
        config = make_config(source=lv.SOURCE_IPWEBCAM,
                             url="http://user:secret@10.0.0.5:8080/video")
        session = lv.ViewerSession(config, output_dir=tmp_path / "run",
                                   capture_opener=lambda cfg, url=None: FakeCapture(0),
                                   gui=lv.NullGui())
        data = json.dumps(session._run_config())
        assert "secret" not in data
        assert "user:***@" in data
        assert "secret" not in json.dumps(session.finalize())

    def test_pause_drops_frames_instead_of_building_a_queue(self, tmp_path):
        session = started(tmp_path, capture=FakeCapture(0, fps=500.0, forever=True),
                          config=make_config(max_frames=None))
        deadline = time.monotonic() + 5.0
        while session.pipeline.processed < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert session.pipeline.processed >= 2
        session.paused = True
        before = session.pipeline.processed
        time.sleep(0.4)
        assert session.pipeline.processed == before
        assert session.paused_dropped > 0
        retained = (session.buffer.published - session.buffer.consumed
                    - session.buffer.stale_dropped)
        assert 0 <= retained <= session.buffer.capacity
        session.paused = False
        deadline = time.monotonic() + 5.0
        while session.pipeline.processed == before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert session.pipeline.processed > before
        finish(session)

    def test_recording_writes_a_video_of_the_composed_view(self, tmp_path):
        session = make_session(tmp_path, capture=FakeCapture(30, fps=300.0),
                               config=make_config(max_frames=4, record_on_start=True))
        session.run()
        report = session.report["recording"]
        assert report["enabled"] is True
        video = Path(report["path"])
        if video.is_file() and video.stat().st_size > 0:
            assert report["size"] == [lv.DISPLAY_SIZE[0], lv.DISPLAY_SIZE[1]]
        else:  # pragma: no cover - only on a host without an mp4v encoder
            assert any("mp4v" in error for error in session.errors)

    def test_snapshot_pairs_an_image_with_its_diagnostics(self, tmp_path):
        session = started(tmp_path, capture=FakeCapture(20, fps=300.0))
        record = session.save_snapshot()
        assert record["view"] == lv.VIEW_LIVE
        assert Path(record["json"]).is_file()
        if record["png"]:
            assert Path(record["png"]).is_file()
        sidecar = json.loads(Path(record["json"]).read_text(encoding="utf-8"))
        assert sidecar["view"] == lv.VIEW_LIVE
        assert sidecar["diagnostics"]["depth"]["drawn"] is False
        finish(session)


class TestKeys:
    def session(self, tmp_path, **kwargs):
        return started(tmp_path, capture=kwargs.get("capture", FakeCapture(30, fps=300.0)),
                       config=kwargs.get("config"), depth=kwargs.get("depth"),
                       segmentation=kwargs.get("segmentation"),
                       mapping=kwargs.get("mapping"))

    def test_every_number_key_selects_a_view(self, tmp_path):
        session = self.session(tmp_path)
        expected = {ord("1"): lv.VIEW_LIVE, ord("2"): lv.VIEW_TRACKING, ord("3"): lv.VIEW_DEPTH,
                    ord("4"): lv.VIEW_SEGMENTATION, ord("5"): lv.VIEW_TERRAIN,
                    ord("6"): lv.VIEW_MOTION, ord("7"): lv.VIEW_RECON,
                    ord("8"): lv.VIEW_CLOUD, ord("9"): lv.VIEW_EVIDENCE,
                    ord("0"): lv.VIEW_OVERVIEW}
        for key, view in expected.items():
            assert session.handle_key(key) is True
            assert session.view == view
        assert set(lv.VIEW_TITLES) == set(expected.values())
        finish(session)

    @pytest.mark.parametrize("key", [ord("q"), 27, ord("Q")])
    def test_stop_keys_terminate_the_session(self, tmp_path, key):
        session = self.session(tmp_path)
        assert session.handle_key(key) is False
        assert session._stop.is_set()
        finish(session)

    def test_an_unbound_key_does_nothing(self, tmp_path):
        session = self.session(tmp_path)
        before = (session.view, session.recording, session.paused)
        assert session.handle_key(ord("w")) is True
        assert (session.view, session.recording, session.paused) == before
        finish(session)

    def test_r_toggles_recording(self, tmp_path):
        session = self.session(tmp_path)
        session.handle_key(ord("r"))
        assert session.recording is True
        session.handle_key(ord("R"))
        assert session.recording is False
        finish(session)

    def test_s_writes_a_snapshot(self, tmp_path):
        session = self.session(tmp_path)
        session.handle_key(ord("s"))
        assert session.snapshot_files
        finish(session)

    def test_h_toggles_the_limits_overlay(self, tmp_path):
        session = self.session(tmp_path)
        session.handle_key(ord("h"))
        assert session.help is True
        image = lv.compose_view(session.view, session.snapshot(), session, lv.DISPLAY_SIZE)
        assert image is not None
        session.handle_key(ord("H"))
        assert session.help is False
        finish(session)

    def test_m_resets_motion_and_trails(self, tmp_path):
        session = self.session(tmp_path)
        session.handle_key(ord("m"))
        assert session.motion._prev_xy is None
        assert "reset by operator" in session.errors[-1]
        finish(session)

    def test_c_starts_a_new_capture_take(self, tmp_path):
        config = make_config(colmap_exe=Path("must-not-exist.exe"))
        session = self.session(tmp_path, config=config)
        session.handle_key(ord("c"))
        assert session.reconstruction.state == lv.ReconstructionManager.FAILED
        assert "COLMAP executable not found" in session.reconstruction.detail
        finish(session)

    def test_u_reresolves_the_url_so_a_new_phone_ip_needs_a_relaunch_free_fix(self, tmp_path):
        url_file = tmp_path / "phone.txt"
        url_file.write_text("http://10.78.70.9:8080", encoding="utf-8")
        config = make_config(source=lv.SOURCE_IPWEBCAM,
                             url="http://10.78.70.188:8080/video", url_file=url_file,
                             max_frames=8)
        captures = []

        def opener(cfg, url=None):
            capture = FakeCapture(6, fps=400.0)
            captures.append((cfg, url))
            return capture

        session = lv.ViewerSession(config, output_dir=tmp_path / "run", capture_opener=opener,
                                   gui=lv.NullGui(),
                                   detector_factory=lambda *a: FakeDetector())
        session.start()
        deadline = time.monotonic() + 6.0
        while session.pipeline is None and time.monotonic() < deadline:
            time.sleep(0.02)
        session.handle_key(ord("u"))
        assert session.active_url == "http://10.78.70.9:8080/video"
        assert session.reopen_count >= 1
        finish(session)

    def test_p_pauses_inference(self, tmp_path):
        session = self.session(tmp_path)
        session.handle_key(ord("p"))
        assert session.paused is True
        assert "dropped" in session.state_label()
        session.handle_key(ord("P"))
        assert session.paused is False
        finish(session)

    def test_k_starts_the_offline_pass_without_faking_a_model(self, tmp_path):
        config = make_config(colmap_exe=Path("must-not-exist.exe"))
        session = self.session(tmp_path, config=config)
        session.handle_key(ord("k"))
        deadline = time.monotonic() + 3.0
        while session._recon_thread is None and time.monotonic() < deadline:
            time.sleep(0.02)
        session._recon_thread.join(timeout=3.0)
        assert session.reconstruction.state == lv.ReconstructionManager.FAILED
        assert session.report == {}
        finish(session)
