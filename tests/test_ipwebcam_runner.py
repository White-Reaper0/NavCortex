"""Model-free tests for the live IP webcam runner; capture is always mocked."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_prism_ipwebcam", ROOT / "tools" / "run_prism_ipwebcam.py")
ipcam = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ipcam
SPEC.loader.exec_module(ipcam)

FRAME = np.zeros((24, 32, 3), dtype=np.uint8)


class FakeCapture:
    def __init__(self, reads, *, fps: float = 0.0, opened: bool = True) -> None:
        self._reads = list(reads)
        self._fps = float(fps)
        self._opened = opened
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened

    def get(self, property_id: int) -> float:
        return self._fps if property_id == ipcam.cv2.CAP_PROP_FPS else 0.0

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        return True

    def read(self):
        if not self._reads:
            return False, None
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, *, opened: bool = True) -> None:
        self._opened = opened
        self.frames: list = []
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class FakeHarnessReport:
    opened = True
    decoded_frames = 3
    records: tuple = ()

    def summary(self) -> dict:
        return {"opened": True, "decoded_frames": 3, "tracking_only": True}


class FakeHarness:
    def replay(self, path, **kwargs) -> FakeHarnessReport:
        return FakeHarnessReport()


class FakeTracker:
    backend = "fake-tracker"

    def update(self, detections, *, frame_index: int, capture_at_ns: int) -> tuple:
        return ()


@pytest.fixture
def runtime(monkeypatch):
    writers: dict[str, FakeWriter] = {}

    def writer_factory(path, fps, size):
        writer = FakeWriter()
        writers[str(path)] = writer
        return writer

    monkeypatch.setattr(ipcam, "_open_writer", writer_factory)
    monkeypatch.setattr(ipcam, "create_image_tracker", lambda: FakeTracker())
    monkeypatch.setattr(ipcam, "norfair_availability", lambda: {"status": "MOCKED"})
    monkeypatch.setattr(ipcam, "VideoOdometryReplayHarness", FakeHarness)

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("TCP probe must not run here")

    monkeypatch.setattr(ipcam, "_probe_tcp", unexpected_probe)
    return writers


def config(tmp_path: Path, **overrides) -> "ipcam.RunConfig":
    values = dict(
        url=ipcam.DEFAULT_URL,
        output_root=tmp_path / "runs",
        max_frames=3,
        open_retries=0,
        read_retries=0,
        retry_delay_s=0.0,
        read_retry_delay_s=0.0,
        weights=tmp_path / "missing-weights.pt",
    )
    values.update(overrides)
    return ipcam.RunConfig(**values)


def test_default_url_and_precedence():
    assert ipcam.DEFAULT_URL == "http://10.78.70.188:8080/video"
    env = ipcam.URL_ENV_VAR
    assert ipcam.resolve_url_with_source(None, {}) == (ipcam.DEFAULT_URL, "default")
    assert ipcam.resolve_url_with_source(
        None, {env: "http://192.0.2.10:8080/video"}) == (
            "http://192.0.2.10:8080/video", "environment")
    assert ipcam.resolve_url_with_source(
        "http://192.0.2.99:8080/video", {env: "http://192.0.2.10:8080/video"}) == (
            "http://192.0.2.99:8080/video", "cli")
    assert ipcam.resolve_url_with_source(None, {env: "   "}) == (ipcam.DEFAULT_URL, "default")


def test_video_endpoint_is_not_appended_twice():
    cases = {
        "http://10.78.70.188:8080": "http://10.78.70.188:8080/video",
        "http://10.78.70.188:8080/": "http://10.78.70.188:8080/video",
        "http://10.78.70.188:8080/video": "http://10.78.70.188:8080/video",
        "http://10.78.70.188:8080/video/": "http://10.78.70.188:8080/video",
        "10.78.70.188:8080": "http://10.78.70.188:8080/video",
        "http://host:8080/video?x=1": "http://host:8080/video?x=1",
        "http://host:8080/?x=1": "http://host:8080/video?x=1",
        "http://host:8080/stream": "http://host:8080/stream",
    }
    for raw, expected in cases.items():
        normalized = ipcam.normalize_stream_url(raw)
        assert normalized == expected
        assert "/video/video" not in normalized
    assert ipcam.normalize_stream_url(
        "https://host:8443/video") == "https://host:8443/video"


def test_invalid_urls_are_rejected():
    with pytest.raises(ValueError, match="scheme"):
        ipcam.normalize_stream_url("ftp://host/video")
    with pytest.raises(ValueError, match="non-empty"):
        ipcam.normalize_stream_url("   ")
    with pytest.raises(ValueError, match="host"):
        ipcam.normalize_stream_url("http:///video")
    with pytest.raises(ValueError, match="port"):
        ipcam.normalize_stream_url("http://host:notaport/video")
    with pytest.raises(ValueError, match="must not be empty"):
        ipcam.resolve_url_with_source("   ", {})


def test_redaction_removes_credentials():
    redacted = ipcam.redact_url("http://user:secret@10.0.0.5:8080/video")
    assert "secret" not in redacted
    assert redacted == "http://user:***@10.0.0.5:8080/video"


def test_invalid_arguments_exit_without_capture(tmp_path, monkeypatch):
    def unexpected_capture(*args, **kwargs):
        raise AssertionError("capture must not start for invalid arguments")

    monkeypatch.setattr(ipcam, "_open_capture", unexpected_capture)
    root = tmp_path / "runs"
    assert ipcam.main(["--max-frames", "0", "--output-root", str(root)]) == 2
    assert ipcam.main(["--url", "ftp://host/video", "--output-root", str(root)]) == 2
    assert ipcam.main(["--confidence", "2", "--output-root", str(root)]) == 2
    assert not root.exists()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    def unexpected_capture(*args, **kwargs):
        raise AssertionError("capture must not start when output exists")

    monkeypatch.setattr(ipcam, "_open_capture", unexpected_capture)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    assert ipcam.main(["--output", str(existing), "--max-frames", "1"]) == 2
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_main_uses_environment_url_and_cli_override(tmp_path, runtime, monkeypatch):
    seen: list[str] = []

    def fake_open(url, *args, **kwargs):
        seen.append(url)
        return FakeCapture([(True, FRAME)])

    monkeypatch.setattr(ipcam, "_open_capture", fake_open)
    monkeypatch.setenv(ipcam.URL_ENV_VAR, "http://10.0.0.9:8080")
    base = ["--max-frames", "1", "--output-root", str(tmp_path / "runs"),
            "--weights", str(tmp_path / "missing.pt")]

    assert ipcam.main(base) == 0
    assert seen[-1] == "http://10.0.0.9:8080/video"

    assert ipcam.main(base + ["--url", "http://10.0.0.10:8080/video"]) == 0
    assert seen[-1] == "http://10.0.0.10:8080/video"


def test_main_exits_cleanly_on_connection_failure(tmp_path, runtime, monkeypatch, capsys):
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([], opened=False))
    monkeypatch.setattr(
        ipcam, "_probe_tcp", lambda *args, **kwargs: (False, "ConnectionRefusedError: refused"))

    code = ipcam.main(["--max-frames", "1", "--output-root", str(tmp_path / "runs"),
                       "--weights", str(tmp_path / "missing.pt")])

    captured = capsys.readouterr()
    assert code == 2
    assert "status: CONNECTION_FAILED" in captured.err
    assert "same hotspot" in captured.err
    assert "warning: detector unavailable" in captured.err


def test_connection_failure_reports_diagnostics(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([], opened=False))
    monkeypatch.setattr(
        ipcam, "_probe_tcp", lambda *args, **kwargs: (False, "ConnectionRefusedError: refused"))

    report = ipcam.execute(config(tmp_path, open_retries=1))

    assert report["status"] == "CONNECTION_FAILED"
    assert report["exit_code"] == 2
    assert report["connection"]["status"] == "HOST_UNREACHABLE"
    assert len(report["connection"]["open_errors"]) == 2
    diagnostics = " ".join(report["connection"]["diagnostics"])
    assert "same hotspot" in diagnostics
    assert "dashboard" in diagnostics
    assert "IP Webcam app is running" in diagnostics
    saved = json.loads((Path(report["output_dir"]) / "run_report.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == 2
    assert saved["capture"]["frame_count"] == 0


def test_open_retries_are_bounded(tmp_path, runtime, monkeypatch):
    calls: list[int] = []

    def closed_capture(*args, **kwargs):
        calls.append(1)
        return FakeCapture([], opened=False)

    monkeypatch.setattr(ipcam, "_open_capture", closed_capture)
    monkeypatch.setattr(ipcam, "_probe_tcp", lambda *args, **kwargs: (True, None))

    report = ipcam.execute(config(tmp_path, open_retries=3))

    assert len(calls) == 4
    assert report["connection"]["open_attempts"] == 4
    assert report["connection"]["status"] == "ENDPOINT_UNREACHABLE"
    assert any("could not open the MJPEG stream" in hint
               for hint in report["connection"]["diagnostics"])


def test_successful_capture_records_evidence_and_limits(tmp_path, runtime, monkeypatch):
    capture = FakeCapture([(True, FRAME)] * 3, fps=0.0)
    monkeypatch.setattr(ipcam, "_open_capture", lambda *args, **kwargs: capture)

    report = ipcam.execute(config(tmp_path, max_frames=3))

    assert report["status"] == "COMPLETED"
    assert report["exit_code"] == 0
    assert report["capture"]["frame_count"] == 3
    assert report["capture"]["read_attempts"] == 3
    assert report["capture"]["writer_fps"] == 10.0
    assert report["capture"]["stream_reported_fps"] == 0.0
    assert report["detector"]["status"] == "UNAVAILABLE"
    assert "does not exist" in report["detector"]["error"]
    assert report["tracking"]["backend"] == "fake-tracker"
    assert report["tracking"]["coordinate_frame"] == "image"
    assert report["visual_motion"]["status"] == "AVAILABLE"
    assert capture.released

    limits = " ".join(report["limits"])
    assert "not hazards" in limits
    assert "not evidence that a path is safe" in limits
    assert "no commands are sent" in limits
    assert "arbitrary scale" in limits

    output = Path(report["output_dir"])
    assert (output / "run_report.json").is_file()
    assert (output / "detections.json").is_file()
    assert (output / "tracks.csv").is_file()
    events = (output / "frame_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 3
    assert all(json.loads(line)["detector_status"] == "UNAVAILABLE" for line in events)

    raw_writer = runtime[str(output / "capture.mp4")]
    annotated_writer = runtime[str(output / "annotated_capture.mp4")]
    assert len(raw_writer.frames) == 3
    assert len(annotated_writer.frames) == 3
    assert raw_writer.released and annotated_writer.released


def test_read_interruption_stops_after_bounded_retries(tmp_path, runtime, monkeypatch):
    capture = FakeCapture([(True, FRAME), (False, None), (False, None), (False, None)])
    monkeypatch.setattr(ipcam, "_open_capture", lambda *args, **kwargs: capture)

    report = ipcam.execute(config(tmp_path, max_frames=10, read_retries=2))

    assert report["status"] == "STREAM_INTERRUPTED"
    assert report["exit_code"] == 3
    assert report["capture"]["frame_count"] == 1
    assert report["capture"]["read_failures"] == 3
    assert report["capture"]["stop_reason"] == "stream_interrupted"
    assert capture.released


def test_reads_are_not_duplicated_to_hide_a_failure(tmp_path, runtime, monkeypatch):
    capture = FakeCapture([(True, FRAME), (False, None), (True, FRAME)])
    monkeypatch.setattr(ipcam, "_open_capture", lambda *args, **kwargs: capture)

    report = ipcam.execute(config(tmp_path, max_frames=2, read_retries=1))

    assert report["status"] == "COMPLETED"
    assert report["capture"]["frame_count"] == 2
    assert report["capture"]["read_failures"] == 1
    assert report["capture"]["read_recoveries"] == 1
    events = (Path(report["output_dir"]) / "frame_events.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert [json.loads(line)["frame_index"] for line in events] == [0, 1]


def test_user_interrupt_writes_report_and_skips_orb(tmp_path, runtime, monkeypatch):
    class InterruptingCapture(FakeCapture):
        def read(self):
            if self._reads:
                return super().read()
            raise KeyboardInterrupt

    capture = InterruptingCapture([(True, FRAME)])
    monkeypatch.setattr(ipcam, "_open_capture", lambda *args, **kwargs: capture)

    report = ipcam.execute(config(tmp_path, max_frames=5))

    assert report["status"] == "INTERRUPTED"
    assert report["exit_code"] == 130
    assert report["capture"]["frame_count"] == 1
    assert report["capture"]["stop_reason"] == "interrupted_by_user"
    assert report["visual_motion"]["status"] == "SKIPPED"
    assert (Path(report["output_dir"]) / "run_report.json").is_file()
    assert capture.released


def test_stream_open_without_frames_is_reported(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([], opened=True))
    monkeypatch.setattr(ipcam, "_probe_tcp", lambda *args, **kwargs: (True, None))

    report = ipcam.execute(config(tmp_path, max_frames=2, read_retries=0))

    assert report["status"] == "NO_FRAMES"
    assert report["exit_code"] == 3
    assert report["connection"]["status"] == "CONNECTED_NO_FRAMES"
    diagnostics = " ".join(report["connection"]["diagnostics"])
    assert "stream handle opened" in diagnostics
    assert "no decodable frame" in diagnostics


def test_https_failure_points_to_plain_http():
    hints = " ".join(ipcam._connection_diagnostics(
        "https://10.78.70.188:8080/video", tcp_reachable=True, tcp_error=None))
    assert "plain HTTP" in hints
    hints = " ".join(ipcam._connection_diagnostics(
        "http://10.78.70.188:443/video", tcp_reachable=False, tcp_error="refused"))
    assert "implies HTTPS" in hints
    assert "same hotspot" in hints


def test_provider_errors_do_not_become_detections(tmp_path, runtime, monkeypatch):
    class ExplodingDetector:
        config = {"identifier": "exploding"}

        def __call__(self, frame):
            raise RuntimeError("synthetic provider failure")

    monkeypatch.setattr(ipcam, "_load_detector", lambda *args, **kwargs: ExplodingDetector())
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([(True, FRAME)] * 2))

    report = ipcam.execute(config(tmp_path, max_frames=2))

    assert report["status"] == "COMPLETED"
    assert report["detector"]["status"] == "AVAILABLE"
    assert report["detector"]["provider_error_frames"] == 2
    detections = json.loads(
        (Path(report["output_dir"]) / "detections.json").read_text(encoding="utf-8"))
    assert detections["frames"] == []
    events = (Path(report["output_dir"]) / "frame_events.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert all(json.loads(line)["detector_status"] == "PROVIDER_ERROR" for line in events)
    assert all(json.loads(line)["detection_count"] == 0 for line in events)


def test_visual_motion_failure_is_recorded(tmp_path, runtime, monkeypatch):
    class BrokenHarness:
        def replay(self, path, **kwargs):
            raise RuntimeError("synthetic harness failure")

    monkeypatch.setattr(ipcam, "VideoOdometryReplayHarness", BrokenHarness)
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([(True, FRAME)] * 2))

    report = ipcam.execute(config(tmp_path, max_frames=2))

    assert report["status"] == "COMPLETED"
    assert report["visual_motion"]["status"] == "UNAVAILABLE"
    assert "synthetic harness failure" in report["visual_motion"]["reason"]


def test_each_run_gets_a_unique_directory(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(
        ipcam, "_open_capture", lambda *args, **kwargs: FakeCapture([(True, FRAME)]))
    root = tmp_path / "runs"

    first = ipcam.execute(config(tmp_path, output_root=root, max_frames=1))
    second = ipcam.execute(config(tmp_path, output_root=root, max_frames=1))

    assert first["output_dir"] != second["output_dir"]
    assert Path(first["output_dir"]).is_dir()
    assert Path(second["output_dir"]).is_dir()
    assert Path(first["output_dir"]).parent == Path(second["output_dir"]).parent == root
