"""Advisory-only live IP Webcam capture and perception probe.

Connects to the Android IP Webcam MJPEG endpoint (default
``http://10.78.70.188:8080/video``), decodes a bounded number of frames, runs
the existing LibreYOLO9t provider and image-space tracker, records the stream to
MP4, and audits the existing ORB/fundamental-matrix front end on the recorded
clip.

Nothing here commands a vehicle, flight controller, or motor. Detections are
2-D image observations, not hazards, depth, traversability, or safety; visual
motion is diagnostic and monocular (no metric scale without calibration).

Connection attempts, read retries, and the frame budget are all bounded. No
frame is duplicated to cover a failed read, and frames are timestamped when
they are read, with gaps recorded rather than hidden.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import sys
import time
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
except ImportError as error:  # pragma: no cover - dependency failure
    raise SystemExit(
        "OpenCV (cv2) is required for live capture; install requirements.txt in .venv"
    ) from error

from prism.belief_ledger import EvidenceRelation
from prism.integrations.tracking import create_image_tracker, norfair_availability
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.video_observation_replay import DecodedFrame
from prism.visual_odometry_replay import (
    ARBITRARY_SCALE_NOTE,
    CALIBRATION_REQUIREMENT,
    COORDINATE_FRAME_NOTE,
    VideoOdometryReplayHarness,
)

DEFAULT_URL = "http://10.78.70.188:8080/video"
URL_ENV_VAR = "PRISM_IPWEBCAM_URL"
STREAM_ENDPOINT = "/video"
CAPTURE_START_NS = 1_000_000_000
DETECTION_HYPOTHESIS = "object-detection-evidence"

LIMITS = (
    "Advisory-only experiment: no commands are sent to a drone, flight controller, motor, or vehicle.",
    "Detections are 2-D COCO image observations, not hazards, depth, clearance, or traversability; "
    "an empty detection list is not evidence that a path is safe.",
    "Visual motion is monocular ORB/fundamental-matrix diagnostics over the recorded clip; without "
    "validated calibration it is tracking-only, and any relative motion has arbitrary scale.",
    "The MJPEG stream has no hardware timestamps; frames are timestamped when read and gaps are "
    "recorded, so buffered frames may be older than their processing time.",
    "No GPS-denied navigation, metric localization, terrain mapping, obstacle-avoidance, or hazard "
    "accuracy is validated by this tool.",
)


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def get(self, property_id: int) -> float: ...
    def set(self, property_id: int, value: float) -> bool: ...
    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...


class WriterLike(Protocol):
    def isOpened(self) -> bool: ...
    def write(self, frame: Any) -> None: ...
    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One bounded live-capture run; every input is explicit."""

    url: str
    url_source: str = "default"
    output: Path | None = None
    output_root: Path = PROJECT_ROOT / "output" / "ipwebcam_runs"
    max_frames: int = 30
    fallback_fps: float = 10.0
    open_retries: int = 2
    read_retries: int = 3
    retry_delay_s: float = 0.5
    read_retry_delay_s: float = 0.2
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    stale_gap_ms: float = 1000.0
    weights: Path = PROJECT_ROOT / "model_assets" / "libreyolo" / "LibreYOLO9t.pt"
    confidence: float = 0.25

    def __post_init__(self) -> None:
        _require_int("max_frames", self.max_frames, 1, 3600)
        _require_int("open_retries", self.open_retries, 0, 10)
        _require_int("read_retries", self.read_retries, 0, 10)
        _require_int("open_timeout_ms", self.open_timeout_ms, 100, 60000)
        _require_int("read_timeout_ms", self.read_timeout_ms, 100, 60000)
        _require_finite("fallback_fps", self.fallback_fps, 0.0, 240.0, minimum_inclusive=False)
        _require_finite("retry_delay_s", self.retry_delay_s, 0.0, 30.0)
        _require_finite("read_retry_delay_s", self.read_retry_delay_s, 0.0, 30.0)
        _require_finite("stale_gap_ms", self.stale_gap_ms, 0.0, 600000.0, minimum_inclusive=False)
        _require_finite("confidence", self.confidence, 0.0, 1.0)
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "weights", Path(self.weights))


def _require_int(name: str, value: Any, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _require_finite(name: str, value: Any, low: float, high: float,
                    *, minimum_inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    if not minimum_inclusive and number == low:
        raise ValueError(f"{name} must be greater than {low}")


def normalize_stream_url(url: str) -> str:
    """Return a plain-HTTP(s) MJPEG URL with the standard ``/video`` endpoint.

    ``/video`` is only added when the path is empty or ``/`` so an explicit
    endpoint is never doubled. IP Webcam serves plain HTTP by default.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("stream URL must be a non-empty string")
    text = url.strip()
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(
            f"unsupported stream URL scheme {scheme!r}: use http:// (IP Webcam default) "
            "or https:// only when the server is explicitly configured for SSL")
    if not parts.hostname:
        raise ValueError("stream URL must include a host, for example http://10.78.70.188:8080/video")
    try:
        parts.port  # noqa: B018 - validates the port eagerly
    except ValueError as error:
        raise ValueError(f"invalid port in stream URL: {error}") from error

    path = parts.path
    if path in ("", "/"):
        path = STREAM_ENDPOINT
    elif path.rstrip("/").lower().endswith(STREAM_ENDPOINT):
        path = path.rstrip("/")
    return urlunsplit((scheme, parts.netloc, path, parts.query, parts.fragment))


def resolve_url_with_source(cli_url: str | None,
                            environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Apply precedence: explicit ``--url``, then ``PRISM_IPWEBCAM_URL``, then default."""
    env = os.environ if environ is None else environ
    if cli_url is not None:
        if not cli_url.strip():
            raise ValueError("--url must not be empty")
        return normalize_stream_url(cli_url), "cli"
    env_url = env.get(URL_ENV_VAR)
    if env_url is not None and env_url.strip():
        return normalize_stream_url(env_url), "environment"
    return normalize_stream_url(DEFAULT_URL), "default"


def redact_url(url: str) -> str:
    """Hide credentials if a URL contains them; never write them to reports."""
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        username = userinfo.split(":", 1)[0]
        netloc = f"{username}:***@{host}" if ":" in userinfo else f"{username}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _source_identifier(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or "unknown"
    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    return f"ipwebcam:{host}:{port}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latency(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    at = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {"count": len(ordered), "p50_ms": at(.5), "p95_ms": at(.95), "max_ms": ordered[-1]}


def _prepare_output(config: RunConfig) -> Path:
    if config.output is not None:
        path = Path(config.output)
        if path.exists():
            raise ValueError(f"refusing to overwrite existing output directory: {path}")
        path.mkdir(parents=True, exist_ok=False)
        return path
    return _unique_run_dir(Path(config.output_root))


def _unique_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(8):
        candidate = root / f"{stamp}-{secrets.token_hex(3)}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {root}")


def _open_capture(url: str, open_timeout_ms: int, read_timeout_ms: int) -> CaptureLike:
    """Open the stream with the FFmpeg backend and bounded open/read timeouts."""
    params: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params += [int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), int(open_timeout_ms)]
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params += [int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), int(read_timeout_ms)]

    capture: CaptureLike | None = None
    if params:
        try:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            capture = None
    if capture is None:
        try:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        except cv2.error:
            capture = cv2.VideoCapture(url)
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
    except Exception:
        pass
    return capture


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> WriterLike:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)


def _probe_tcp(host: str, port: int, timeout_s: float = 2.0) -> tuple[bool, str | None]:
    if not host:
        return False, "no host in stream URL"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True, None
    except OSError as error:
        return False, f"{type(error).__name__}: {error}"


def _connection_diagnostics(url: str, *, tcp_reachable: bool, tcp_error: str | None,
                            stream_opened: bool = False) -> list[str]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname or "<phone-ip>"
    port = parts.port or (443 if scheme == "https" else 80)
    dashboard = f"{scheme}://{host}:{port}/"
    hints = [
        "Confirm the phone and this PC are on the same hotspot or Wi-Fi network; the phone "
        "address changes when the hotspot restarts.",
        f"Open {dashboard} in a browser on this PC. If the IP Webcam dashboard does not load, "
        "this is a network/connectivity problem, not a P.R.I.S.M. defect.",
        "Confirm the IP Webcam app is running and its server is started; the MJPEG stream is "
        "usually plain HTTP on port 8080.",
    ]
    if parts.port is None:
        hints.append(
            "The URL has no explicit port; IP Webcam usually listens on 8080, for example "
            "http://<phone-ip>:8080/video.")
    if scheme == "https":
        hints.append(
            "IP Webcam serves plain HTTP by default; use http:// unless SSL was explicitly "
            "configured on the server.")
    elif port in (443, 8443):
        hints.append(
            "This port usually implies HTTPS; if SSL is not configured, use "
            "http://<phone-ip>:8080/video.")
    if stream_opened:
        hints.append(
            "The stream handle opened but no decodable frame arrived within the read timeout; "
            "verify that /video is the MJPEG endpoint, that the phone is streaming, and that "
            "OpenCV/FFmpeg supports the stream.")
    elif tcp_reachable:
        hints.append(
            f"TCP {host}:{port} is reachable, but OpenCV/FFmpeg could not open the MJPEG stream; "
            f"verify the {STREAM_ENDPOINT} endpoint and the stream codec.")
    else:
        hints.append(
            f"TCP {host}:{port} failed: the server may not be running, the IP/port may be stale, "
            "or a firewall may block it.")
        if tcp_error:
            hints.append(f"TCP probe detail: {tcp_error}")
    normalized_path = parts.path.rstrip("/")
    if normalized_path and normalized_path.lower() != STREAM_ENDPOINT:
        hints.append(
            f"The configured path is {parts.path!r}; the standard IP Webcam MJPEG endpoint is "
            f"{STREAM_ENDPOINT}.")
    hints.append(
        f"Override the address with --url or the {URL_ENV_VAR} environment variable instead of "
        "editing the tool.")
    return hints


def _diagnose_failure(url: str, open_errors: Sequence[str], *,
                      stream_opened: bool = False) -> dict:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    reachable, probe_error = _probe_tcp(host, port)
    if stream_opened:
        status = "CONNECTED_NO_FRAMES"
    elif reachable:
        status = "ENDPOINT_UNREACHABLE"
    else:
        status = "HOST_UNREACHABLE"
    return {
        "status": status,
        "host": host,
        "port": port,
        "tcp_probe": {"reachable": reachable, "error": probe_error},
        "open_errors": list(open_errors),
        "diagnostics": _connection_diagnostics(
            url, tcp_reachable=reachable, tcp_error=probe_error, stream_opened=stream_opened),
    }


def _load_detector(weights: Path, confidence: float, *, source: str):
    return LibreYOLODetectionProvider.from_checkpoint(
        weights, source=source, hypothesis_id=DETECTION_HYPOTHESIS,
        relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000,
        confidence_threshold=confidence)


def _draw(frame, tracks, status: str, frame_index: int, gap_ms: float):
    for track in tracks:
        x0, y0, x1, y1 = (round(value) for value in track.bbox_xyxy)
        color = (0, 210, 0) if track.observed else (0, 170, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        tag = f"T{track.track_id} {track.class_id} {track.score:.2f}"
        if track.predicted:
            tag += " predicted"
        cv2.putText(frame, tag, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    label = f"frame {frame_index} | {status} | gap {gap_ms:.0f} ms | advisory only"
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def _base_report(config: RunConfig, output_dir: Path, created_at: str,
                 weights_sha256: str | None) -> dict:
    return {
        "format": "prism-ipwebcam-live/v1",
        "status": None,
        "exit_code": None,
        "created_at_utc": created_at,
        "output_dir": str(output_dir),
        "configuration": {
            "source_url": redact_url(config.url),
            "url_source": config.url_source,
            "max_frames": config.max_frames,
            "writer_fps_fallback": config.fallback_fps,
            "open_retries": config.open_retries,
            "read_retries": config.read_retries,
            "open_timeout_ms": config.open_timeout_ms,
            "read_timeout_ms": config.read_timeout_ms,
            "stale_gap_threshold_ms": config.stale_gap_ms,
            "checkpoint": str(config.weights),
            "checkpoint_sha256": weights_sha256,
            "confidence_threshold": config.confidence,
        },
        "connection": {"status": "NOT_ATTEMPTED", "open_errors": [], "diagnostics": []},
        "capture": {
            "frame_count": 0,
            "read_attempts": 0,
            "read_failures": 0,
            "read_recoveries": 0,
            "stale_gaps": 0,
            "stop_reason": None,
            "stream_reported_fps": None,
            "writer_fps": None,
            "dimensions": None,
            "capture_duration_s": 0.0,
            "measured_fps": None,
            "videos": {"raw": None, "annotated": None, "writer_errors": []},
        },
        "detector": {
            "provider": "LibreYOLODetectionProvider",
            "status": "NOT_LOADED",
            "error": None,
            "checkpoint": str(config.weights),
            "checkpoint_exists": Path(config.weights).is_file(),
            "weights_sha256": weights_sha256,
            "confidence_threshold": config.confidence,
            "model": None,
            "latency_ms": _latency([]),
            "provider_error_frames": 0,
        },
        "tracking": {"backend": None, "norfair": None, "coordinate_frame": "image", "records": 0},
        "visual_motion": {"status": "NOT_RUN", "reason": None},
        "limits": list(LIMITS),
    }


def execute(config: RunConfig) -> dict:
    """Run one bounded capture; always returns a report and writes it to disk."""
    started = perf_counter()
    output_dir = _prepare_output(config)
    weights_sha256 = None
    if Path(config.weights).is_file():
        weights_sha256 = _sha256_file(Path(config.weights))
    report = _base_report(config, output_dir, datetime.now(timezone.utc).isoformat(), weights_sha256)

    detector = None
    detector_error = None
    try:
        detector = _load_detector(Path(config.weights), config.confidence,
                                  source=_source_identifier(config.url))
    except Exception as error:
        detector_error = f"{type(error).__name__}: {error}"
    report["detector"].update({
        "status": "AVAILABLE" if detector is not None else "UNAVAILABLE",
        "error": detector_error,
        "model": None if detector is None else _json(detector.config),
    })

    tracker = create_image_tracker()
    report["tracking"].update({"backend": tracker.backend, "norfair": norfair_availability()})

    open_errors: list[str] = []
    open_attempts = 0
    capture: CaptureLike | None = None
    for attempt in range(config.open_retries + 1):
        open_attempts += 1
        candidate: CaptureLike | None = None
        try:
            candidate = _open_capture(config.url, config.open_timeout_ms, config.read_timeout_ms)
        except Exception as error:
            open_errors.append(f"attempt {open_attempts}: {type(error).__name__}: {error}")
        if candidate is not None and candidate.isOpened():
            capture = candidate
            break
        if candidate is not None:
            try:
                candidate.release()
            except Exception:
                pass
            open_errors.append(
                f"attempt {open_attempts}: stream handle returned but isOpened() is false")
        if attempt < config.open_retries:
            time.sleep(config.retry_delay_s)

    if capture is None:
        connection = _diagnose_failure(config.url, open_errors)
        connection["open_attempts"] = open_attempts
        report["connection"] = connection
        report.update({"status": "CONNECTION_FAILED", "exit_code": 2})
        _write_json(output_dir / "run_report.json", report)
        _write_readme(output_dir, report)
        return report

    report["connection"] = {
        "status": "CONNECTED",
        "open_attempts": open_attempts,
        "open_errors": open_errors,
        "host": urlsplit(config.url).hostname,
        "port": urlsplit(config.url).port
        or (443 if urlsplit(config.url).scheme.lower() == "https" else 80),
        "tcp_probe": None,
        "diagnostics": [],
    }

    stream_fps = 0.0
    try:
        stream_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        stream_fps = 0.0
    if not math.isfinite(stream_fps):
        stream_fps = 0.0
    writer_fps = (stream_fps if 0.0 < stream_fps <= 240.0 else float(config.fallback_fps))
    report["capture"]["stream_reported_fps"] = stream_fps
    report["capture"]["writer_fps"] = writer_fps

    events: list[dict] = []
    detections_log: list[dict] = []
    tracks_log: list[dict] = []
    latencies: list[float] = []
    provider_errors = 0
    read_failures = 0
    read_recoveries = 0
    consecutive_failures = 0
    stale_gaps = 0
    frame_index = 0
    size: tuple[int, int] | None = None
    last_capture_ns: int | None = None
    raw_writer: WriterLike | None = None
    annotated_writer: WriterLike | None = None
    writer_errors: list[str] = []
    stop_reason: str | None = None
    processing_error: str | None = None
    capture_started = perf_counter()

    try:
        while frame_index < config.max_frames:
            try:
                ok, bgr = capture.read()
            except Exception as error:
                ok, bgr = False, None
                writer_errors.append(f"read exception: {type(error).__name__}: {error}")
            if not ok or bgr is None:
                read_failures += 1
                consecutive_failures += 1
                if consecutive_failures <= config.read_retries:
                    time.sleep(config.read_retry_delay_s)
                    continue
                stop_reason = "stream_interrupted"
                break
            if consecutive_failures:
                read_recoveries += 1
                consecutive_failures = 0

            if getattr(bgr, "ndim", 0) != 3 or bgr.shape[2] != 3:
                stop_reason = "invalid_frame"
                break
            height, width = int(bgr.shape[0]), int(bgr.shape[1])
            if size is None:
                size = (width, height)
                raw_writer, annotated_writer = _start_writers(
                    output_dir, writer_fps, size, writer_errors)
            elif size != (width, height):
                stop_reason = "frame_dimensions_changed"
                break

            capture_at_ns = time.monotonic_ns()
            if last_capture_ns is not None and capture_at_ns <= last_capture_ns:
                capture_at_ns = last_capture_ns + 1
            gap_ms = 0.0 if last_capture_ns is None else (capture_at_ns - last_capture_ns) / 1e6
            if last_capture_ns is not None and gap_ms > config.stale_gap_ms:
                stale_gaps += 1

            detector_status = "UNAVAILABLE"
            detector_error_for_frame = None
            frame_detections: Sequence[Any] = ()
            detector_latency_ms = None
            if detector is not None:
                tick = perf_counter()
                try:
                    supplied = detector(DecodedFrame(frame_index, capture_at_ns, width, height, bgr))
                    detector_latency_ms = (perf_counter() - tick) * 1000.0
                    latencies.append(detector_latency_ms)
                    observation = supplied.observation
                    evidence = observation.evidence.object_detections
                    frame_detections = () if evidence is None else evidence.detections
                    detector_status = "AVAILABLE"
                    detections_log.append({
                        "frame_index": frame_index,
                        "capture_at_ns": capture_at_ns,
                        "observation_id": observation.observation_id,
                        "detections": [
                            {"class_id": item.class_id, "taxonomy": item.taxonomy,
                             "score": item.score, "bbox_xyxy": list(item.bbox_xyxy)}
                            for item in frame_detections],
                        "provenance": _json(observation.provenance),
                    })
                except Exception as error:
                    detector_latency_ms = (perf_counter() - tick) * 1000.0
                    latencies.append(detector_latency_ms)
                    provider_errors += 1
                    detector_status = "PROVIDER_ERROR"
                    detector_error_for_frame = f"{type(error).__name__}: {error}"

            try:
                frame_tracks = tracker.update(
                    frame_detections, frame_index=frame_index, capture_at_ns=capture_at_ns)
            except Exception as error:
                frame_tracks = ()
                writer_errors.append(f"tracker error: {type(error).__name__}: {error}")
            tracks_log.extend(_json(item) for item in frame_tracks)

            if raw_writer is not None:
                raw_writer.write(bgr)
            if annotated_writer is not None:
                annotated_writer.write(_draw(
                    bgr.copy(), frame_tracks, detector_status, frame_index, gap_ms))

            events.append({
                "frame_index": frame_index,
                "capture_at_ns": capture_at_ns,
                "read_gap_ms": gap_ms,
                "stale_gap": last_capture_ns is not None and gap_ms > config.stale_gap_ms,
                "width": width,
                "height": height,
                "detector_status": detector_status,
                "detector_error": detector_error_for_frame,
                "detector_latency_ms": detector_latency_ms,
                "detection_count": len(frame_detections),
                "track_count": len(frame_tracks),
                "tracks": [{"track_id": item.track_id, "observed": item.observed,
                            "predicted": item.predicted} for item in frame_tracks],
            })
            frame_index += 1
            last_capture_ns = capture_at_ns
    except KeyboardInterrupt:
        stop_reason = "interrupted_by_user"
    except Exception as error:
        processing_error = f"{type(error).__name__}: {error}"
        stop_reason = "processing_error"
    finally:
        try:
            capture.release()
        except Exception:
            pass
        for writer in (raw_writer, annotated_writer):
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    duration_s = perf_counter() - capture_started
    measured_fps = (frame_index / duration_s) if duration_s > 0 else None
    if frame_index >= config.max_frames and stop_reason is None:
        stop_reason = "frame_limit"

    report["capture"].update({
        "frame_count": frame_index,
        "read_attempts": frame_index + read_failures,
        "read_failures": read_failures,
        "read_recoveries": read_recoveries,
        "stale_gaps": stale_gaps,
        "stop_reason": stop_reason,
        "dimensions": None if size is None else [size[0], size[1]],
        "capture_duration_s": duration_s,
        "measured_fps": measured_fps,
        "fps_note": "end-to-end read cadence including inference, not a camera hardware rate",
        "videos": {
            "raw": "capture.mp4" if raw_writer is not None else None,
            "annotated": "annotated_capture.mp4" if annotated_writer is not None else None,
            "writer_errors": writer_errors,
            "timing_note": "writer FPS is nominal; frames are not hardware-synchronized",
        },
    })
    report["detector"].update({
        "latency_ms": _latency(latencies),
        "provider_error_frames": provider_errors,
    })
    report["tracking"]["records"] = len(tracks_log)
    if processing_error is not None:
        report["capture"]["processing_error"] = processing_error

    if stop_reason == "interrupted_by_user":
        report["visual_motion"] = {
            "status": "SKIPPED",
            "reason": "run interrupted by the user; recorded-clip diagnostics were not run",
        }
    else:
        report["visual_motion"] = _visual_motion_report(
            output_dir, frame_index, raw_writer is not None)

    if stop_reason == "interrupted_by_user":
        status, exit_code = "INTERRUPTED", 130
    elif frame_index == 0:
        connection = _diagnose_failure(
            config.url,
            [f"stream opened but no frame was decoded after {read_failures} read attempt(s)"],
            stream_opened=True)
        connection["open_attempts"] = open_attempts
        report["connection"] = connection
        status, exit_code = "NO_FRAMES", 3
    elif stop_reason == "processing_error":
        status, exit_code = "PROCESSING_ERROR", 4
    elif stop_reason in {"stream_interrupted", "frame_dimensions_changed", "invalid_frame"}:
        status, exit_code = "STREAM_INTERRUPTED", 3
    else:
        status, exit_code = "COMPLETED", 0
    report.update({"status": status, "exit_code": exit_code})

    _write_reports(output_dir, report, events, detections_log, tracks_log)
    _write_readme(output_dir, report)
    return report


def _start_writers(output_dir: Path, fps: float, size: tuple[int, int],
                   errors: list[str]) -> tuple[WriterLike | None, WriterLike | None]:
    writers: list[WriterLike | None] = []
    for name, path in (("raw", output_dir / "capture.mp4"),
                       ("annotated", output_dir / "annotated_capture.mp4")):
        writer = None
        try:
            writer = _open_writer(path, fps, size)
            if not writer.isOpened():
                writer.release()
                writer = None
                errors.append(f"{name} video writer could not be opened")
        except Exception as error:
            writer = None
            errors.append(f"{name} video writer error: {type(error).__name__}: {error}")
        writers.append(writer)
    return writers[0], writers[1]


def _visual_motion_report(output_dir: Path, frame_count: int, video_written: bool) -> dict:
    if frame_count < 2:
        return {"status": "UNAVAILABLE",
                "reason": "at least two decoded frames are required for inter-frame diagnostics"}
    if not video_written:
        return {"status": "UNAVAILABLE", "reason": "capture video was not written"}
    path = output_dir / "capture.mp4"
    try:
        harness_report = VideoOdometryReplayHarness().replay(
            path, max_frames=frame_count, capture_start_ns=CAPTURE_START_NS)
        summary = harness_report.summary()
    except Exception as error:
        return {"status": "UNAVAILABLE",
                "reason": f"{type(error).__name__}: {error}"}
    if not harness_report.opened or harness_report.decoded_frames == 0:
        return {"status": "UNAVAILABLE",
                "reason": "recorded clip could not be decoded for motion diagnostics",
                "summary": summary}
    with (output_dir / "camera_motion.csv").open("w", encoding="utf-8") as file:
        file.write("frame_index,capture_order_ns,status,n_features,n_matches,n_inliers,"
                   "inlier_ratio,latency_ms,reason\n")
        for item in harness_report.records:
            file.write(",".join(str(value) for value in (
                item.frame_index, item.capture_order_ns, item.status.value,
                item.n_features, item.n_matches, item.n_inliers, item.inlier_ratio,
                item.latency_ms, item.reason.replace(",", ";"))) + "\n")
    return {
        "status": "AVAILABLE",
        **summary,
        "interpretation": {
            "claim": "diagnostic only; not metric odometry, localization, or navigation readiness",
            "calibration": CALIBRATION_REQUIREMENT,
            "scale": ARBITRARY_SCALE_NOTE,
            "frame": COORDINATE_FRAME_NOTE,
        },
    }


def _write_reports(output_dir: Path, report: dict, events: list[dict],
                   detections: list[dict], tracks: list[dict]) -> None:
    _write_json(output_dir / "run_report.json", report)
    with (output_dir / "frame_events.jsonl").open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(_json(event), sort_keys=True) + "\n")
    _write_json(output_dir / "detections.json", {
        "detector_error": report["detector"]["error"],
        "frames": detections,
    })
    with (output_dir / "tracks.csv").open("w", encoding="utf-8") as file:
        file.write("track_id,frame_index,capture_at_ns,class_id,taxonomy,score,x_min,y_min,"
                   "x_max,y_max,observed,predicted,age_frames,prediction_note\n")
        for item in tracks:
            box = item["bbox_xyxy"]
            file.write(",".join(str(value) for value in (
                item["track_id"], item["frame_index"], item["capture_at_ns"],
                item["class_id"], item["taxonomy"], item["score"], *box,
                item["observed"], item["predicted"], item["age_frames"],
                (item["prediction_note"] or "").replace(",", ";"))) + "\n")


def _write_readme(output_dir: Path, report: dict) -> None:
    lines = [
        "# P.R.I.S.M. live IP webcam run\n\n",
        f"Status: `{report['status']}`\n\n",
        "1. `run_report.json` records configuration, connection diagnostics, latency, and limits.\n",
        "2. `capture.mp4` is the bounded recording; `annotated_capture.mp4` adds 2-D boxes and track IDs.\n",
        "3. `detections.json`, `tracks.csv`, and `frame_events.jsonl` hold per-frame evidence.\n",
        "4. `camera_motion.csv` and the `visual_motion` report section are diagnostic ORB/fundamental-matrix results.\n\n",
        "This run is advisory-only. Detections are not hazards, no vehicle command is produced, and "
        "missing detections or motion evidence never prove that a path is safe.\n",
    ]
    (output_dir / "README.md").write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded, advisory-only live IP Webcam perception probe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/run_prism_ipwebcam.py\n"
            "  python tools/run_prism_ipwebcam.py --url http://10.78.70.188:8080/video --max-frames 60\n"
            f"  set {URL_ENV_VAR}=http://10.78.70.188:8080/video\n"
        ))
    parser.add_argument("--url", default=None,
                        help=f"IP Webcam MJPEG URL (default: ${URL_ENV_VAR}, then {DEFAULT_URL})")
    parser.add_argument("--output", type=Path, default=None,
                        help="exact new output directory; existing directories are never overwritten")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "output" / "ipwebcam_runs",
                        help="root for a unique timestamped run directory")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="bounded number of frames to capture (1-3600)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="writer FPS used only when the stream reports none")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "model_assets" / "libreyolo" / "LibreYOLO9t.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--open-retries", type=int, default=2,
                        help="bounded stream-open retries after the first attempt")
    parser.add_argument("--read-retries", type=int, default=3,
                        help="bounded consecutive read retries before stopping")
    parser.add_argument("--retry-delay-ms", type=float, default=500.0)
    parser.add_argument("--read-retry-delay-ms", type=float, default=200.0)
    parser.add_argument("--open-timeout-ms", type=int, default=5000)
    parser.add_argument("--read-timeout-ms", type=int, default=5000)
    parser.add_argument("--stale-gap-ms", type=float, default=1000.0,
                        help="read gaps above this are counted as stale-frame gaps")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        url, url_source = resolve_url_with_source(args.url)
        config = RunConfig(
            url=url, url_source=url_source, output=args.output,
            output_root=args.output_root, max_frames=args.max_frames,
            fallback_fps=args.fps, open_retries=args.open_retries,
            read_retries=args.read_retries,
            retry_delay_s=args.retry_delay_ms / 1000.0,
            read_retry_delay_s=args.read_retry_delay_ms / 1000.0,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
            stale_gap_ms=args.stale_gap_ms, weights=args.weights,
            confidence=args.confidence)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        report = execute(config)
    except KeyboardInterrupt:
        print("status: INTERRUPTED", file=sys.stderr)
        return 130
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # clean CLI failure instead of a traceback
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    if report["detector"]["status"] != "AVAILABLE":
        print(f"warning: detector unavailable: {report['detector']['error']}", file=sys.stderr)
    if report["status"] != "COMPLETED":
        print(f"status: {report['status']}", file=sys.stderr)
        for hint in report.get("connection", {}).get("diagnostics", []):
            print(f"hint: {hint}", file=sys.stderr)
    print(json.dumps({
        "output": report["output_dir"],
        "status": report["status"],
        "frames": report["capture"]["frame_count"],
        "exit_code": report["exit_code"],
    }, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
