"""Native live vision workbench for P.R.I.S.M. - advisory only.

Opens a real camera (PC webcam or the Android IP Webcam MJPEG stream) in a
native OpenCV window and runs the perception capabilities that this repository
can actually execute today, using their existing entry points:

* 2-D detection through ``prism.libreyolo_provider.LibreYOLODetectionProvider``;
* image-space tracking through ``prism.integrations.tracking.create_image_tracker``;
* ORB / fundamental-matrix camera-motion diagnostics built from the same
  primitives as ``prism.visual_odometry_replay`` (``_OrbFrontend``,
  ``_mutual_ratio_match``, ``estimate_fundamental_inliers``,
  ``estimate_relative_motion``);
* optional YOLOP evaluation through ``prism.integrations.yolop_provider``;
* real evidence state through ``prism.belief_ledger.BeliefLedger`` and
  ``prism.doubt_engine.DoubtEngine``;
* an offline COLMAP reconstruction workflow reusing
  ``tools/colmap_sparse_probe.run_colmap``, ``prism.colmap_inspection`` and
  ``prism.perception.cloud_accum.render_points``.

The honesty contract is the point of this tool. Depth and semantic segmentation
run REAL inference from pretrained weights that were downloaded once by
``tools/download_prism_pretrained.py`` into ``model_assets/pretrained/`` (Depth
Anything V2-Small and SegFormer-B0/ADE20K); loading is local-only and the viewer
never downloads anything. What those views show is labelled for what it is:
relative inverse depth with metres aligned to the ASSUMED 0.12 m camera height
(never a calibrated distance), and pretrained ADE20K semantics mapped onto
prism-7 (never ground-truth traversability). The terrain view is an experimental
GEOMETRIC grid at assumed scale with an UNKNOWN floor, explicitly UNVALIDATED;
this viewer never loads a learned traversability model (``checkpoints/
trav_unc_shared.pt`` belongs to the separate manual training task). The metric
terrain gate and the offline COLMAP workflow are unchanged.
See ``docs/live_capability_inventory.md``.

Nothing here commands a vehicle: ``prism/runtime.py``, ``prism/nav/planner.py``,
``prism/nav/supervisor.py`` and ``ros2/`` are deliberately not imported.
Detections are 2-D observations, not hazards; an empty detection list is not
evidence that a path is clear; monocular motion has arbitrary scale; the MJPEG
stream has no hardware timestamps.
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
except ImportError as error:  # pragma: no cover - dependency failure
    raise SystemExit(
        "OpenCV (cv2) is required for the live viewer; install requirements.txt in .venv"
    ) from error

import numpy as np

from prism.belief_ledger import BeliefLedger, EvidenceRelation
from prism.calibration import CalibrationError
from prism.colmap_inspection import read_sparse_model
from prism.config import CFG, PROC_H, PROC_W
from prism.doubt_engine import DoubtEngine
from prism.integrations.geometry import sparse_geometry_report
from prism.integrations.terrain import terrain_status
from prism.integrations.tracking import create_image_tracker, norfair_availability
from prism.integrations.yolop_provider import (
    YOLOP_320_SHA256,
    YOLOPDetectionProvider,
    segmentation_summary,
)
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.observations import (CoordinateFrame, EvidencePayload, Observation, Validity)
from prism.perception import geometry as _pgeo
from prism.perception.cloud_accum import WorldCamera, render_points
from prism.perception.mapping import MappingStage, cell_stats
from prism.types import FramePacket
from prism.viz_common import colorize_height, colorize_terrain, colorize_trav, overlay as viz_overlay
from prism.runtime_adapter import RuntimeAdapter
from prism.video_observation_replay import DecodedFrame
from prism.visual_odometry_replay import (
    ARBITRARY_SCALE_NOTE,
    CALIBRATION_REQUIREMENT,
    CameraCalibration,
    MotionStatus,
    OdomParams,
    _OrbFrontend,
    _mutual_ratio_match,
    estimate_fundamental_inliers,
    estimate_relative_motion,
)

DEFAULT_COLMAP_EXE = (PROJECT_ROOT / "tools" / "external" / "colmap"
                      / "4.2.1-nocuda" / "bin" / "colmap.exe")
YOLOP_ONNX = PROJECT_ROOT / "tools" / "external" / "yolop" / "yolop-320-320.onnx"
DEFAULT_DEPTH_DIR = (PROJECT_ROOT / "model_assets" / "pretrained"
                     / "depth_anything_v2_small_hf")
DEFAULT_SEGMENTATION_DIR = (PROJECT_ROOT / "model_assets" / "pretrained"
                            / "segformer_b0_ade_512_512")
TRAV_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "trav_unc_shared.pt"
DEPTH_MODEL_ID_FOR_LOADER = "depth-anything/Depth-Anything-V2-Small-hf"
SEGMENTATION_MODEL_ID_FOR_LOADER = "nvidia/segformer-b0-finetuned-ade-512-512"
HYPOTHESIS_DETECTION = "object-detection-evidence"
HYPOTHESIS_TERRAIN = "terrain-spatial-evidence"
SOURCE_WEBCAM = "webcam"
SOURCE_IPWEBCAM = "ipwebcam"
TOOL_NOTE = (
    "The URL, capture-open and connection-diagnostic helpers are imported from the "
    "audited recording runner (tools/run_prism_ipwebcam.py) instead of being "
    "reimplemented here.")
MAX_MATCH_SEGMENTS = 240


def learned_traversability_note() -> str:
    """Whether this viewer has a learned traversability model, stated from actual state.

    The terrain view always draws the geometric mapping stage; the checkpoint belongs to the
    separate manual training task and may appear in the repository while the viewer runs, so the
    wording must stay true either way.
    """
    if TRAV_CHECKPOINT.exists():
        return (f"NOT USED BY THIS VIEWER - {TRAV_CHECKPOINT.name} exists (separate manual "
                "training task) but is unvalidated and not loaded here; the live grid is the "
                "geometric fallback")
    return f"NOT TRAINED ({TRAV_CHECKPOINT.name} does not exist)"


def _is_complete_model_dir_safe(path: Any) -> bool:
    if path is None:
        return False
    p = Path(path)
    if not (p / "config.json").is_file():
        return False
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


def _build_mapping_stage(config: ViewerConfig):
    from prism.perception.mapping import MappingStage
    return MappingStage(device="cpu")


def _load_runner_module():
    """Import tools/run_prism_ipwebcam.py by path so its tested helpers are reused."""
    spec = importlib.util.spec_from_file_location(
        "prism_ipwebcam_runner_for_viewer",
        PROJECT_ROOT / "tools" / "run_prism_ipwebcam.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_colmap_probe_module():
    spec = importlib.util.spec_from_file_location(
        "prism_colmap_sparse_probe_for_viewer", PROJECT_ROOT / "tools" / "colmap_sparse_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()
DEFAULT_URL = RUNNER.DEFAULT_URL
URL_ENV_VAR = RUNNER.URL_ENV_VAR

LIMITS = (
    "Advisory-only workbench: no command is sent to a drone, flight controller, motor or "
    "vehicle, and prism/runtime.py, nav/planner.py, nav/supervisor.py and ros2/ are not imported.",
    "Detections are 2-D COCO image observations, not hazards, depth, clearance or traversability; "
    "an empty detection list is not evidence that a path is safe.",
    "Camera motion is monocular ORB/fundamental-matrix geometry. Without a validated calibration "
    "artifact it is tracking-only, and any relative motion has arbitrary scale.",
    "Depth is Depth Anything V2-Small RELATIVE inverse depth. The metre column comes from the "
    "assumed 0.12 m camera height (prism/perception/geometry.py), never from a calibration, "
    "so it is an assumed-scale estimate, not a measurement.",
    "Terrain in this viewer is GEOMETRIC / ASSUMED SCALE / UNVALIDATED: a BEV grid built from "
    "assumed-scale depth plus pretrained ADE20K semantics. No learned traversability model runs "
    "here and the metric terrain grid stays blocked; unknown cells stay UNKNOWN.",
    "The depth-based point cloud is a single-frame assumed-scale visualization, not a "
    "reconstruction; the offline COLMAP workflow remains the only 3-D geometry pipeline.",
    "3-D reconstruction is an offline COLMAP pass over a captured sequence, labelled "
    "METRIC_SCALE_UNVERIFIED unless a reviewed metric reference and calibration exist.",
    "A live MJPEG or webcam frame has no hardware timestamp: results are stamped with the frame "
    "they were computed from, and that frame age is always displayed.",
)

DEPTH_BLOCK = {
    "capability": "Depth estimation",
    "state": "BLOCKED_MISSING_INPUT",
    "drawn": False,
    "missing": (
        "Model weights: Depth Anything V2-Small is not installed under "
        "model_assets/pretrained/depth_anything_v2_small_hf. Install locally with: "
        "python tools/download_prism_pretrained.py --only depth "
        "(the viewer itself never downloads weights).",
        "Calibration: prism/calibration.py states 'The repository does not ship a calibration "
        "asset'; CFG.cam hfov 92 deg and height 0.12 m are documented assumptions, not results.",
    ),
    "note": (
        "Even with those weights the metric column of DepthResult comes from the assumed camera "
        "height (prism/perception/geometry.py), so depth is displayed as relative inverse depth "
        "with an assumed-scale metre estimate, never as calibrated measurement."),
}
TERRAIN_BLOCK = {
    "capability": "Terrain / traversability",
    "state": "BLOCKED_MISSING_INPUT",
    "drawn": False,
    "missing": (
        "prism/integrations/terrain.py:24 still returns BLOCKED_MISSING_INPUT unless the geometry "
        "report says physical_scale == 'METRIC_SCALE_ALIGNED'; a metric layered grid needs a "
        "reviewed scale reference that this repository does not have.",
        "The geometric fallback grid shown while depth is available is labelled "
        "GEOMETRIC / ASSUMED SCALE / UNVALIDATED and is not the metric terrain grid.",
        "Learned traversability would need checkpoints/trav_unc_shared.pt from the separate "
        "manual training task; this viewer never loads it.",
    ),
    "note": ("Unknown cells stay UNKNOWN for display: no cell is converted into free space, and "
             "pretrained ADE20K road-scene semantics are never presented as validated "
             "off-road traversability."),
}
MAPPING_BLOCK = {
    "capability": "Active map update",
    "state": "DIAGNOSTIC_ONLY",
    "drawn": False,
    "missing": ("prism/integrations/mapping_diagnostic.py always reports active_map_updated=False; "
                "MappingStage requires metric depth or vehicle-frame points, which are blocked."),
    "note": "No occupancy or cost grid is drawn, because none is legitimately produced from these "
            "frames.",
}


class CaptureError(RuntimeError):
    """The camera source could not be opened."""


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    """One workbench session; every input is explicit and validated."""

    source: str = SOURCE_WEBCAM
    camera_index: int = 0
    url: str = DEFAULT_URL
    url_source: str = "default"
    url_file: Path | None = None
    output: Path | None = None
    output_root: Path = PROJECT_ROOT / "output" / "live_viewer_runs"
    max_frames: int | None = None
    buffer_frames: int = 2
    display_fps: float = 30.0
    detector_interval: int = 1
    motion_interval: int = 1
    with_depth: bool = True
    depth_model: Path | None = DEFAULT_DEPTH_DIR
    depth_interval: int = 5
    with_segmentation: bool = True
    segmentation_model: Path | None = DEFAULT_SEGMENTATION_DIR
    seg_interval: int = 5
    map_interval: int = 5
    cloud_stride: int = 3
    confidence: float = 0.25
    weights: Path = PROJECT_ROOT / "model_assets" / "libreyolo" / "LibreYOLO9t.pt"
    with_yolop: bool = False
    yolop_interval: int = 15
    yolop_onnx: Path = YOLOP_ONNX
    trail_length: int = 24
    evidence_ttl_ns: int = 1_000_000_000
    open_retries: int = 2
    read_retries: int = 3
    retry_delay_s: float = 0.5
    read_retry_delay_s: float = 0.2
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    calibration_path: Path | None = None
    allow_assumed_intrinsics: bool = False
    recon_frames: int = 40
    recon_interval: int = 2
    colmap_exe: Path = DEFAULT_COLMAP_EXE
    recon_threads: int = 4
    recon_timeout_s: int = 900
    auto_reconstruct: bool = False
    open3d_viewer: bool = False
    headless: bool = False
    record_on_start: bool = False

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_WEBCAM, SOURCE_IPWEBCAM):
            raise ValueError(f"source must be '{SOURCE_WEBCAM}' or '{SOURCE_IPWEBCAM}'")
        _require_int("camera_index", self.camera_index, 0, 64)
        _require_int("buffer_frames", self.buffer_frames, 1, 30)
        _require_int("detector_interval", self.detector_interval, 1, 600)
        _require_int("motion_interval", self.motion_interval, 1, 600)
        _require_int("depth_interval", self.depth_interval, 1, 600)
        _require_int("seg_interval", self.seg_interval, 1, 600)
        _require_int("map_interval", self.map_interval, 1, 600)
        _require_int("cloud_stride", self.cloud_stride, 1, 16)
        _require_int("yolop_interval", self.yolop_interval, 1, 6000)
        _require_int("trail_length", self.trail_length, 0, 600)
        _require_int("open_retries", self.open_retries, 0, 10)
        _require_int("read_retries", self.read_retries, 0, 10)
        _require_int("open_timeout_ms", self.open_timeout_ms, 100, 60000)
        _require_int("read_timeout_ms", self.read_timeout_ms, 100, 60000)
        _require_int("recon_frames", self.recon_frames, 4, 600)
        _require_int("recon_interval", self.recon_interval, 1, 600)
        _require_int("recon_threads", self.recon_threads, 1, 32)
        _require_int("recon_timeout_s", self.recon_timeout_s, 10, 7200)
        _require_int("evidence_ttl_ns", self.evidence_ttl_ns, 1, 10 ** 15)
        _require_finite("display_fps", self.display_fps, 1.0, 120.0)
        _require_finite("confidence", self.confidence, 0.0, 1.0)
        _require_finite("retry_delay_s", self.retry_delay_s, 0.0, 30.0)
        _require_finite("read_retry_delay_s", self.read_retry_delay_s, 0.0, 30.0)
        if self.max_frames is not None:
            _require_int("max_frames", self.max_frames, 1, 100000)
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("url must be a non-empty string")
        for name in ("output", "output_root", "weights", "yolop_onnx", "colmap_exe",
                     "calibration_path", "url_file"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

    @property
    def source_label(self) -> str:
        if self.source == SOURCE_WEBCAM:
            return f"webcam:{self.camera_index}"
        return f"ipwebcam:{self.url}"


def _require_int(name: str, value: Any, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _require_finite(name: str, value: Any, low: float, high: float,
                    *, exclusive_low: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    if exclusive_low and number == low:
        raise ValueError(f"{name} must be greater than {low}")


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if hasattr(value, "value") and not isinstance(value, (dict, list, tuple)):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _json(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latency(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)

    def at(quantile: float) -> float:
        return ordered[round((len(ordered) - 1) * quantile)]

    return {"count": len(ordered), "p50_ms": at(.5), "p95_ms": at(.95), "max_ms": ordered[-1]}


def resolve_source(requested: str | None, cli_url: str | None,
                   environ: Mapping[str, str] | None = None) -> str:
    """Pick the camera source; an explicit URL or env override implies the phone stream."""
    if requested is not None:
        if requested not in (SOURCE_WEBCAM, SOURCE_IPWEBCAM):
            raise ValueError(f"--source must be '{SOURCE_WEBCAM}' or '{SOURCE_IPWEBCAM}'")
        return requested
    env = os.environ if environ is None else environ
    if cli_url or (env.get(URL_ENV_VAR) or "").strip():
        return SOURCE_IPWEBCAM
    return SOURCE_WEBCAM


def open_stream_capture(config: ViewerConfig, url: str | None = None) -> Any:
    """Open the IP Webcam MJPEG endpoint through the runner's tested helper."""
    return RUNNER._open_capture(config.url if url is None else url,
                                config.open_timeout_ms, config.read_timeout_ms)


def open_camera_capture(config: ViewerConfig) -> Any:
    """Open a local webcam index, preferring the native Windows backends."""
    backends = [("CAP_MSMF", cv2.CAP_MSMF), ("CAP_DSHOW", cv2.CAP_DSHOW),
                ("CAP_ANY", cv2.CAP_ANY)]
    errors: list[str] = []
    for name, api in backends:
        try:
            capture = cv2.VideoCapture(int(config.camera_index), api)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if capture.isOpened():
            _tune_capture(capture, config)
            return capture
        capture.release()
        errors.append(f"{name}: opened handle reports isOpened() == false")
    raise CaptureError("; ".join(errors) or "no backend could open the camera")


def _tune_capture(capture: Any, config: ViewerConfig) -> None:
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(config.read_timeout_ms))
    except Exception:
        pass


def stream_diagnostics(config: ViewerConfig, errors: Sequence[str]) -> dict:
    """Phone-stream failure report, using the runner's TCP probe and hint set."""
    diagnostics = RUNNER._diagnose_failure(config.url, list(errors))
    diagnostics["open_attempts"] = config.open_retries + 1
    return diagnostics


def camera_diagnostics(config: ViewerConfig, errors: Sequence[str]) -> dict:
    host = f"webcam index {config.camera_index}"
    return {
        "status": "HOST_UNREACHABLE",
        "host": host,
        "port": None,
        "tcp_probe": None,
        "open_errors": list(errors),
        "diagnostics": [
            f"Confirm {host} exists: try another index, for example "
            "--camera-index 1, or list devices with pycamctl/WebCameraMP first.",
            "Close other applications that hold the camera (Teams, Zoom, OBS, the Windows "
            "Camera app); Windows allows one exclusive reader per device.",
            "On Windows check Settings > Privacy & security > Camera: desktop apps must be "
            "allowed to access the camera.",
            "If the driver exposes the device only through DirectShow, retry with "
            "--camera-backend dshow.",
            f"If the phone stream is the intended source, use --source {SOURCE_IPWEBCAM} "
            f"or set {URL_ENV_VAR}.",
        ],
    }


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    capture_at_ns: int
    bgr: Any


class LatestFrameBuffer:
    """Bounded hand-off that always yields the newest frame and counts the rest as stale."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque[CapturedFrame] = deque(maxlen=self.capacity)
        self._condition = threading.Condition()
        self.closed = False
        self.published = 0
        self.consumed = 0
        self.stale_dropped = 0

    def publish(self, frame: CapturedFrame) -> None:
        with self._condition:
            if len(self._items) == self._items.maxlen:
                self.stale_dropped += 1
            self._items.append(frame)
            self.published += 1
            self._condition.notify_all()

    def consume(self, timeout_s: float) -> CapturedFrame | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while not self._items and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._items:
                return None
            while len(self._items) > 1:
                self._items.popleft()
                self.stale_dropped += 1
            self.consumed += 1
            return self._items.popleft()

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()


def load_detector(weights: Path, confidence: float, source: str):
    """Local checkpoint only; this never downloads weights."""
    return LibreYOLODetectionProvider.from_checkpoint(
        weights, source=source, hypothesis_id=HYPOTHESIS_DETECTION,
        relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000,
        confidence_threshold=confidence)


def load_yolop(onnx_path: Path, source: str):
    return YOLOPDetectionProvider.from_onnx(
        onnx_path, expected_sha256=YOLOP_320_SHA256, source=source,
        hypothesis_id="yolop-detection-evidence", relation=EvidenceRelation.SUPPORTS,
        expires_after_ns=1_000_000_000)


def build_calibration(config: ViewerConfig) -> CameraCalibration:
    """Missing calibration is the default; an artifact is only used when it validates."""
    if config.calibration_path is not None:
        try:
            return CameraCalibration.from_artifact(config.calibration_path)
        except (CalibrationError, OSError, ValueError) as error:
            return CameraCalibration.invalid(f"calibration artifact rejected: {error}")
    if config.allow_assumed_intrinsics:
        return CameraCalibration.assumed_from_config()
    return CameraCalibration.missing()


class MotionDiagnostics:
    """Incremental ORB / fundamental-matrix front end on live frames.

    Mirrors the per-frame block of ``VideoOdometryReplayHarness`` so the numbers
    shown here are the same geometry the offline harness reports; it never
    integrates a metric pose.
    """

    def __init__(self, params: OdomParams | None = None,
                 calibration: CameraCalibration | None = None) -> None:
        self.params = params or OdomParams()
        self.calibration = calibration or CameraCalibration.missing()
        self.frontend = _OrbFrontend(self.params)
        self._prev_xy = None
        self._prev_des = None
        self.last: dict[str, Any] = {}

    def reset(self) -> None:
        self._prev_xy = None
        self._prev_des = None

    def update(self, bgr: Any, frame_index: int, capture_at_ns: int) -> dict[str, Any]:
        started = perf_counter()
        try:
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError("frame is not a 3-channel image")
            proc = (bgr.copy() if bgr.shape[1] == PROC_W and bgr.shape[0] == PROC_H
                    else cv2.resize(bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA))
            xy, descriptors = self.frontend.detect(bgr, frame_index)
            previous_xy = self._prev_xy
            n_features = int(len(xy))
            n_matches = n_inliers = 0
            inlier_pairs: tuple[np.ndarray, np.ndarray] | None = None
            ratio = 0.0
            reason = ""
            status = MotionStatus.FIRST_FRAME
            translation_unit = None
            yaw_deg = sampson_px = None
            if previous_xy is None:
                reason = f"first frame ({n_features} features; no previous view)"
            elif n_features < self.params.min_matches:
                status = MotionStatus.INSUFFICIENT_FEATURES
                reason = f"only {n_features} features (< {self.params.min_matches})"
            else:
                pairs = _mutual_ratio_match(self.frontend.matcher, self._prev_des,
                                            descriptors, self.params.ratio)
                n_matches = int(len(pairs))
                if n_matches < self.params.min_matches:
                    status = MotionStatus.INSUFFICIENT_MATCHES
                    reason = f"only {n_matches} matches (< {self.params.min_matches})"
                else:
                    p0 = previous_xy[pairs[:, 0]].astype(np.float64)
                    p1 = xy[pairs[:, 1]].astype(np.float64)
                    if self.calibration.usable_intrinsics:
                        estimate = estimate_relative_motion(
                            p0, p1, self.calibration.matrix, self.params,
                            dt_s=None, distortion=self.calibration.distortion)
                        if estimate.status is MotionStatus.OK:
                            status = (MotionStatus.OK if self.calibration.calibrated
                                      else MotionStatus.UNVALIDATED_MOTION)
                        else:
                            status = estimate.status
                    else:
                        estimate = estimate_fundamental_inliers(p0, p1, self.params)
                        status = estimate.status
                    n_inliers = int(estimate.n_inliers)
                    ratio = float(estimate.inlier_ratio)
                    reason = estimate.reason
                    yaw_deg = estimate.yaw_deg
                    sampson_px = estimate.sampson_px
                    translation_unit = estimate.translation_direction
                    keep = min(n_matches, MAX_MATCH_SEGMENTS)
                    inlier_pairs = (p0[:keep], p1[:keep])
            self._prev_xy, self._prev_des = xy, descriptors
            result = {
                "frame_index": frame_index,
                "capture_at_ns": capture_at_ns,
                "status": status.value,
                "n_features": n_features,
                "n_matches": n_matches,
                "n_inliers": n_inliers,
                "inlier_ratio": ratio,
                "reason": reason,
                "yaw_deg": yaw_deg,
                "sampson_px": sampson_px,
                "translation_unit": translation_unit,
                "tracking_only": not self.calibration.usable_intrinsics,
                "calibration": self.calibration.as_dict(),
                "latency_ms": (perf_counter() - started) * 1000.0,
                "proc_bgr": proc,
                "matches": inlier_pairs,
                "geometry_note": (CALIBRATION_REQUIREMENT if self.calibration.usable_intrinsics
                                  else "no calibration artifact: fundamental-matrix inliers only"),
                "scale_note": (None if self.calibration.usable_intrinsics
                               else ARBITRARY_SCALE_NOTE),
            }
            self.last = result
            return result
        except Exception as error:
            result = {
                "frame_index": frame_index, "capture_at_ns": capture_at_ns,
                "status": MotionStatus.FRAME_ERROR.value, "n_features": 0,
                "n_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0,
                "reason": f"{type(error).__name__}: {error}", "yaw_deg": None,
                "sampson_px": None, "translation_unit": None, "tracking_only": True,
                "calibration": self.calibration.as_dict(),
                "latency_ms": (perf_counter() - started) * 1000.0, "proc_bgr": None,
                "matches": None, "geometry_note": None, "scale_note": None,
            }
            self.last = result
            return result


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


class ReconstructionManager:
    """Offline COLMAP workflow fed by live frames; never presented as real-time 3-D.

    A sequence is captured to the frames directory with the manifest schema
    ``tools/colmap_sparse_probe.extract_frames`` already publishes, the local
    COLMAP binary is driven by ``run_colmap``, and the result is inspected with
    ``prism.colmap_inspection`` plus ``sparse_geometry_report``. The geometry is
    reported with its own scale status; nothing here is rescaled or relabelled.
    """

    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    CAPTURED = "CAPTURED"
    RUNNING = "RUNNING"
    READY = "READY"
    FAILED = "FAILED"

    def __init__(self, root: Path, *, target_frames: int, stride: int, colmap_exe: Path,
                 threads: int, timeout_s: int, source_label: str, auto_run: bool = False,
                 probe: Any | None = None, open3d_viewer: bool = False) -> None:
        self.root = Path(root)
        self.directory = self.root
        self.frames_dir = self.root / "frames"
        self.target_frames = int(target_frames)
        self.stride = max(1, int(stride))
        self.colmap_exe = Path(colmap_exe)
        self.threads = int(threads)
        self.timeout_s = int(timeout_s)
        self.source_label = source_label
        self.auto_run = bool(auto_run)
        self.open3d_viewer = bool(open3d_viewer)
        self._probe = probe
        self.state = self.IDLE
        self.detail = "no reconstruction sequence requested"
        self.captured = 0
        self.dropped = 0
        self.submit_count = 0
        self.errors: list[str] = []
        self.summary: dict[str, Any] = {}
        self._take = 0
        self._queue: queue.Queue = queue.Queue(maxsize=max(2, self.target_frames))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._points: np.ndarray | None = None
        self._colors: np.ndarray | None = None
        self._preset = 0
        self._records: list[dict[str, Any]] = []
        self._manifest: dict[str, Any] = {"frames": []}
        self._viewer_result: dict[str, Any] = {"status": "NOT_REQUESTED"}
        self._capture_lock = threading.Lock()
        self._manifest_written = False
        self._finish_requested = threading.Event()

    @property
    def exe_available(self) -> bool:
        return Path(self.colmap_exe).is_file()

    def begin(self) -> bool:
        if self.state in (self.CAPTURING, self.RUNNING):
            return False
        if not self.exe_available:
            self.state = self.FAILED
            self.detail = (f"COLMAP executable not found at {self.colmap_exe}; reconstruction "
                           "needs a local COLMAP build and the project does not vendor one")
            return False
        self._take += 1
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = self.root / f"take_{self._take:03d}"
        self.frames_dir = self.directory / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.captured = self.dropped = self.submit_count = 0
        self.errors = []
        self.summary = {}
        self._records = []
        self._manifest = {"frames": []}
        self._manifest_written = False
        self._viewer_result = {"status": "NOT_REQUESTED"}
        self._finish_requested.clear()
        self._stop.clear()
        self._points = self._colors = None
        self.state = self.CAPTURING
        self.detail = f"capturing {self.target_frames} frames every {self.stride} processed frame"
        self._ensure_writer()
        return True

    def submit(self, bgr: Any, capture_at_ns: int, frame_index: int) -> str:
        if self.state != self.CAPTURING:
            return "idle"
        self.submit_count += 1
        if self.submit_count % self.stride:
            return "skipped"
        try:
            self._queue.put_nowait((bgr.copy(), int(capture_at_ns), int(frame_index)))
        except queue.Full:
            self.dropped += 1
            return "dropped"
        return "queued"

    def _ensure_writer(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._writer_loop, name="recon-writer",
                                        daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                bgr, capture_at_ns, frame_index = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._finish_requested.is_set():
                    self._finish_capture()
                    return
                if self.state != self.CAPTURING:
                    return
                continue
            name = f"frame_{frame_index:04d}.png"
            try:
                if not cv2.imwrite(str(self.frames_dir / name), bgr):
                    raise RuntimeError(f"cv2.imwrite returned false for {name}")
            except Exception as error:
                self.errors.append(f"frame write failed: {type(error).__name__}: {error}")
                continue
            self.captured += 1
            self._records.append({"file": name, "source_frame_index": int(frame_index),
                                  "capture_at_ns": int(capture_at_ns)})
            if self.captured >= self.target_frames:
                self._finish_capture()
                if self.auto_run and self.state == self.CAPTURED:
                    self.run()
                return
            if self._finish_requested.is_set() and self._queue.empty():
                self._finish_capture()
                return

    def finish_capture(self) -> dict[str, Any] | None:
        """End a capture early (the operator asked to reconstruct now).

        Frames already queued are still written, then the manifest is closed.
        """
        if self.state != self.CAPTURING:
            return self._manifest if self._manifest_written else None
        self._finish_requested.set()
        if self._thread is None or not self._thread.is_alive():
            self._finish_capture()
        else:
            self._thread.join(timeout=10.0)
        return self._manifest

    def _finish_capture(self) -> None:
        with self._capture_lock:
            already = self._manifest_written
            if not already:
                self._write_manifest()
                self._manifest_written = True
        if already:
            return
        self.state = self.CAPTURED
        self.detail = (f"{self.captured} frames captured; run the offline COLMAP pass with 'k' "
                       "(it takes minutes and is not a live model)")

    def _write_manifest(self) -> None:
        records = []
        for item in self._records:
            path = self.frames_dir / item["file"]
            if not path.is_file():
                continue
            decoded = cv2.imread(str(path))
            if decoded is None:
                continue
            height, width = decoded.shape[:2]
            records.append({
                "file": item["file"],
                "source_frame_index": item["source_frame_index"],
                "source_time_s": None,
                "capture_at_ns": item["capture_at_ns"],
                "sha256": _sha256_file(path),
                "width": int(width),
                "height": int(height),
            })
        first_ns = self._records[0]["capture_at_ns"] if self._records else 0
        last_ns = self._records[-1]["capture_at_ns"] if self._records else 0
        span_s = max(0.0, (last_ns - first_ns) / 1e9)
        manifest = {
            "source_video": f"live:{self.source_label}",
            "source_sha256": None,
            "source_width": records[0]["width"] if records else None,
            "source_height": records[0]["height"] if records else None,
            "source_frame_count": None,
            "source_fps": (len(records) / span_s) if span_s > 0 and len(records) else None,
            "capture_mode": "live frames written in read order; not a container decode",
            "timestamp_note": ("source_time_s is null: a live MJPEG or webcam frame has no "
                               "hardware timestamp; capture_at_ns is the read time"),
            "selection": f"every {self.stride}th processed frame until {self.target_frames}",
            "requested_frames": self.target_frames,
            "extracted_frames": len(records),
            "output_width": records[0]["width"] if records else None,
            "output_height": records[0]["height"] if records else None,
            "resize_interpolation": "none (frames written at their read size)",
            "frames": records,
        }
        _write_json(self.directory / "frames_manifest.json", manifest)
        self._manifest = manifest

    def run(self) -> dict[str, Any]:
        if self.state == self.READY and self.summary:
            return self.summary
        if self.state == self.CAPTURING:
            self.finish_capture()
        if self.state not in (self.CAPTURED, self.FAILED, self.READY):
            self.errors.append("reconstruction requested before a sequence was captured")
            return {"status": "NO_SEQUENCE"}
        frames = [row["file"] for row in self._manifest.get("frames", ())]
        if len(frames) < 3:
            self.state = self.FAILED
            self.detail = (f"a sequence needs at least 3 frames for COLMAP, only {len(frames)} "
                           f"were written to {self.frames_dir}")
            self.errors.append(self.detail)
            return {"status": self.detail}
        self.state = self.RUNNING
        self.detail = "running local COLMAP (feature extraction, matching, mapping) on CPU"
        try:
            probe = self._probe if self._probe is not None else _load_colmap_probe_module()
            probe.run_colmap(
                colmap_exe=self.colmap_exe, output=self.directory, frames_dir=self.frames_dir,
                threads=self.threads, timeout_s=self.timeout_s, overwrite=False,
                camera_model="SIMPLE_RADIAL", camera_params=None,
                default_focal_length_factor=None, init_image_id1=None, init_image_id2=None,
                init_image_name1=None, init_image_name2=None, free_focal_length=False)
        except Exception as error:
            self.state = self.FAILED
            self.detail = f"COLMAP run failed: {type(error).__name__}: {error}"
            self.errors.append(self.detail)
            return {"status": self.detail}
        model_dir = self.directory / "sparse" / "0"
        if not model_dir.is_dir():
            self.state = self.FAILED
            self.detail = "COLMAP produced no sparse model directory (sparse/0)"
            self.errors.append(self.detail)
            return {"status": self.detail}
        try:
            model = read_sparse_model(model_dir)
            geometry = sparse_geometry_report(
                model_dir, self.directory / "frames_manifest.json",
                [row["source_frame_index"] for row in self._manifest["frames"]],
                export_open3d_ply=self.directory / "sparse_points_native.ply")
        except Exception as error:
            self.state = self.FAILED
            self.detail = f"sparse model inspection failed: {type(error).__name__}: {error}"
            self.errors.append(self.detail)
            return {"status": self.detail}
        terrain = terrain_status(geometry_report=geometry)
        self.summary = {
            "status": "SPARSE_RECONSTRUCTION",
            "model_dir": str(model_dir),
            "point_count": len(model.points),
            "registered_images": len(model.images),
            "camera_model": model.cameras[0].model_id if model.cameras else None,
            "coordinate_frame": geometry["coordinate_frame"],
            "physical_scale": geometry["physical_scale"],
            "geometry_status": geometry["status"],
            "open3d": geometry["open3d"],
            "terrain_gate": {"status": terrain["status"], "reason": terrain.get("reason")},
            "scale_note": ("COLMAP coordinates have arbitrary monocular scale; a metre value "
                           "requires a reviewed metric reference and calibration, and "
                           "scale_alignment currently reports METRIC_SCALE_UNVERIFIED"),
        }
        _write_json(self.directory / "reconstruction_report.json", self.summary)
        self._points = (np.asarray([point.xyz for point in model.points], np.float64)
                        if model.points else None)
        self._colors = (np.asarray([point.rgb for point in model.points], np.uint8)[:, ::-1]
                        if model.points else None)
        self.state = self.READY
        self.detail = (f"{len(model.points)} sparse points, {len(model.images)} registered "
                       f"images, scale {geometry['physical_scale']}")
        if self.open3d_viewer:
            self.launch_open3d_viewer()
        return self.summary

    def presets(self) -> int:
        return 4

    def cycle_preset(self) -> None:
        self._preset = (self._preset + 1) % self.presets()

    def view(self, size: tuple[int, int]) -> np.ndarray | None:
        if self._points is None or len(self._points) == 0:
            return None
        width, height = int(size[0]), int(size[1])
        image = np.full((height, width, 3), (18, 17, 16), np.uint8)
        centre = np.median(self._points, axis=0)
        extent = float(np.percentile(np.abs(self._points - centre), 95, axis=0).max()) or 1.0
        offsets = [(1.8, -1.8, 1.8), (-1.8, -1.8, 1.8), (0.0, -2.4, 0.6), (2.2, 0.4, 1.2)]
        direction = np.asarray(offsets[self._preset % len(offsets)], np.float64)
        eye = centre + direction * extent
        camera = WorldCamera(eye, centre, (width, height), vfov_deg=46.0)
        drawn = render_points(image, camera, self._points, self._colors, bg=(18, 17, 16),
                              fog_m=extent * 6.0)
        cv2.putText(image, f"{drawn}/{len(self._points)} points drawn | COLMAP world, "
                           f"arbitrary units", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return image

    def launch_open3d_viewer(self) -> dict[str, Any]:
        ply = self.directory / "sparse_points_native.ply"
        if not ply.is_file():
            return {"status": "UNAVAILABLE", "reason": f"no exported PLY at {ply}"}
        result: dict[str, Any] = {"status": "REQUESTED", "ply": str(ply)}
        self._viewer_result = result

        def worker() -> None:
            try:
                import open3d as o3d
                cloud = o3d.io.read_point_cloud(str(ply))
                o3d.visualization.draw_geometries(
                    [cloud], window_name="P.R.I.S.M. sparse reconstruction (arbitrary units)")
                self._viewer_result["status"] = "CLOSED"
            except Exception as error:
                self._viewer_result["status"] = "FAILED"
                self._viewer_result["reason"] = f"{type(error).__name__}: {error}"

        threading.Thread(target=worker, name="open3d-viewer", daemon=True).start()
        return result

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state, "detail": self.detail, "directory": str(self.directory),
            "captured_frames": self.captured, "dropped_frames": self.dropped,
            "target_frames": self.target_frames, "stride": self.stride,
            "colmap_exe": str(self.colmap_exe), "colmap_exe_present": self.exe_available,
            "errors": list(self.errors), "summary": self.summary,
        }

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class LivePipeline:
    """Runs the real perception components on incoming frames and keeps their outputs.

    Every displayed result is stamped with the frame it came from, so a slow
    detector can never look like the current frame.
    """

    def __init__(self, config: ViewerConfig, *, source_label: str, detector: Any,
                 tracker: Any, motion: MotionDiagnostics, yolop: Any = None,
                 depth_stage: Any = None, segmentation_stage: Any = None,
                 mapping_stage: Any = None,
                 ledger: BeliefLedger | None = None,
                 doubt_engine: DoubtEngine | None = None) -> None:
        self.config = config
        self.source_label = source_label
        self.detector = detector
        self.tracker = tracker
        self.motion = motion
        self.yolop = yolop
        self.depth_stage = depth_stage
        self.seg_stage = segmentation_stage
        self.mapping_stage = mapping_stage
        self.ledger = ledger or BeliefLedger()
        self.doubt = doubt_engine or DoubtEngine()
        self.adapter = RuntimeAdapter()
        self.lock = threading.Lock()
        self.processed = 0
        self.frame_index = -1
        self.current: Any = None
        self.current_capture_ns = 0
        self.detection: dict[str, Any] | None = None
        self.depth_result: dict[str, Any] | None = None
        self.seg_result: dict[str, Any] | None = None
        self.bev_result: dict[str, Any] | None = None
        self._last_depth_obj: Any = None
        self._last_seg_obj: Any = None
        self.yolop_result: dict[str, Any] | None = None
        self.advisory: dict[str, Any] = {"disposition": None, "recommendations": [],
                                         "hypotheses": {}}
        self.adapter_gate: dict[str, Any] = {"attempted": False}
        self.trails: dict[int, deque] = {}
        self.detector_runs = 0
        self.detector_errors = 0
        self.depth_runs = 0
        self.depth_errors = 0
        self.seg_runs = 0
        self.seg_errors = 0
        self.map_runs = 0
        self.map_errors = 0
        self.yolop_runs = 0
        self.yolop_errors = 0
        self.evidence_errors = 0
        self.detector_latencies: list[float] = []
        self.motion_latencies: list[float] = []
        self.depth_latencies: list[float] = []
        self.seg_latencies: list[float] = []
        self.map_latencies: list[float] = []
        self.yolop_latencies: list[float] = []
        self.events: list[dict] = []
        self.detections_log: list[dict] = []
        self.tracks_log: list[dict] = []
        self.motion_log: list[dict] = []
        self.evidence_log: list[dict] = []
        self.errors: list[str] = []
        self._terrain_evidence = False
        self._detector_identity = None
        self.statuses = self._initial_statuses()

    def _initial_statuses(self) -> dict[str, dict[str, Any]]:
        calibration = self.motion.calibration
        if calibration.usable_intrinsics:
            motion_state = ("ACTIVE_CALIBRATED" if calibration.calibrated
                            else "ACTIVE_UNVALIDATED_INTRINSICS")
        else:
            motion_state = "ACTIVE_TRACKING_ONLY"
        if self.depth_stage is not None:
            depth_state = {
                "state": "ACTIVE", "drawn": True,
                "model": "Depth Anything V2-Small (relative inverse depth)",
                "source": getattr(self.depth_stage, "model_source", None),
                "semantics": ("relative depth + assumed-scale metres from the 0.12 m camera "
                              "height assumption; NOT calibrated measurement"),
                "interval": int(self.config.depth_interval),
                "missing": [],
            }
        else:
            depth_state = dict(DEPTH_BLOCK)
        if self.mapping_stage is not None:
            terrain_state = {
                "state": "EXPERIMENTAL_GEOMETRIC", "drawn": True,
                "learned_traversability": learned_traversability_note(),
                "semantics": ("geometric BEV grid from assumed-scale depth; "
                              "GEOMETRIC / ASSUMED SCALE / UNVALIDATED"),
                "metric_grid_gate": "BLOCKED_MISSING_INPUT (unchanged)",
                "interval": int(self.config.map_interval),
                "missing": [item for item in TERRAIN_BLOCK["missing"]
                            if "BLOCKED_MISSING_INPUT" in item or "trav_unc_shared" in item],
            }
            mapping_state = {
                "state": "EXPERIMENTAL", "drawn": True,
                "backend": "prism.perception.mapping.MappingStage",
                "semantics": ("BEV accumulation from assumed-scale monocular depth; no "
                              "ego-motion warp because VO is tracking-only"),
                "missing": [],
            }
        else:
            terrain_state = dict(TERRAIN_BLOCK)
            mapping_state = dict(MAPPING_BLOCK)
        if self.seg_stage is not None:
            segmentation_state = {
                "state": "ACTIVE", "drawn": True,
                "model": getattr(self.seg_stage, "model_name", None),
                "backend": getattr(self.seg_stage, "backend", None),
                "source": getattr(self.seg_stage, "model_source", None),
                "semantics": ("pretrained ADE20K semantic segmentation mapped to prism-7; "
                              "not ground-truth traversability"),
                "interval": int(self.config.seg_interval),
                "missing": [],
            }
        else:
            segmentation_state = {
                "state": "UNAVAILABLE", "drawn": False,
                "missing": (
                    ["SegFormer-B0 (ADE20K) weights not installed under "
                     "model_assets/pretrained/segformer_b0_ade_512_512; install with: "
                     "python tools/download_prism_pretrained.py --only segmentation"]
                    if self.config.with_segmentation
                    else ["segmentation disabled by --no-segmentation for this run"]),
                "note": "pretrained road-scene semantics; never treated as trained traversability",
            }
        return {
            "detection": {
                "state": "ACTIVE" if self.detector is not None else "UNAVAILABLE",
                "drawn": self.detector is not None,
                "provider": "LibreYOLODetectionProvider" if self.detector else None,
                "model": _json(self.detector.config) if self.detector is not None else None,
                "missing": ([] if self.detector is not None else
                            [f"local checkpoint not loadable: {self.config.weights}"]),
            },
            "tracking": {
                "state": "ACTIVE", "drawn": True, "backend": getattr(self.tracker, "backend", None),
                "coordinate_frame": "image", "norfair": norfair_availability(),
                "note": "image-space IDs and short-horizon extrapolation, not world identities",
            },
            "camera_motion": {
                "state": motion_state, "drawn": True,
                "calibration": calibration.as_dict(),
                "missing": ([] if calibration.usable_intrinsics else [CALIBRATION_REQUIREMENT]),
                "scale": (None if calibration.usable_intrinsics else ARBITRARY_SCALE_NOTE),
            },
            "depth": depth_state,
            "segmentation": segmentation_state,
            "terrain": terrain_state,
            "mapping": mapping_state,
            "yolop": {
                "state": "ACTIVE" if self.yolop is not None else "NOT_REQUESTED",
                "drawn": self.yolop is not None,
                "semantics": ("BDD100K road-scene drivable/lane output statistics; not terrain, "
                              "not traversability, not accuracy"),
            },
            "evidence": {"state": "ACTIVE", "drawn": True,
                         "hypotheses": [HYPOTHESIS_DETECTION, HYPOTHESIS_TERRAIN]},
            "reconstruction": {
                "state": "OFFLINE_WORKFLOW",
                "colmap_exe": str(self.config.colmap_exe),
                "colmap_exe_present": Path(self.config.colmap_exe).is_file(),
                "missing": ([] if Path(self.config.colmap_exe).is_file()
                            else ["no local COLMAP binary at " + str(self.config.colmap_exe)]),
                "note": "not a real-time model; sparse reconstruction has arbitrary scale",
            },
            "vehicle_control": {
                "state": "NOT_IMPORTED",
                "note": "prism/runtime.py, nav/planner.py, nav/supervisor.py and ros2/ are "
                        "not imported by this workbench",
            },
        }

    def process(self, frame: CapturedFrame) -> dict[str, Any]:
        with self.lock:
            self.frame_index = frame.sequence
            self.current = frame.bgr
            self.current_capture_ns = frame.capture_at_ns
            self.processed += 1
            processed_now = self.processed
        motion = None
        if frame.sequence % self.config.motion_interval == 0:
            motion = self.motion.update(frame.bgr, frame.sequence, frame.capture_at_ns)
            self.motion_latencies.append(motion["latency_ms"])
            self.motion_log.append(_plain_motion(motion))
        detection = None
        if self.detector is not None and frame.sequence % self.config.detector_interval == 0:
            detection = self._run_detector(frame)
        # the heavyweight stages are sampled and always fire on the first frame, so a
        # short run still reports a real result instead of an empty view
        if self.depth_stage is not None and (processed_now - 1) % self.config.depth_interval == 0:
            self._run_depth(frame)
        if self.seg_stage is not None and (processed_now - 1) % self.config.seg_interval == 0:
            self._run_segmentation(frame)
        if (self.mapping_stage is not None
                and (processed_now - 1) % self.config.map_interval == 0):
            self._run_mapping(frame)
        if self.yolop is not None and frame.sequence % self.config.yolop_interval == 0:
            self._run_yolop(frame)
        depth_result = self.depth_result or {}
        seg_result = self.seg_result or {}
        bev_result = self.bev_result or {}
        event = {
            "frame_index": frame.sequence,
            "capture_at_ns": frame.capture_at_ns,
            "width": int(frame.bgr.shape[1]) if frame.bgr is not None else None,
            "height": int(frame.bgr.shape[0]) if frame.bgr is not None else None,
            "detector_status": None if detection is None else detection["status"],
            "detector_latency_ms": None if detection is None else detection["latency_ms"],
            "detection_count": None if detection is None else detection["detection_count"],
            "track_count": None if detection is None else detection["track_count"],
            "depth_status": depth_result.get("status"),
            "depth_latency_ms": depth_result.get("latency_ms"),
            "depth_valid_fraction": depth_result.get("valid_fraction"),
            "seg_status": seg_result.get("status"),
            "seg_latency_ms": seg_result.get("latency_ms"),
            "seg_drivable_fraction": seg_result.get("drivable_fraction"),
            "terrain_status": bev_result.get("status"),
            "terrain_trav_source": bev_result.get("trav_source"),
            "motion_status": None if motion is None else motion["status"],
            "n_matches": None if motion is None else motion["n_matches"],
            "n_inliers": None if motion is None else motion["n_inliers"],
            "advisory": self.advisory.get("disposition"),
        }
        self.events.append(event)
        return event

    def _run_detector(self, frame: CapturedFrame) -> dict[str, Any]:
        started = perf_counter()
        height, width = int(frame.bgr.shape[0]), int(frame.bgr.shape[1])
        try:
            supplied = self.detector(DecodedFrame(frame.sequence, frame.capture_at_ns,
                                                  width, height, frame.bgr))
            observation = supplied.observation
            payload = observation.evidence.object_detections
            detections = () if payload is None else tuple(payload.detections)
            latency_ms = (perf_counter() - started) * 1000.0
        except Exception as error:
            self.detector_errors += 1
            self.detector_latencies.append((perf_counter() - started) * 1000.0)
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"detector frame {frame.sequence}: {message}")
            self.detection = {"frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
                              "status": "PROVIDER_ERROR", "error": message, "tracks": (),
                              "detections": (), "detection_count": 0, "track_count": 0,
                              "observed": 0, "predicted": 0, "latency_ms": None}
            return self.detection
        tracks = tuple(self.tracker.update(detections, frame_index=frame.sequence,
                                           capture_at_ns=frame.capture_at_ns) or ())
        self._update_trails(tracks)
        self.detector_runs += 1
        self.detector_latencies.append(latency_ms)
        self.detection = {
            "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
            "status": "AVAILABLE", "error": None, "latency_ms": latency_ms,
            "detections": detections, "tracks": tracks,
            "detection_count": len(detections), "track_count": len(tracks),
            "observed": sum(1 for item in tracks if item.observed),
            "predicted": sum(1 for item in tracks if item.predicted),
            "observation_id": observation.observation_id,
        }
        self.detections_log.append({
            "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
            "observation_id": observation.observation_id,
            "latency_ms": latency_ms,
            "detections": [{"class_id": item.class_id, "taxonomy": item.taxonomy,
                            "score": item.score, "bbox_xyxy": list(item.bbox_xyxy)}
                           for item in detections],
            "provenance": _json(observation.provenance),
        })
        self.tracks_log.extend(_json(item) for item in tracks)
        self._record_evidence(supplied, frame.capture_at_ns)
        if self._detector_identity is None and self.detector is not None:
            self._note_runtime_gate(observation)
            self._detector_identity = observation.observation_id
        return self.detection

    def _packet_for(self, frame: CapturedFrame) -> FramePacket:
        return FramePacket(clip_id=self.source_label, idx=frame.sequence,
                           t=frame.sequence / 30.0, rgb=frame.bgr)

    def _run_depth(self, frame: CapturedFrame) -> None:
        """DepthStage on the live frame: relative depth + assumed-scale alignment."""
        started = perf_counter()
        packet = self._packet_for(frame)
        try:
            self.depth_stage(packet)
        except Exception as error:
            self.depth_errors += 1
            self.depth_latencies.append((perf_counter() - started) * 1000.0)
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"depth frame {frame.sequence}: {message}")
            self.depth_result = {"frame_index": frame.sequence,
                                 "capture_at_ns": frame.capture_at_ns,
                                 "status": "STAGE_ERROR", "error": message,
                                 "latency_ms": None}
            return
        depth = packet.depth
        valid = depth.valid if depth.valid is not None else np.isfinite(depth.depth_m)
        self.depth_runs += 1
        self.depth_latencies.append(packet.timings_ms.get("depth", 0.0))
        self._last_depth_obj = depth
        self.depth_result = {
            "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
            "status": "AVAILABLE", "error": None,
            "latency_ms": packet.timings_ms.get("depth"),
            "rel_inv": depth.rel_inv, "depth_m": depth.depth_m, "valid": valid,
            "fit": {"scale_a": float(depth.scale), "shift_b": float(depth.shift),
                    "residual": float(depth.align_residual),
                    "inliers": int(depth.align_inliers)},
            "valid_fraction": float(np.count_nonzero(valid)) / float(valid.size),
            "model": "Depth Anything V2-Small (relative inverse depth)",
        }

    def _run_segmentation(self, frame: CapturedFrame) -> None:
        """SegStage (SegFormer-B0/ADE20K teacher) -> prism-7 pretrained semantics."""
        started = perf_counter()
        try:
            result = self.seg_stage.result(frame.bgr)
            latency_ms = (perf_counter() - started) * 1000.0
        except Exception as error:
            self.seg_errors += 1
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"segmentation frame {frame.sequence}: {message}")
            self.seg_result = {"frame_index": frame.sequence,
                               "capture_at_ns": frame.capture_at_ns,
                               "status": "STAGE_ERROR", "error": message,
                               "latency_ms": None}
            return
        self.seg_runs += 1
        self.seg_latencies.append(latency_ms)
        self._last_seg_obj = result
        try:
            shares = self.seg_stage.class_shares(result.label)
            drivable = self.seg_stage.drivable_fraction(result.label)
        except Exception:
            shares, drivable = None, None
        self.seg_result = {
            "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
            "status": "AVAILABLE", "error": None, "latency_ms": latency_ms,
            "label": result.label, "prob_max": result.prob_max,
            "entropy": result.entropy,
            "class_shares": None if shares is None else
            [float(value) for value in np.asarray(shares).ravel()],
            "drivable_fraction": None if drivable is None else float(drivable),
            "prob_max_mean": float(np.mean(result.prob_max)) if result.prob_max is not None
            and result.prob_max.size else None,
            "entropy_mean": float(np.mean(result.entropy)) if result.entropy is not None
            and result.entropy.size else None,
            "model": getattr(self.seg_stage, "model_name", None),
            "pretraining": "ADE20K (150 classes) mapped to prism-7; not ground truth",
        }

    def _run_mapping(self, frame: CapturedFrame) -> None:
        """Geometric BEV grid from the latest depth (+ segmentation) result.

        No ego-motion is integrated (VO is tracking-only at best), so the grid
        accumulates in the camera frame; that limit is reported, not hidden.
        """
        if self._last_depth_obj is None:
            self.bev_result = {"frame_index": frame.sequence,
                               "capture_at_ns": frame.capture_at_ns,
                               "status": "WAITING_DEPTH",
                               "error": None, "trav_source": None}
            return
        packet = self._packet_for(frame)
        packet.depth = self._last_depth_obj
        packet.seg = self._last_seg_obj
        started = perf_counter()
        try:
            self.mapping_stage(packet)
        except Exception as error:
            self.map_errors += 1
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"mapping frame {frame.sequence}: {message}")
            self.bev_result = {"frame_index": frame.sequence,
                               "capture_at_ns": frame.capture_at_ns,
                               "status": "STAGE_ERROR", "error": message,
                               "trav_source": None}
            return
        self.map_runs += 1
        self.map_latencies.append(packet.timings_ms.get("bev", 0.0))
        bev = packet.bev
        stats = dict(cell_stats(bev))
        stats.update(self.mapping_stage.last_stats or {})
        self.bev_result = {
            "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
            "status": "AVAILABLE", "error": None,
            "latency_ms": packet.timings_ms.get("bev"),
            "bev": bev, "trav_source": getattr(self.mapping_stage, "trav_source", None),
            "stats": stats,
            "label": "GEOMETRIC / ASSUMED SCALE / UNVALIDATED",
        }

    def _run_yolop(self, frame: CapturedFrame) -> None:
        started = perf_counter()
        height, width = int(frame.bgr.shape[0]), int(frame.bgr.shape[1])
        try:
            supplied, outputs = self.yolop.run_detailed(
                DecodedFrame(frame.sequence, frame.capture_at_ns, width, height, frame.bgr))
            statistics = segmentation_summary(outputs)
            latency_ms = (perf_counter() - started) * 1000.0
            self.yolop_runs += 1
            self.yolop_latencies.append(latency_ms)
            payload = supplied.observation.evidence.object_detections
            count = 0 if payload is None else len(payload.detections)
            self.yolop_result = {
                "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
                "status": "AVAILABLE", "latency_ms": latency_ms, "detection_count": count,
                "segmentation": statistics,
            }
            self.evidence_log.append({
                "at_ns": frame.capture_at_ns, "provider": "YOLOPDetectionProvider",
                "detections": count, "segmentation": statistics,
            })
            self._record_yolop_evidence(supplied, frame.capture_at_ns)
        except Exception as error:
            self.yolop_errors += 1
            self.errors.append(f"yolop frame {frame.sequence}: {type(error).__name__}: {error}")
            self.yolop_result = {"frame_index": frame.sequence,
                                 "capture_at_ns": frame.capture_at_ns,
                                 "status": "PROVIDER_ERROR",
                                 "error": f"{type(error).__name__}: {error}"}

    def _record_yolop_evidence(self, supplied, at_ns: int) -> None:
        try:
            self.ledger.record(supplied.hypothesis_id, supplied.observation, supplied.relation,
                               expires_at_ns=max(int(supplied.expires_at_ns), at_ns + 1))
            self.statuses["evidence"]["hypotheses"] = sorted(
                set(self.statuses["evidence"]["hypotheses"]) | {supplied.hypothesis_id})
        except ValueError as error:
            self.evidence_errors += 1
            self.errors.append(f"yolop evidence: {error}")

    def _record_evidence(self, supplied, at_ns: int) -> None:
        if not self._terrain_evidence:
            self._terrain_evidence = True
            observation = Observation(
                observation_id=f"{self.source_label}:terrain-unavailable",
                source=self.source_label, captured_at_ns=at_ns,
                coordinate_frame=CoordinateFrame.LOCAL_MAP, validity=Validity.UNAVAILABLE,
                provenance={"reason": "no metric spatial input: monocular depth "
                            "carries assumed scale only and the metric terrain gate is "
                            "BLOCKED_MISSING_INPUT",
                            "detail": list(TERRAIN_BLOCK["missing"])},
                evidence=EvidencePayload())
            try:
                self.ledger.record(HYPOTHESIS_TERRAIN, observation, EvidenceRelation.SUPPORTS,
                                   expires_at_ns=at_ns + int(self.config.evidence_ttl_ns))
            except ValueError as error:
                self.evidence_errors += 1
                self.errors.append(f"terrain evidence: {error}")
        try:
            self.ledger.record(supplied.hypothesis_id, supplied.observation, supplied.relation,
                               expires_at_ns=max(int(supplied.expires_at_ns), at_ns + 1))
        except ValueError as error:
            self.evidence_errors += 1
            self.errors.append(f"detection evidence: {error}")
            return
        self.ledger.advance_to(at_ns)
        hypothesis_ids = tuple(self.statuses["evidence"]["hypotheses"])
        try:
            assessment = self.doubt.assess(self.ledger, at_ns=at_ns,
                                           hypothesis_ids=hypothesis_ids)
        except Exception as error:
            self.evidence_errors += 1
            self.errors.append(f"doubt assessment: {type(error).__name__}: {error}")
            return
        recommendations = [{"hypothesis_id": item.hypothesis_id,
                            "kind": _json(item.kind), "reason": _json(item.reason)}
                           for item in assessment.recommendations]
        kinds = {str(item["kind"]) for item in recommendations}
        disposition = "cautious_stop" if "cautious_stop" in kinds else (
            "request_evidence" if recommendations else "no_action")
        statuses = {}
        for hypothesis_id in hypothesis_ids:
            try:
                statuses[hypothesis_id] = _json(self.ledger.hypothesis_status(
                    hypothesis_id, at_ns=at_ns))
            except Exception as error:
                statuses[hypothesis_id] = f"unresolved: {type(error).__name__}"
        self.advisory = {
            "disposition": disposition, "recommendations": recommendations,
            "hypotheses": statuses, "at_ns": at_ns,
            "note": "advisory only: RecommendationKind values have no execution behaviour",
        }
        self.evidence_log.append({"at_ns": at_ns, "advisory": _json(self.advisory)})

    def _note_runtime_gate(self, observation: Observation) -> None:
        """Record what the runtime geometry adapter really says about 2-D evidence."""
        try:
            self.adapter.to_packet(observation, packet_index=0, clip_id=None)
            self.adapter_gate = {"attempted": True, "accepted": True, "reason": None}
        except Exception as error:
            self.adapter_gate = {"attempted": True, "accepted": False,
                                 "reason": f"{type(error).__name__}: {error}"}

    def _update_trails(self, tracks: Sequence[Any]) -> None:
        limit = int(self.config.trail_length)
        for track in tracks:
            if limit <= 0:
                continue
            centre = ((track.bbox_xyxy[0] + track.bbox_xyxy[2]) / 2.0,
                      (track.bbox_xyxy[1] + track.bbox_xyxy[3]) / 2.0)
            trail = self.trails.setdefault(int(track.track_id), deque(maxlen=limit))
            trail.append(centre)
        if len(self.trails) > 64:
            for track_id in list(self.trails)[:len(self.trails) - 64]:
                self.trails.pop(track_id, None)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            detection = self.detection
            frame = self.current
            frame_index, frame_ns = self.frame_index, self.current_capture_ns
            depth_result = self.depth_result
            seg_result = self.seg_result
            bev_result = self.bev_result
        return {
            "frame": frame, "frame_index": frame_index, "frame_capture_ns": frame_ns,
            "source_label": self.source_label, "processed": self.processed,
            "detection": detection, "motion": dict(self.motion.last),
            "depth_result": depth_result, "segmentation": seg_result,
            "terrain_result": bev_result,
            "yolop": self.yolop_result, "trails": {key: list(value)
                                                   for key, value in self.trails.items()},
            "advisory": dict(self.advisory), "adapter_gate": dict(self.adapter_gate),
            "statuses": self.statuses, "errors": list(self.errors[-5:]),
            "detector_now_ns": time.monotonic_ns(),
        }

    def reset_perception_state(self) -> None:
        """Clear temporal smoothing and BEV accumulation (operator M key)."""
        for stage in (self.depth_stage, self.seg_stage, self.mapping_stage):
            if stage is not None:
                try:
                    stage.reset()
                except Exception:
                    pass
        with self.lock:
            self._last_depth_obj = None
            self._last_seg_obj = None


def _plain_motion(motion: Mapping[str, Any]) -> dict[str, Any]:
    """Motion diagnostics without the image or correspondence arrays."""
    return {key: _json(value) for key, value in motion.items()
            if key not in ("proc_bgr", "matches")}


# ==========================================================================
# Native view composition
# ==========================================================================

DISPLAY_SIZE = (1280, 820)
TITLE_H = 26
RAIL_H = 108
PANEL_BG = (24, 23, 22)
PANEL_FILL = (32, 30, 28)
FG = (236, 232, 224)
DIM = (152, 148, 140)
ACCENT = (96, 205, 120)
WARN = (61, 130, 235)
BAD = (58, 58, 226)
BOX = (72, 205, 92)
BOX_PREDICTED = (61, 130, 235)
LETTERBOX = (12, 12, 12)

VIEW_LIVE = "live"
VIEW_TRACKING = "tracking"
VIEW_DEPTH = "depth"
VIEW_SEGMENTATION = "segmentation"
VIEW_TERRAIN = "terrain"
VIEW_MOTION = "motion"
VIEW_RECON = "reconstruction"
VIEW_CLOUD = "live_cloud"
VIEW_EVIDENCE = "evidence"
VIEW_OVERVIEW = "overview"

KEY_VIEWS = {ord("1"): VIEW_LIVE, ord("2"): VIEW_TRACKING, ord("3"): VIEW_DEPTH,
             ord("4"): VIEW_SEGMENTATION, ord("5"): VIEW_TERRAIN, ord("6"): VIEW_MOTION,
             ord("7"): VIEW_RECON, ord("8"): VIEW_CLOUD, ord("9"): VIEW_EVIDENCE,
             ord("0"): VIEW_OVERVIEW}

VIEW_TITLES = {
    VIEW_LIVE: "1 live frame + LibreYOLO 2-D detections (image observations, not hazards)",
    VIEW_TRACKING: "2 image-space tracking (Norfair/IoU IDs, short-horizon, no world position)",
    VIEW_DEPTH: "3 depth - Depth Anything V2-Small RELATIVE depth (metres = assumed camera height)",
    VIEW_SEGMENTATION: "4 segmentation - SegFormer-B0 ADE20K pretrained semantics -> prism-7",
    VIEW_TERRAIN: "5 terrain - geometric BEV, ASSUMED SCALE / UNVALIDATED; no learned model here",
    VIEW_MOTION: "6 camera motion: ORB + fundamental-matrix inliers (arbitrary scale)",
    VIEW_RECON: "7 offline COLMAP reconstruction over a captured sequence (not real-time 3-D)",
    VIEW_CLOUD: "8 single-frame depth point cloud (ASSUMED SCALE, not a reconstruction)",
    VIEW_EVIDENCE: "9 BeliefLedger + DoubtEngine state, runtime adapter gate, module status",
    VIEW_OVERVIEW: "0 overview of what this repository can actually run live right now",
}
KEY_HINTS = ("1 live  2 track  3 depth  4 seg  5 terrain  6 motion  7 3-D  8 cloud  "
             "9 evidence  0 overview  R record  S snapshot  M reset motion+tracks+maps  "
             "H limits  Q/Esc stop")
RECON_HINTS = "C capture sequence  K run COLMAP  V Open3D viewer  O orbit  M reset motion"


def _text(img: np.ndarray, org: Sequence[int], line: str, *, scale: float = 0.42,
          color: tuple = FG, thickness: int = 1) -> None:
    cv2.putText(img, str(line), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _fit_rect(source_size: Sequence[int], rect: Sequence[int]) -> tuple[int, int, int, int]:
    """Largest box with the source aspect ratio that fits inside ``rect``."""
    x, y, width, height = (int(v) for v in rect)
    source_w, source_h = max(1, int(source_size[0])), max(1, int(source_size[1]))
    scale = min(width / source_w, height / source_h)
    draw_w = max(1, int(round(source_w * scale)))
    draw_h = max(1, int(round(source_h * scale)))
    return x + (width - draw_w) // 2, y + (height - draw_h) // 2, draw_w, draw_h


@dataclass(frozen=True, slots=True)
class _Blit:
    x: int
    y: int
    w: int
    h: int
    sx: float
    sy: float

    def point(self, px: float, py: float) -> tuple[int, int]:
        return int(round(self.x + px * self.sx)), int(round(self.y + py * self.sy))

    def box(self, bbox: Sequence[float]) -> tuple[int, int, int, int]:
        left, top = self.point(float(bbox[0]), float(bbox[1]))
        right, bottom = self.point(float(bbox[2]), float(bbox[3]))
        return left, top, right, bottom


def _blit(img: np.ndarray, frame: np.ndarray, rect: Sequence[int],
          bg: tuple = LETTERBOX) -> _Blit:
    x, y, width, height = (int(v) for v in rect)
    if width < 4 or height < 4 or frame is None:
        return _Blit(x, y, max(1, width), max(1, height), 1.0, 1.0)
    source_h, source_w = frame.shape[0], frame.shape[1]
    ox, oy, dw, dh = _fit_rect((source_w, source_h), rect)
    resized = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    img[oy:oy + dh, ox:ox + dw] = resized
    return _Blit(ox, oy, dw, dh, dw / source_w, dh / source_h)


def _panel(img: np.ndarray, rect: Sequence[int], *, fill: tuple = PANEL_FILL) -> None:
    x, y, width, height = (int(v) for v in rect)
    cv2.rectangle(img, (x, y), (x + width - 1, y + height - 1), fill, cv2.FILLED)
    cv2.rectangle(img, (x, y), (x + width - 1, y + height - 1), (72, 68, 64), 1)


def _overlay(img: np.ndarray, rect: Sequence[int], lines: Sequence[str],
             *, color: tuple = FG, scale: float = 0.42) -> None:
    x, y = int(rect[0]) + 8, int(rect[1]) + 8
    dy = int(14 + 12 * scale)
    height = dy * (len(lines) + 1)
    cv2.rectangle(img, (x - 4, y - 12), (x + 420, y + height), (14, 14, 14), cv2.FILLED)
    for index, line in enumerate(lines):
        _text(img, (x, y + dy * index + 4), line, scale=scale, color=color)


def _pending(img: np.ndarray, rect: Sequence[int], session: "ViewerSession") -> None:
    _panel(img, rect)
    lines = _wrap(session.waiting_message(), 58)[:14]
    _text(img, (int(rect[0]) + 14, int(rect[1]) + 34), "WAITING FOR FRAMES",
          scale=0.62, color=WARN)
    for index, line in enumerate(lines):
        _text(img, (int(rect[0]) + 14, int(rect[1]) + 62 + 20 * index), line, color=DIM)
    _text(img, (int(rect[0]) + 14, int(rect[1]) + int(rect[3]) - 16),
          "no image is invented while the camera is unavailable", color=DIM, scale=0.38)


def _block_card(img: np.ndarray, rect: Sequence[int], block: Mapping[str, Any],
                extra: Sequence[str], frame: np.ndarray | None,
                session: "ViewerSession") -> None:
    """Render a blocking gate's own text plus the real frame it is NOT describing."""
    x, y, width, height = (int(v) for v in rect)
    text_w = max(280, int(width * 0.58))
    lines: list[str] = []
    for item in block.get("missing", ()):
        lines.extend(_wrap(str(item), max(40, text_w // 6)))
        lines.append("")
    lines.extend(_wrap(str(block.get("note", "")), max(40, text_w // 6)))
    for item in extra:
        lines.extend(_wrap(str(item), max(40, text_w // 6)))
    _panel(img, (x, y, text_w, height))
    _text(img, (x + 12, y + 28), f"{block.get('capability')}: {block.get('state')}",
          scale=0.58, color=BAD)
    _text(img, (x + 12, y + 48), "drawn by this view: nothing (drawn=False)", color=WARN,
          scale=0.42)
    top = y + 70
    line_h = 15
    index = 0
    while index < len(lines) and top + index * line_h < y + height - 12:
        _text(img, (x + 12, top + index * line_h + 10), lines[index], color=DIM, scale=0.36)
        index += 1
    if index < len(lines):
        _text(img, (x + 12, y + height - 12), f"+{len(lines) - index} further lines in the log",
              color=DIM, scale=0.36)
    image_rect = (x + text_w + 8, y, max(4, width - text_w - 8), height)
    if frame is not None:
        fit = _blit(img, frame, image_rect)
        cv2.rectangle(img, (fit.x, fit.y), (fit.x + fit.w, fit.y + fit.h), (60, 140, 240), 1)
        _overlay(img, image_rect, [
            f"current RGB frame {frame.shape[1]}x{frame.shape[0]} - NOT depth,",
            "NOT terrain, NOT an elevation or cost map.",
            f"captured {session.frame_age_ms():.0f} ms ago",
        ], color=WARN, scale=0.40)
    else:
        _pending(img, image_rect, session)


def _title_bar(img: np.ndarray, title: str, session: "ViewerSession") -> None:
    width = img.shape[1]
    cv2.rectangle(img, (0, 0), (width, TITLE_H), (16, 15, 15), cv2.FILLED)
    _text(img, (8, 18), title[:150], scale=0.46, color=ACCENT)
    right = f"view={session.view}  frame={session.frames_read}  {session.state_label()}"
    _text(img, (max(8, width - 7 * len(right) - 8), 18), right, scale=0.40, color=DIM)


def _status_rail(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                 view: str) -> None:
    height, width = img.shape[0], img.shape[1]
    top = height - RAIL_H
    cv2.rectangle(img, (0, top), (width, height), (16, 15, 15), cv2.FILLED)
    detection = snap.get("detection") or {}
    statuses = snap.get("statuses") or {}
    module_order = ("detection", "tracking", "camera_motion", "depth", "segmentation",
                    "terrain", "mapping", "yolop", "evidence", "reconstruction",
                    "vehicle_control")
    modules = []
    for name in module_order:
        entry = statuses.get(name) or {}
        state = str(entry.get("state", "?")).replace("BLOCKED_MISSING_INPUT", "BLOCKED")
        modules.append(f"{name}={state}")
    line_a = (f"src={session.source_display()} size={session.frame_size_text()} "
              f"proc_fps={session.processed_fps():.2f} "
              f"cap={session.frames_read}/drop={session.buffer.stale_dropped} "
              f"lat p50={_ms(session.detector_p50())} cur={_ms(detection.get('latency_ms'))} "
              f"dets={detection.get('detection_count', '-')} "
              f"tracks={detection.get('track_count', '-')} "
              f"pred={detection.get('predicted', '-')}")
    line_b = " ".join(modules)
    advisory = (snap.get("advisory") or {}).get("disposition")
    line_c = (f"advisory={advisory} (no execution behaviour) "
              f"recording={'ON' if session.recording else 'off'} "
              f"err={len(session.errors)} | {session.last_error_text()}")
    hints = KEY_HINTS if view != VIEW_RECON else (KEY_HINTS + "  " + RECON_HINTS)
    rows = [(line_a, FG), (line_b, DIM), (line_c, WARN)]
    rows.extend((line, (110, 106, 100)) for line in _wrap(hints, 170))
    for index, (line, color) in enumerate(rows):
        if line and top + 16 + 16 * index < height - 4:
            _text(img, (8, top + 16 + 16 * index), line, scale=0.38, color=color)


def _ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.0f}ms"
    except (TypeError, ValueError):
        return "-"


def _draw_boxes(img: np.ndarray, items: Sequence[Any], blit: _Blit, *,
                ids: bool) -> None:
    for item in items:
        bbox = getattr(item, "bbox_xyxy", None)
        if bbox is None:
            continue
        left, top, right, bottom = blit.box(bbox)
        predicted = bool(getattr(item, "predicted", False))
        colour = BOX_PREDICTED if predicted and ids else BOX
        cv2.rectangle(img, (left, top), (right, bottom), colour, 1)
        label = getattr(item, "class_id", "?")
        score = getattr(item, "score", None)
        track_id = getattr(item, "track_id", None)
        text = f"id{track_id} " if ids and track_id is not None else ""
        text += f"{label}{'' if score is None else f' {float(score):.2f}'}"
        if predicted:
            text += " (predicted)"
        y = max(12, top - 4)
        cv2.rectangle(img, (left, y - 12), (left + min(240, 7 * len(text) + 6), y), colour,
                      cv2.FILLED)
        _text(img, (left + 2, y - 2), text[:36], scale=0.36, color=(10, 10, 10))


def _draw_trails(img: np.ndarray, trails: Mapping[int, Sequence[Sequence[float]]],
                 blit: _Blit) -> None:
    for track_id, points in trails.items():
        if len(points) < 2:
            continue
        for left, right in zip(points[:-1], points[1:]):
            cv2.line(img, blit.point(left[0], left[1]), blit.point(right[0], right[1]),
                     (200, 160, 60), 1, cv2.LINE_AA)
        head = blit.point(points[-1][0], points[-1][1])
        cv2.circle(img, head, 3, (200, 160, 60), cv2.FILLED)


def _view_live(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
               rect: Sequence[int]) -> None:
    frame = snap.get("frame")
    if frame is None:
        _pending(img, rect, session)
        return
    blit = _blit(img, frame, rect)
    detection = snap.get("detection")
    lines = []
    if detection is None:
        detector_state = str(((snap.get("statuses") or {}).get("detection") or {}).get("state"))
        if detector_state == "ACTIVE":
            # the weights are loaded; the run cadence, not a load failure, is why there is no box yet
            lines.append("detector loaded, no frame sampled yet "
                         f"(runs every {session.config.detector_interval} processed frame(s))")
        elif detector_state == "UNAVAILABLE":
            lines.append("detector unavailable: no boxes can be drawn")
        else:
            lines.append(f"detector {detector_state.lower()}: no boxes drawn yet")
    else:
        age_frames = int(snap.get("frame_index", 0)) - int(detection.get("frame_index", 0))
        if detection.get("status") == "PROVIDER_ERROR":
            lines.append(f"detector error: {str(detection.get('error'))[:96]}")
            lines.append("no detections are invented after an error")
        else:
            _draw_boxes(img, detection.get("detections", ()), blit, ids=False)
            lines.append(f"{detection.get('detection_count')} detections in "
                         f"{_ms(detection.get('latency_ms'))}")
            if age_frames or detection.get("frame_index") != snap.get("frame_index"):
                lines.append(f"boxes are {age_frames} frame(s) old "
                             f"({session.age_ms(detection.get('capture_at_ns')):.0f} ms): "
                             "the frame shown is newer than the inference")
    lines.append("2-D COCO image boxes: not depth, clearance, hazard or traversability")
    lines.append("empty list != a clear path")
    _overlay(img, rect, lines, color=FG)


def _view_tracking(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                   rect: Sequence[int]) -> None:
    frame = snap.get("frame")
    if frame is None:
        _pending(img, rect, session)
        return
    blit = _blit(img, frame, rect)
    detection = snap.get("detection") or {}
    tracks = tuple(detection.get("tracks") or ())
    _draw_trails(img, snap.get("trails") or {}, blit)
    _draw_boxes(img, tracks, blit, ids=True)
    statuses = snap.get("statuses") or {}
    backend = (statuses.get("tracking") or {}).get("backend")
    age_frames = int(snap.get("frame_index", 0)) - int(detection.get("frame_index", -1))
    notes = (f"backend={backend} observed={detection.get('observed', 0)} "
             f"predicted={detection.get('predicted', 0)} age={age_frames} frame(s)",
             "ids are image-space continuity, not permanent identities",
             "no world/vehicle coordinates are produced by tracking")
    _overlay(img, rect, notes, color=FG)


def _colorized_relative_depth(rel_inv: np.ndarray) -> np.ndarray:
    """Near = warm, far = cool, on the network's own normalised inverse depth."""
    q = np.clip(np.asarray(rel_inv, np.float32), 0.0, 1.0)
    return cv2.applyColorMap((q * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _result_age_line(result: Mapping[str, Any], snap: Mapping[str, Any],
                     session: "ViewerSession") -> str:
    age_frames = int(snap.get("frame_index", 0)) - int(result.get("frame_index", 0))
    return (f"result age: {age_frames} frame(s) / "
            f"{session.age_ms(result.get('capture_at_ns')):.0f} ms "
            f"(from frame {result.get('frame_index')})")


def _view_depth(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                rect: Sequence[int]) -> None:
    x, y, width, height = (int(v) for v in rect)
    status = (snap.get("statuses") or {}).get("depth") or {}
    result = snap.get("depth_result")
    frame = snap.get("frame")
    if result is None or result.get("status") in ("STAGE_ERROR",):
        extra = []
        if result is not None:
            extra.append(f"last stage error: {str(result.get('error'))[:120]}")
            extra.append("no depth image is drawn after an error")
        extra.append(f"camera motion status: {(snap.get('motion') or {}).get('status')}")
        _block_card(img, rect, status if status.get("missing") else DEPTH_BLOCK, extra,
                    frame, session)
        return
    if frame is None:
        _pending(img, rect, session)
        return
    text_w = max(300, int(width * 0.34))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    half_w = (image_rect[2] - 6) // 2
    colour = _colorized_relative_depth(result["rel_inv"])
    depth_blit = _blit(img, colour, (image_rect[0], image_rect[1], half_w, image_rect[3]))
    _text(img, (depth_blit.x + 6, depth_blit.y + 16), "RELATIVE inverse depth (near=warm)",
          scale=0.36)
    fit_rect = (image_rect[0] + half_w + 6, image_rect[1], half_w, image_rect[3])
    frame_blit = _blit(img, frame, fit_rect)
    valid = result.get("valid")
    if valid is not None and valid.shape == np.asarray(frame).shape[:2]:
        contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # the mask is in frame pixels: move it onto the letterboxed copy actually drawn
        scale = np.array([frame_blit.sx, frame_blit.sy], np.float32)
        offset = np.array([frame_blit.x, frame_blit.y], np.float32)
        placed = [(np.asarray(c, np.float32) * scale + offset).astype(np.int32)
                  for c in contours]
        cv2.drawContours(img, placed, -1, (90, 200, 90), 1)
    _overlay(img, (frame_blit.x, frame_blit.y, frame_blit.w, frame_blit.h),
             ["current camera frame (valid-region outline where it matches)"], scale=0.36)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    fit = result.get("fit") or {}
    lines = [
        f"status: {result.get('status')}  model: Depth Anything V2-Small",
        _result_age_line(result, snap, session),
        f"inference latency: {_ms(result.get('latency_ms'))}",
        f"valid region: {result.get('valid_fraction', 0) * 100:.1f}% of the PROC frame",
        "",
        f"ground fit: a={fit.get('scale_a')} b={fit.get('shift_b')}",
        f"  residual={fit.get('residual')} 1/m inliers={fit.get('inliers')}",
        "",
        "WHAT THIS IS: relative inverse depth from a pretrained",
        "monocular model, normalised per frame. The metre",
        "column is aligned against the ASSUMED 0.12 m camera",
        "height (CFG.cam) - it is not an independently",
        "calibrated measurement and never a safety distance.",
    ]
    cursor = y + 24
    for line in lines:
        if cursor > y + height - 12:
            break
        _text(img, (panel_rect[0] + 10, cursor), line, scale=0.36,
              color=FG if cursor < y + 110 else DIM)
        cursor += 15
    _text(img, (panel_rect[0] + 10, y + height - 12),
          "ASSUMED SCALE / UNCALIBRATED", scale=0.44, color=BAD)


def _view_segmentation(img: np.ndarray, snap: Mapping[str, Any],
                       session: "ViewerSession", rect: Sequence[int]) -> None:
    x, y, width, height = (int(v) for v in rect)
    status = (snap.get("statuses") or {}).get("segmentation") or {}
    result = snap.get("segmentation")
    frame = snap.get("frame")
    if result is None or result.get("status") == "STAGE_ERROR":
        extra = []
        if result is not None:
            extra.append(f"last stage error: {str(result.get('error'))[:120]}")
        block = {"capability": "Semantic segmentation", "state": status.get("state", "UNAVAILABLE"),
                 "drawn": False, "missing": status.get("missing", ()),
                 "note": status.get("note", "")}
        _block_card(img, rect, block, extra, frame, session)
        return
    if frame is None:
        _pending(img, rect, session)
        return
    label = result.get("label")
    text_w = max(300, int(width * 0.34))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    half_w = (image_rect[2] - 6) // 2
    if label is not None:
        colour = colorize_terrain(np.asarray(label, np.uint8))
        label_blit = _blit(img, colour,
                           (image_rect[0], image_rect[1], half_w, image_rect[3]))
        _text(img, (label_blit.x + 6, label_blit.y + 16), "prism-7 labels (pretrained)",
              scale=0.36)
    else:
        _pending(img, (image_rect[0], image_rect[1], half_w, image_rect[3]), session)
    fit_rect = (image_rect[0] + half_w + 6, image_rect[1], half_w, image_rect[3])
    if label is None:
        overlay_img = frame
    else:
        base = np.asarray(frame, np.uint8)
        colour_map = colorize_terrain(np.asarray(label, np.uint8))
        if colour_map.shape[:2] != base.shape[:2]:
            # the label grid is the PROC map; the overlay must sit on the real frame
            colour_map = cv2.resize(colour_map, (base.shape[1], base.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        overlay_img = viz_overlay(base, colour_map, 0.55)
    overlay_blit = _blit(img, overlay_img, fit_rect)
    _overlay(img, (overlay_blit.x, overlay_blit.y, overlay_blit.w, overlay_blit.h),
             ["overlay on the live frame"], scale=0.36)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    shares = result.get("class_shares")
    share_line = "-"
    if shares:
        from prism.config import TERRAIN_CLASSES
        share_line = "  ".join(f"{name}={value * 100:.0f}%"
                               for name, value in zip(TERRAIN_CLASSES, shares))
    lines = [
        f"status: {result.get('status')}  model: {result.get('model')}",
        _result_age_line(result, snap, session),
        f"inference latency: {_ms(result.get('latency_ms'))}",
        f"mean prob_max: {result.get('prob_max_mean')}  mean entropy: "
        f"{result.get('entropy_mean')}",
        _wrap(f"class shares: {share_line}", 44),
        f"drivable fraction (drive prior): {result.get('drivable_fraction')}",
        "",
        "PRETRAINED SEMANTIC SEGMENTATION: SegFormer-B0",
        "finetuned on ADE20K and mapped to prism-7. It is",
        "not trained on P.R.I.S.M. terrain and is not",
        "ground-truth traversability; class votes feed the",
        "geometric terrain view as a prior only.",
    ]
    cursor = y + 24
    for line in lines:
        block = line if isinstance(line, list) else [str(line)]
        for item in block:
            if cursor > y + height - 12:
                break
            _text(img, (panel_rect[0] + 10, cursor), item[:120], scale=0.36,
                  color=FG if cursor < y + 110 else DIM)
            cursor += 15
        cursor += 2


def _view_terrain(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                  rect: Sequence[int]) -> None:
    x, y, width, height = (int(v) for v in rect)
    status = (snap.get("statuses") or {}).get("terrain") or {}
    result = snap.get("terrain_result")
    frame = snap.get("frame")
    geometry = (snap.get("recon") or {}).get("summary") or {}
    gate = terrain_status(geometry_report=geometry)
    session.terrain_gate_seen = gate["status"]
    if result is None or result.get("status") in ("STAGE_ERROR", "WAITING_DEPTH"):
        extra = [f"terrain_status(geometry_report=...) -> {gate['status']}",
                 f"reason: {gate['reason']}"]
        if result is not None:
            extra.append(f"geometric grid: {result.get('status')}"
                         + (f" ({result.get('error')})" if result.get("error") else ""))
        _block_card(img, rect, status if status.get("missing") else TERRAIN_BLOCK, extra,
                    frame, session)
        return
    bev = result.get("bev")
    if bev is None:
        _pending(img, rect, session)
        return
    text_w = max(320, int(width * 0.42))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    height_img = colorize_height(np.nan_to_num(bev.height, nan=0.0),
                                 invalid=~np.isfinite(bev.height))
    trav_img = colorize_trav(bev.trav)
    blended = cv2.addWeighted(height_img, 0.5, trav_img, 0.5, 0.0)
    _blit(img, blended, image_rect)
    _overlay(img, image_rect, [
        "top-down BEV: height above ground (colour) x traversability (blend);",
        "dark/grey cells are UNKNOWN - never free. ego at bottom-centre, forward = up.",
    ], scale=0.36)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    stats = result.get("stats") or {}
    lines = [
        f"status: {result.get('status')}  trav_source: {result.get('trav_source')}",
        _result_age_line(result, snap, session),
        f"grid latency: {_ms(result.get('latency_ms'))}",
        "",
        _wrap(f"LEARNED TRAVERSABILITY: {learned_traversability_note()}. The grid above is"
              " the geometric mapping stage; this viewer never loads the learned"
              " traversability checkpoint.", 46),
        "",
        f"cells: safe {stats.get('frac_safe', 0) * 100:.1f}%  "
        f"risky {stats.get('frac_risky', 0) * 100:.1f}%",
        f"      obstacle {stats.get('frac_obstacle', 0) * 100:.1f}%  "
        f"UNKNOWN {stats.get('frac_unknown', 1) * 100:.1f}%",
        f"mean live confidence: {stats.get('mean_conf')}",
        f"points fused last update: {stats.get('n_points')}",
        "",
        _wrap(f"metric terrain gate: {gate['status']} - {gate['reason']}", 46),
        "",
        "This grid is GEOMETRIC / ASSUMED SCALE / UNVALIDATED:",
        "built from assumed-scale monocular depth plus",
        "pretrained ADE20K semantics, accumulated in the",
        "camera frame (no ego-motion warp: VO is tracking-",
        "only). It is not the metric terrain grid, not",
        "learned traversability, and not a safety map.",
    ]
    cursor = y + 24
    for line in lines:
        block = line if isinstance(line, list) else [str(line)]
        for item in block:
            if cursor > y + height - 12:
                break
            colour = BAD if item.startswith("LEARNED") else (
                FG if cursor < y + 90 else DIM)
            _text(img, (panel_rect[0] + 10, cursor), item[:120], scale=0.36, color=colour)
            cursor += 15
        cursor += 2
    _text(img, (panel_rect[0] + 10, y + height - 12),
          "GEOMETRIC / ASSUMED SCALE / UNVALIDATED", scale=0.42, color=BAD)


def _live_depth_cloud(result: Mapping[str, Any], frame: np.ndarray,
                      stride: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(xyz_vehicle (N,3), colours (N,3)) from the latest depth result, or None."""
    depth_m = np.asarray(result["depth_m"], np.float32)
    valid = np.asarray(result.get("valid"), bool) if result.get("valid") is not None \
        else np.isfinite(depth_m)
    h, w = depth_m.shape
    keep = np.isfinite(depth_m) & (depth_m > 0.10) & (depth_m < 12.0) & valid
    sub = np.zeros_like(keep)
    sub[::stride, ::stride] = True
    keep &= sub
    if keep.sum() < 16:
        return None
    fit = _pgeo.GroundFit(a=1.0, b=0.0,
                          normal=np.array([0.0, 1.0, 0.0], np.float32),
                          height=CFG.cam.height_above_ground_m, ok=True)
    points_cam = _pgeo.unproject(np.nan_to_num(depth_m, nan=0.0), CFG.cam.K)
    points_veh = _pgeo.to_vehicle(points_cam, fit)
    rgb = frame if (frame.shape[1], frame.shape[0]) == (w, h) else \
        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return points_veh[keep].astype(np.float32), rgb[keep].astype(np.uint8)


def _view_cloud(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                rect: Sequence[int]) -> None:
    x, y, width, height = (int(v) for v in rect)
    result = snap.get("depth_result")
    frame = snap.get("frame")
    status = (snap.get("statuses") or {}).get("depth") or {}
    if result is None or result.get("status") != "AVAILABLE" or frame is None:
        block = {"capability": "Depth point cloud", "state": status.get("state", "UNAVAILABLE"),
                 "drawn": False,
                 "missing": status.get("missing") or
                            ("the cloud needs one completed depth inference",),
                 "note": "a single-frame cloud is a visualization of the depth estimate, "
                         "not a reconstruction"}
        _block_card(img, rect, block, [], frame, session)
        return
    cloud = _live_depth_cloud(result, frame, int(session.config.cloud_stride))
    text_w = max(300, int(width * 0.36))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    canvas = np.full((image_rect[3], image_rect[2], 3), (16, 15, 14), np.uint8)
    drawn = 0
    if cloud is not None:
        points, colours = cloud
        camera = WorldCamera(eye=(0.0, -1.4, 1.1), target=(0.0, 4.0, -0.2),
                             size=(image_rect[2], image_rect[3]), vfov_deg=46.0)
        drawn = render_points(canvas, camera, points, colours, bg=(16, 15, 14))
    _blit(img, canvas, image_rect)
    _overlay(img, image_rect, [
        f"{drawn} points drawn from frame {result.get('frame_index')}",
        "vehicle frame: x right, y forward, z up; camera at the origin looking forward",
    ], scale=0.36)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    lines = [
        f"status: {result.get('status')}  points drawn: {drawn}",
        _result_age_line(result, snap, session),
        f"source depth latency: {_ms(result.get('latency_ms'))}",
        "",
        "SINGLE-FRAME DEPTH CLOUD, ASSUMED SCALE:",
        "each point is one depth pixel unprojected with the",
        "assumed intrinsics and the assumed 0.12 m camera",
        "height. It is a visualization of the current depth",
        "estimate - not a reconstruction, not a map, and",
        "not metric geometry.",
        "",
        "The offline COLMAP workflow (view 7) remains the",
        "only 3-D reconstruction pipeline in this project.",
    ]
    cursor = y + 24
    for line in lines:
        if cursor > y + height - 12:
            break
        _text(img, (panel_rect[0] + 10, cursor), line[:120], scale=0.36,
              color=FG if cursor < y + 90 else DIM)
        cursor += 15
    _text(img, (panel_rect[0] + 10, y + height - 12),
          "ASSUMED SCALE / NOT A RECONSTRUCTION", scale=0.42, color=BAD)


def _view_motion(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                 rect: Sequence[int]) -> None:
    motion = snap.get("motion") or {}
    x, y, width, height = (int(v) for v in rect)
    proc = motion.get("proc_bgr")
    text_w = max(300, int(width * 0.42))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    if proc is None:
        _pending(img, image_rect, session)
    else:
        blit = _blit(img, proc, image_rect)
        pairs = motion.get("matches")
        if pairs is not None:
            p0, p1 = pairs
            scale_x = blit.w / float(PROC_W)
            scale_y = blit.h / float(PROC_H)
            for left, right in zip(p0, p1):
                a = (int(blit.x + left[0] * scale_x), int(blit.y + left[1] * scale_y))
                b = (int(blit.x + right[0] * scale_x), int(blit.y + right[1] * scale_y))
                cv2.line(img, a, b, (90, 200, 90), 1, cv2.LINE_AA)
        _overlay(img, image_rect, [f"processed {PROC_W}x{PROC_H} (source {session.frame_size_text()})",
                                   f"{motion.get('n_matches', 0)} matches, "
                                   f"{MAX_MATCH_SEGMENTS} segments drawn max"],
                 scale=0.36)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    calibration = motion.get("calibration") or {}
    lines = [
        f"status: {motion.get('status')}",
        f"features {motion.get('n_features')}  matches {motion.get('n_matches')}",
        f"inliers {motion.get('n_inliers')}  ratio {motion.get('inlier_ratio')}",
        f"sampson {motion.get('sampson_px')}  yaw {motion.get('yaw_deg')}",
        f"translation dir {motion.get('translation_unit')}",
        f"latency {_ms(motion.get('latency_ms'))}",
        "",
        _wrap(f"reason: {motion.get('reason')}", 46),
        _wrap(f"calibration: status={calibration.get('status')} "
              f"usable_intrinsics={calibration.get('usable_intrinsics')} "
              f"calibrated={calibration.get('calibrated')}", 46),
        _wrap(f"geometry note: {motion.get('geometry_note')}", 46),
        _wrap(f"scale: {motion.get('scale_note')}", 46),
        _wrap("no metric pose is integrated and no IMU input exists in this "
              "repository, so nothing here is VIO", 46),
    ]
    cursor = y + 24
    for line in lines:
        block = line if isinstance(line, list) else [line]
        for item in block:
            if cursor > y + height - 12:
                break
            _text(img, (panel_rect[0] + 10, cursor), item, scale=0.36,
                  color=FG if cursor < y + 110 else DIM)
            cursor += 15
        cursor += 4


def _view_recon(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                rect: Sequence[int]) -> None:
    recon = snap.get("recon") or {}
    x, y, width, height = (int(v) for v in rect)
    text_w = max(320, int(width * 0.40))
    image_rect = (x, y, max(4, width - text_w - 8), height)
    rendered = session.reconstruction.view((image_rect[2], image_rect[3])) \
        if recon.get("state") == ReconstructionManager.READY else None
    if rendered is None:
        _panel(img, image_rect)
        lines = [str(item) for item in _wrap(
            "no point cloud exists yet: this view stays empty until a captured sequence has "
            "been reconstructed offline.", 60)]
        for index, line in enumerate(lines):
            _text(img, (x + 12, y + 34 + 18 * index), line, color=DIM, scale=0.40)
        _text(img, (x + 12, y + height - 16),
              "nothing is drawn here as a placeholder reconstruction", color=WARN, scale=0.36)
    else:
        blit = _blit(img, rendered, image_rect)
        cv2.rectangle(img, (blit.x, blit.y), (blit.x + blit.w, blit.y + blit.h),
                      (72, 68, 64), 1)
    panel_rect = (x + width - text_w, y, text_w, height)
    _panel(img, panel_rect)
    summary = recon.get("summary") or {}
    lines = [
        f"state: {recon.get('state')}",
        _wrap(f"detail: {recon.get('detail')}", 46),
        _wrap(f"colmap exe present: {recon.get('colmap_exe_present')} at {recon.get('colmap_exe')}", 46),
        f"captured {recon.get('captured_frames')}/{recon.get('target_frames')} "
        f"(writer dropped {recon.get('dropped_frames')})",
        f"points {summary.get('point_count', '-')} images {summary.get('registered_images', '-')}",
        _wrap(f"coordinate frame: {summary.get('coordinate_frame')}", 46),
        _wrap(f"physical scale: {summary.get('physical_scale')} "
              f"geometry status: {summary.get('geometry_status')}", 46),
        _wrap(f"open3d: {summary.get('open3d')}", 46),
        _wrap(f"viewer: {(session.open3d_result or {}).get('status')}", 46),
        _wrap(f"terrain gate on this geometry: "
              f"{(summary.get('terrain_gate') or {}).get('status')}", 46),
        _wrap(f"scale note: {summary.get('scale_note')}", 46),
        RECON_HINTS,
    ]
    cursor = y + 24
    for line in lines:
        block = line if isinstance(line, list) else [line]
        for item in block:
            if cursor > y + height - 12:
                break
            _text(img, (panel_rect[0] + 10, cursor), item, scale=0.36,
                  color=FG if cursor < y + 60 else DIM)
            cursor += 15
        cursor += 4


def _view_evidence(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                   rect: Sequence[int]) -> None:
    _panel(img, rect)
    x, y, width, height = (int(v) for v in rect)
    advisory = snap.get("advisory") or {}
    gate = snap.get("adapter_gate") or {}
    pipeline = session.pipeline
    runs = "not started" if pipeline is None else pipeline.detector_runs
    detector_errors = "not started" if pipeline is None else pipeline.detector_errors
    evidence_errors = "not started" if pipeline is None else pipeline.evidence_errors
    records = "not started" if pipeline is None else len(pipeline.ledger.history)
    columns = (max(200, width // 3), max(200, width // 3), width - 2 * max(200, width // 3))
    blocks = [
        ("BeliefLedger / DoubtEngine", [
            f"disposition: {advisory.get('disposition')}",
            _wrap(f"note: {advisory.get('note')}", 40),
            _wrap("hypotheses:", 40),
        ] + [f"  {_wrap(f'{key}: {value}', 40)}" for key, value in
             (advisory.get("hypotheses") or {}).items()]
        + [f"  {item.get('hypothesis_id')} -> {item.get('kind')}" for item in
           (advisory.get("recommendations") or [])]),
        ("RuntimeAdapter geometry gate", [
            _wrap(f"attempted: {gate.get('attempted')}", 40),
            _wrap(f"accepted: {gate.get('accepted')}", 40),
            _wrap(f"reason: {gate.get('reason')}", 40),
            _wrap("2-D boxes are not geometry: a conservative rejection here is the "
                  "expected, correct outcome.", 40),
        ]),
        ("module status / errors", [
            _wrap(f"detector runs {runs} errors {detector_errors}", 40),
            _wrap(f"evidence errors {evidence_errors}", 40),
            _wrap(f"ledger records {records}", 40),
            _wrap(f"errors {len(session.errors)}", 40),
        ] + [f"  {line[:60]}" for line in session.errors[-3:]]),
    ]
    cursor_x = x
    for title, lines in blocks:
        _text(img, (cursor_x + 10, y + 24), title, scale=0.44, color=ACCENT)
        cursor_y = y + 44
        for line in lines:
            block = line if isinstance(line, list) else [str(line)]
            for item in block:
                if cursor_y > y + height - 12:
                    break
                _text(img, (cursor_x + 10, cursor_y), item[:120], scale=0.36, color=DIM)
                cursor_y += 15
        cursor_x += columns[0]
        if cursor_x >= x + width:
            break


def _view_overview(img: np.ndarray, snap: Mapping[str, Any], session: "ViewerSession",
                   rect: Sequence[int]) -> None:
    x, y, width, height = (int(v) for v in rect)
    gap = 6
    tile_w = (width - 3 * gap) // 4
    tile_h = (height - 2 * gap) // 2
    tiles = (
        (VIEW_LIVE, _view_live_tile),
        (VIEW_MOTION, _view_motion_tile),
        (VIEW_RECON, _view_recon_tile),
        (VIEW_DEPTH, _view_depth_tile),
        (VIEW_SEGMENTATION, _view_segmentation_tile),
        (VIEW_TERRAIN, _view_terrain_tile),
        (VIEW_CLOUD, _view_cloud_tile),
        (VIEW_EVIDENCE, _view_evidence_tile),
    )
    for index, (title, drawer) in enumerate(tiles):
        column, row = index % 4, index // 4
        tile = (x + column * (tile_w + gap), y + row * (tile_h + gap), tile_w, tile_h)
        _panel(img, tile, fill=(18, 17, 16))
        _text(img, (tile[0] + 6, tile[1] + 14), title, scale=0.38, color=ACCENT)
        drawer(img, snap, session, (tile[0], tile[1] + 18, tile[2], tile[3] - 18))


def _view_live_tile(img, snap, session, rect):
    frame = snap.get("frame")
    if frame is None:
        _text(img, (rect[0] + 6, rect[1] + 24), "no frame yet", scale=0.36, color=WARN)
        return
    blit = _blit(img, frame, rect)
    detection = snap.get("detection") or {}
    _draw_boxes(img, detection.get("tracks") or detection.get("detections") or (), blit,
                ids=True)


def _view_motion_tile(img, snap, session, rect):
    motion = snap.get("motion") or {}
    proc = motion.get("proc_bgr")
    if proc is None:
        _text(img, (rect[0] + 6, rect[1] + 24), "no motion result yet", scale=0.36, color=WARN)
        return
    blit = _blit(img, proc, rect)
    pairs = motion.get("matches")
    if pairs is not None:
        p0, p1 = pairs
        for left, right in zip(p0[::2], p1[::2]):
            cv2.line(img, blit.point(left[0], left[1]), blit.point(right[0], right[1]),
                     (90, 200, 90), 1, cv2.LINE_AA)
    _text(img, (rect[0] + 6, rect[1] + rect[3] - 6),
          f"{motion.get('status')} m={motion.get('n_matches')} i={motion.get('n_inliers')}",
          scale=0.34, color=DIM)


def _view_recon_tile(img, snap, session, rect):
    recon = snap.get("recon") or {}
    rendered = (session.reconstruction.view((rect[2], rect[3]))
                if recon.get("state") == ReconstructionManager.READY else None)
    if rendered is None:
        _text(img, (rect[0] + 6, rect[1] + 22), f"3-D: {recon.get('state')}",
              scale=0.36, color=WARN)
        for index, line in enumerate(_wrap(str(recon.get("detail")), 42)[:3]):
            _text(img, (rect[0] + 6, rect[1] + 40 + 14 * index), line, scale=0.32, color=DIM)
        return
    _blit(img, rendered, rect)


def _view_depth_tile(img, snap, session, rect):
    result = snap.get("depth_result")
    if result is not None and result.get("status") == "AVAILABLE" and \
            result.get("rel_inv") is not None:
        colour = _colorized_relative_depth(result["rel_inv"])
        placed = _blit(img, colour, rect)
        # the caption belongs on the drawn map: the tile rect is letterboxed, so text at
        # rect[1] would land on the dark canvas above the image and disappear
        _text(img, (placed.x + 6, placed.y + 14), "RELATIVE inverse depth",
              scale=0.32)
        _text(img, (rect[0] + 6, rect[1] + rect[3] - 6),
              f"lat {_ms(result.get('latency_ms'))} "
              f"valid {result.get('valid_fraction', 0) * 100:.0f}% ASSUMED SCALE",
              scale=0.30, color=WARN)
        return
    status = (snap.get("statuses") or {}).get("depth") or {}
    _text(img, (rect[0] + 6, rect[1] + 22),
          f"DEPTH: {str(status.get('state', 'UNAVAILABLE')).replace('BLOCKED_MISSING_INPUT', 'BLOCKED')}",
          scale=0.40, color=BAD)
    lines = _wrap(str(status.get("note") or "waiting for the first depth inference"), 44)[:4]
    for index, line in enumerate(lines):
        _text(img, (rect[0] + 6, rect[1] + 42 + 15 * index), line, scale=0.32, color=DIM)


def _view_segmentation_tile(img, snap, session, rect):
    result = snap.get("segmentation")
    label = (result or {}).get("label")
    if label is None:
        status = (snap.get("statuses") or {}).get("segmentation") or {}
        _text(img, (rect[0] + 6, rect[1] + 22),
              f"SEG: {str(status.get('state', 'UNAVAILABLE')).replace('BLOCKED_MISSING_INPUT', 'BLOCKED')}",
              scale=0.40, color=BAD)
        lines = _wrap(str(status.get("note") or "waiting for the first segmentation"), 44)[:4]
        for index, line in enumerate(lines):
            _text(img, (rect[0] + 6, rect[1] + 42 + 15 * index), line, scale=0.32, color=DIM)
        return
    placed = _blit(img, colorize_terrain(np.asarray(label, np.uint8)), rect)
    _text(img, (placed.x + 6, placed.y + 14), "prism-7 labels (pretrained ADE20K)",
          scale=0.32)
    _text(img, (rect[0] + 6, rect[1] + rect[3] - 6),
          f"drivable {result.get('drivable_fraction')} lat {_ms(result.get('latency_ms'))}",
          scale=0.30, color=WARN)


def _view_terrain_tile(img, snap, session, rect):
    result = snap.get("terrain_result")
    bev = (result or {}).get("bev")
    if bev is not None:
        height_map = np.asarray(bev.height, np.float32)
        colour = colorize_height(np.nan_to_num(height_map, nan=0.0),
                                 invalid=~np.isfinite(height_map))
        trav = np.asarray(bev.trav, np.uint8) if getattr(bev, "trav", None) is not None else None
        if trav is not None:
            colour = cv2.addWeighted(colour, 0.5, colorize_trav(trav), 0.5, 0)
        _blit(img, colour, rect)
        stats = (result.get("stats") or {})
        _text(img, (rect[0] + 6, rect[1] + 14), "GEOMETRIC / ASSUMED SCALE", scale=0.32)
        _text(img, (rect[0] + 6, rect[1] + rect[3] - 6),
              f"unknown {stats.get('frac_unknown')} LEARNED: NOT USED HERE", scale=0.30, color=WARN)
        return
    gate = (snap.get("terrain_gate") or {}).get("status", "BLOCKED_MISSING_INPUT")
    _text(img, (rect[0] + 6, rect[1] + 22), f"TERRAIN: {gate}", scale=0.40, color=BAD)
    lines = _wrap(str((snap.get("terrain_gate") or {}).get(
        "reason", "metric terrain is not justified by these frames")), 44)[:4]
    for index, line in enumerate(lines):
        _text(img, (rect[0] + 6, rect[1] + 42 + 15 * index), line, scale=0.32, color=DIM)


def _view_cloud_tile(img, snap, session, rect):
    result = snap.get("depth_result")
    frame = snap.get("frame")
    if result is None or result.get("status") != "AVAILABLE" or frame is None:
        _text(img, (rect[0] + 6, rect[1] + 22), "CLOUD: waiting for depth",
              scale=0.40, color=WARN)
        _text(img, (rect[0] + 6, rect[1] + 42), "single-frame assumed-scale preview",
              scale=0.32, color=DIM)
        return
    canvas = np.full((rect[3], rect[2], 3), 18, np.uint8)
    cloud = _live_depth_cloud(result, frame, session.config.cloud_stride)
    if cloud is None:
        _text(img, (rect[0] + 6, rect[1] + 22), "CLOUD: no finite depth points",
              scale=0.40, color=WARN)
        return
    xyz, colours = cloud
    cam = WorldCamera(eye=(0, -1.4, 1.1), target=(0, 4.0, -0.2), size=(rect[2], rect[3]),
                      vfov_deg=46)
    render_points(canvas, cam, xyz, colours)
    _blit(img, canvas, rect)
    _text(img, (rect[0] + 6, rect[1] + 14), f"{xyz.shape[0]} pts ASSUMED SCALE", scale=0.32)


def _view_evidence_tile(img, snap, session, rect):
    advisory = snap.get("advisory") or {}
    _text(img, (rect[0] + 6, rect[1] + 22),
          f"advisory: {advisory.get('disposition')}", scale=0.38, color=WARN)
    index = 0
    for key, value in (advisory.get("hypotheses") or {}).items():
        for line in _wrap(f"{key}: {value}", 44)[:2]:
            _text(img, (rect[0] + 6, rect[1] + 42 + 15 * index), line, scale=0.32, color=DIM)
            index += 1
    _text(img, (rect[0] + 6, rect[1] + rect[3] - 6),
          "advisory only: nothing here moves a vehicle", scale=0.32, color=DIM)


def _help_overlay(img: np.ndarray, rect: Sequence[int],
                  snap: Mapping[str, Any] | None = None) -> None:
    x, y, width, height = (int(v) for v in rect)
    box = (x + 40, y + 30, width - 80, height - 60)
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (10, 10, 10),
                  cv2.FILLED)
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), ACCENT, 1)
    _text(img, (box[0] + 16, box[1] + 30), "Limitations of this workbench (press H to close)",
          scale=0.52, color=ACCENT)
    cursor = box[1] + 56
    for item in LIMITS:
        for line in _wrap(item, max(40, box[2] // 6)):
            if cursor > box[1] + box[3] - 16:
                break
            _text(img, (box[0] + 16, cursor), line, scale=0.38, color=FG)
            cursor += 17
        cursor += 8
    statuses = (snap or {}).get("statuses") or {}
    for name in ("depth", "segmentation", "terrain", "mapping"):
        entry = statuses.get(name) or {}
        state = str(entry.get("state", "?")).replace("BLOCKED_MISSING_INPUT", "BLOCKED")
        note = str(entry.get("note", ""))[:70]
        line = f"{name}: state={state}"
        if note:
            line += f"  {note}"
        if cursor <= box[1] + box[3] - 16:
            _text(img, (box[0] + 16, cursor), line[:110], scale=0.38, color=BAD)
            cursor += 18


def compose_view(view: str, snap: Mapping[str, Any], session: "ViewerSession",
                 size: tuple[int, int] = DISPLAY_SIZE) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    image = np.full((height, width, 3), PANEL_BG, np.uint8)
    content = (8, TITLE_H + 4, width - 16, height - TITLE_H - RAIL_H - 10)
    drawers = {
        VIEW_LIVE: _view_live, VIEW_TRACKING: _view_tracking, VIEW_DEPTH: _view_depth,
        VIEW_SEGMENTATION: _view_segmentation, VIEW_TERRAIN: _view_terrain,
        VIEW_MOTION: _view_motion, VIEW_RECON: _view_recon, VIEW_CLOUD: _view_cloud,
        VIEW_EVIDENCE: _view_evidence, VIEW_OVERVIEW: _view_overview,
    }
    if view in drawers:
        if view == VIEW_OVERVIEW:
            terrain_gate = terrain_status(
                geometry_report=(snap.get("recon") or {}).get("summary") or {})
            snap = dict(snap, terrain_gate=terrain_gate)
        drawers[view](image, snap, session, content)
    else:
        _text(image, (16, 60), f"unknown view '{view}'", color=BAD)
    _title_bar(image, VIEW_TITLES.get(view, view), session)
    _status_rail(image, snap, session, view)
    if session.help:
        _help_overlay(image, (0, 0, width, height), snap)
    return image


# ==========================================================================
# Native window
# ==========================================================================

class OpenCvGui:
    """Thin wrapper so headless tests and CI never touch a window."""

    def __init__(self) -> None:
        self.available = True
        self.reason: str | None = None
        self.size = DISPLAY_SIZE
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self.size[0], self.size[1])
            cv2.moveWindow(WINDOW_NAME, 40, 40)
        except Exception as error:
            self.available = False
            self.reason = f"{type(error).__name__}: {error}"

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(WINDOW_NAME, image)

    def wait_key(self, delay_ms: int) -> int:
        return cv2.waitKey(int(delay_ms))

    def close(self) -> None:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass


class NullGui:
    """Stand-in used by --headless and by the tests: no window is ever created."""

    def __init__(self) -> None:
        self.available = False
        self.reason = "native window disabled"
        self.size = DISPLAY_SIZE
        self.frames = 0

    def show(self, image: np.ndarray) -> None:
        self.frames += 1

    def wait_key(self, delay_ms: int) -> int:
        time.sleep(min(0.05, max(0.001, delay_ms / 1000.0)))
        return -1

    def close(self) -> None:
        return None


# ==========================================================================
# Session
# ==========================================================================

class ViewerSession:
    """Threads, keys, views and audit writers for one workbench session.

    Acquisition, inference and display are separated; the frame hand-off is
    bounded and always yields the newest frame, so a slow detector slows the
    annotations and never builds a queue of stale images.
    """

    def __init__(self, config: ViewerConfig, *, output_dir: Path | None = None,
                 capture_opener: Callable[..., Any] | None = None,
                 gui: Any | None = None, detector_factory: Callable[..., Any] | None = None,
                 depth_factory: Callable[..., Any] | None = None,
                 segmentation_factory: Callable[..., Any] | None = None,
                 mapping_factory: Callable[..., Any] | None = None,
                 environ: Mapping[str, str] | None = None) -> None:
        self.config = config
        self.output = Path(output_dir) if output_dir is not None else self._new_output_dir()
        self.capture_opener = capture_opener
        self.detector_factory = detector_factory
        self.depth_factory = depth_factory
        self.segmentation_factory = segmentation_factory
        self.mapping_factory = mapping_factory
        self.gui = gui if gui is not None else (NullGui() if config.headless else OpenCvGui())
        self.environ = dict(os.environ if environ is None else environ)
        self.view = VIEW_LIVE
        self.help = False
        self.recording = bool(config.record_on_start)
        self.active_url = config.url
        self.source_label = config.source_label

        self.buffer = LatestFrameBuffer(config.buffer_frames)
        self.pipeline_lock = threading.Lock()
        self.pipeline: LivePipeline | None = None
        self.motion: MotionDiagnostics | None = None
        self.reconstruction = ReconstructionManager(
            self.output / "reconstruction", target_frames=config.recon_frames,
            stride=config.recon_interval, colmap_exe=Path(config.colmap_exe),
            threads=config.recon_threads, timeout_s=config.recon_timeout_s,
            source_label=config.source_label, auto_run=config.auto_reconstruct)
        self.open3d_result: dict[str, Any] | None = None

        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._pipeline_thread: threading.Thread | None = None
        self._recon_thread: threading.Thread | None = None
        self._capture_done = False
        self._pipeline_done = False
        self._modules_ready = False
        self._waiting_message = f"loading perception modules for {config.source_label} ..."

        self.frames_read = 0
        self.read_failures = 0
        self.reopen_count = 0
        self.consecutive_open_failures = 0
        self.consecutive_read_failures = 0
        self._capture_sequence = 0
        self._last_capture_ns = 0
        self.errors: list[str] = []
        self.capture_errors: list[str] = []
        self.status_log: list[dict[str, Any]] = []
        self.snapshot_files: list[str] = []
        self.terrain_gate_seen: str | None = None
        self.started_wall = datetime.now(timezone.utc)
        self.started_mono = time.monotonic()
        self._fps_window: list[float] = []
        self._last_processed_mono: float | None = None
        self._last_status_processed = -1
        self._writer: Any = None
        self._writer_path: Path | None = None
        self._writer_size: tuple[int, int] | None = None
        self.frame_sizes: deque = deque(maxlen=8)
        self.detection_now_ms: float | None = None
        self.paused = False
        self.paused_dropped = 0
        self.stop_reason: str | None = None
        self.report: dict[str, Any] = {}
        self.connection_diagnostics: dict[str, Any] | None = None

    # ------------------------------------------------------------ lifecycle
    def _new_output_dir(self) -> Path:
        """Unique per run; a Windows clock coarse enough to reuse a stamp is not a collision."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(self.config.output_root)
        token = hashlib.sha1(f"{stamp}:{os.getpid()}:{os.urandom(8).hex()}".encode()).hexdigest()[:8]
        candidate = root / f"{stamp}-{token}"
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = root / f"{stamp}-{token}-{suffix}"
        return candidate

    def start(self) -> int:
        self.output.mkdir(parents=True, exist_ok=True)
        _write_json(self.output / "run_config.json", self._run_config())
        self._capture_thread = threading.Thread(target=self._capture_loop,
                                                name="viewer-capture", daemon=True)
        self._capture_thread.start()
        self._pipeline_thread = threading.Thread(target=self._pipeline_loop,
                                                 name="viewer-pipeline", daemon=True)
        self._pipeline_thread.start()
        return 0

    def stop(self, reason: str) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.stop_reason = reason
        if reason not in ("max_frames reached", "capture and inference finished"):
            self.errors.append(f"stop requested: {reason}")
        self.buffer.close()

    def gui_available(self) -> bool:
        return bool(getattr(self.gui, "available", False))

    def join(self, timeout_s: float = 6.0) -> None:
        for thread in (self._capture_thread, self._pipeline_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout_s)
        if self._recon_thread is not None and self._recon_thread.is_alive():
            self._recon_thread.join(timeout=0.1)
        self.reconstruction.finish_capture()
        self.reconstruction.shutdown()

    def run(self) -> int:
        """Run to completion and finalise. Returns the process exit code."""
        self.start()
        try:
            if self.config.headless or not self.gui_available():
                code = self._loop_headless()
            else:
                code = self._loop_gui()
        except KeyboardInterrupt:
            self.stop("keyboard interrupt")
            code = 130
        finally:
            self.join()
            self._close_writer()
            self.report = self.finalize()
            self._print_summary()
        frames_read = int(self.report["capture"]["frames_read"])
        if code == 0 and frames_read == 0:
            return 3
        if code == 0 and not self.gui_available() and not self.config.headless:
            return 4
        return code

    def _print_summary(self) -> None:
        report = self.report
        detection_latency = report["latency_ms"]["detection"] or {}
        depth_latency = report["latency_ms"].get("depth") or {}
        seg_latency = report["latency_ms"].get("segmentation") or {}
        print(json.dumps({
            "output_dir": str(self.output),
            "frames_read": report["capture"]["frames_read"],
            "frames_processed": report["processing"]["frames_processed"],
            "stale_frames_dropped": report["capture"]["stale_frames_dropped"],
            "detector_p50_ms": detection_latency.get("p50_ms"),
            "depth_p50_ms": depth_latency.get("p50_ms"),
            "segmentation_p50_ms": seg_latency.get("p50_ms"),
            "depth_status": report.get("depth", {}).get("status"),
            "segmentation_status": report.get("segmentation", {}).get("status"),
            "terrain_status": report.get("terrain", {}).get("status"),
            "reconstruction_state": report["reconstruction"]["state"],
            "live_camera_connected": report["honesty"]["live_camera_connected"],
            "gui": report["gui"],
            "errors": len(report["errors"]),
        }, indent=2, sort_keys=True))

    def _loop_gui(self) -> int:
        delay = max(1, int(1000 / max(1.0, self.config.display_fps)))
        while not self._stop.is_set():
            image = compose_view(self.view, self.snapshot(), self, self.view_size())
            try:
                self.gui.show(image)
            except Exception as error:
                self.errors.append(f"window failure: {type(error).__name__}: {error}")
                return self._loop_headless()
            key = self.gui.wait_key(delay)
            if key in (-1, 255):
                continue
            if not self.handle_key(key):
                return 0
        return 0

    def _loop_headless(self) -> int:
        while not self._stop.is_set():
            snap = self.snapshot()
            limit = self.config.max_frames
            if limit is not None and int(snap["processed"]) >= int(limit):
                self.stop("max_frames reached")
                break
            if self._capture_done and self._pipeline_done:
                self.stop("capture and inference finished" if self.frames_read
                          else "capture ended without frames")
                break
            self.gui.wait_key(33)
            self._print_status(snap)
        for thread in (self._capture_thread, self._pipeline_thread):
            if thread is not None:
                thread.join(timeout=4.0)
        return 0

    def _print_status(self, snap: Mapping[str, Any]) -> None:
        processed = int(snap["processed"])
        if processed == self._last_status_processed or (processed != 1 and processed % 30):
            return
        self._last_status_processed = processed
        detection = snap.get("detection") or {}
        motion = snap.get("motion") or {}
        print(f"[headless] frames_read={self.frames_read} processed={processed} "
              f"dets={detection.get('detection_count')} motion={motion.get('status')}")

    # ------------------------------------------------------------- capture
    def view_size(self) -> tuple[int, int]:
        return tuple(int(v) for v in getattr(self.gui, "size", DISPLAY_SIZE))

    def _open_source(self) -> Any:
        if self.capture_opener is not None:
            return (self.capture_opener(self.config, self.active_url)
                    if self.config.source == SOURCE_IPWEBCAM
                    else self.capture_opener(self.config))
        if self.config.source == SOURCE_IPWEBCAM:
            self.active_url = self._resolve_url()
            return open_stream_capture(self.config, self.active_url)
        return open_camera_capture(self.config)

    def _resolve_url(self) -> str:
        """Re-resolve the stream URL so a phone whose IP changed needs no code edit."""
        if self.config.url_file is not None:
            try:
                text = Path(self.config.url_file).read_text(encoding="utf-8").strip()
                if text:
                    return RUNNER.normalize_stream_url(text.splitlines()[0])
            except OSError as error:
                self.errors.append(f"url-file unreadable, keeping previous URL: {error}")
        return self.active_url or RUNNER.resolve_url_with_source(None, self.environ)[0]

    def _capture_loop(self) -> None:
        try:
            self._capture_work()
        except Exception as error:
            self.capture_errors.append(f"capture thread failed: {type(error).__name__}: {error}")
            self.errors.append(f"capture thread failed: {type(error).__name__}: {error}")
        finally:
            self._capture_done = True
            if self.frames_read == 0 and not self.config.headless:
                # The bounded TCP probe is only run for an interactive operator: a headless or
                # test run records the raw open/read errors without touching the network.
                self.connection_diagnostics = (
                    stream_diagnostics(self.config, self.capture_errors)
                    if self.config.source == SOURCE_IPWEBCAM
                    else camera_diagnostics(self.config, self.capture_errors))
                self._waiting_message = self._diagnostic_message()
            elif self.frames_read == 0:
                detail = self.capture_errors[-1] if self.capture_errors else "no reason recorded"
                self._waiting_message = (f"no frame was ever read from "
                                         f"{self.source_display()}: {detail}")
            self.buffer.close()

    def _capture_work(self) -> None:
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._open_source()
                self.consecutive_open_failures = 0
                if self.reopen_count:
                    self.errors.append(f"reopened {self.source_label} after "
                                       f"{self.reopen_count} reopen(s)")
            except Exception as error:
                self.consecutive_open_failures += 1
                message = f"{type(error).__name__}: {error}"
                self.capture_errors.append(message)
                self.errors.append(f"capture open failed: {message}")
                if self.consecutive_open_failures > self.config.open_retries:
                    self.errors.append("giving up: open retries exhausted")
                    break
                self._stop.wait(self.config.retry_delay_s)
                continue
            self._read_loop(capture)
            try:
                capture.release()
            except Exception:
                pass
            if self._stop.is_set() or self._capture_done:
                break
            self.reopen_count += 1
            if self.reopen_count > self.config.open_retries:
                self.errors.append("giving up: read retries exhausted after reconnects")
                break
            self._stop.wait(self.config.read_retry_delay_s)

    def _diagnostic_message(self) -> str:
        diagnostics = self.connection_diagnostics or {}
        hints = diagnostics.get("diagnostics") or []
        return (f"no frame was ever read from {self.source_display()} "
                f"(status {diagnostics.get('status')}). "
                + (hints[0] if hints else "see run_report.json for the full diagnosis"))

    def _read_loop(self, capture: Any) -> None:
        while not self._stop.is_set():
            try:
                opened = capture.isOpened()
            except Exception as error:
                self.capture_errors.append(f"isOpened raised: {error}")
                return
            if not opened:
                self.capture_errors.append("capture reports isOpened() == false")
                return
            ok = False
            bgr = None
            for _ in range(self.config.read_retries + 1):
                try:
                    ok, bgr = capture.read()
                except Exception as error:
                    self.read_failures += 1
                    self.capture_errors.append(f"read raised {type(error).__name__}: {error}")
                    ok = False
                    self._stop.wait(self.config.read_retry_delay_s)
                    continue
                if ok and bgr is not None:
                    break
                self.read_failures += 1
                self._stop.wait(self.config.read_retry_delay_s)
            if not (ok and isinstance(bgr, np.ndarray) and bgr.ndim == 3 and bgr.shape[2] == 3):
                self.consecutive_read_failures += 1
                if self.consecutive_read_failures > self.config.read_retries:
                    self.capture_errors.append(
                        f"no frame decoded in {self.consecutive_read_failures} attempts")
                    return
                continue
            self.consecutive_read_failures = 0
            self.consecutive_open_failures = 0
            self.frames_read += 1
            self.frame_sizes.append((int(bgr.shape[1]), int(bgr.shape[0])))
            self._capture_sequence += 1
            stamp = time.monotonic_ns()
            if stamp <= self._last_capture_ns:
                stamp = self._last_capture_ns + 1
            self._last_capture_ns = stamp
            self.buffer.publish(CapturedFrame(self._capture_sequence, stamp, bgr))

    # ------------------------------------------------------------ pipeline
    def _prepare_modules(self) -> None:
        calibration = build_calibration(self.config)
        self.motion = MotionDiagnostics(calibration=calibration)
        detector = None
        detection_status: dict[str, Any] = {}
        try:
            if self.detector_factory is not None:
                detector = self.detector_factory(self.config.weights, self.config.confidence,
                                                 self.source_label)
            elif not self.config.weights.is_file():
                self.errors.append(f"checkpoint not found: {self.config.weights}; "
                                   "detection stays UNAVAILABLE and no boxes are drawn")
            else:
                detector = load_detector(self.config.weights, self.config.confidence,
                                         self.source_label)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"detector initialisation failed: {message}")
            detection_status = {"init_error": message}
        tracker = None
        try:
            tracker = create_image_tracker()
        except Exception as error:
            self.errors.append(f"tracker initialisation failed: {type(error).__name__}: {error}")
        yolop = None
        if self.config.with_yolop:
            try:
                yolop = load_yolop(Path(self.config.yolop_onnx), self.source_label)
            except Exception as error:
                self.errors.append(f"YOLOP initialisation failed: {type(error).__name__}: {error}")

        depth_stage = None
        segmentation_stage = None
        mapping_stage = None
        if self.config.with_depth:
            depth_stage, depth_init_error = self._build_depth_stage()
            if depth_init_error:
                self.errors.append(f"depth stage initialisation failed: {depth_init_error}")
        if self.config.with_segmentation:
            segmentation_stage, seg_init_error = self._build_segmentation_stage()
            if seg_init_error:
                self.errors.append(
                    f"segmentation stage initialisation failed: {seg_init_error}")
        if depth_stage is not None:
            try:
                mapping_stage = (self.mapping_factory or _build_mapping_stage)(self.config)
            except Exception as error:
                self.errors.append(
                    f"mapping stage initialisation failed: {type(error).__name__}: {error}")
                mapping_stage = None

        self.pipeline = LivePipeline(self.config, source_label=self.source_label,
                                     detector=detector, tracker=tracker or _NullTracker(),
                                     motion=self.motion, yolop=yolop,
                                     depth_stage=depth_stage,
                                     segmentation_stage=segmentation_stage,
                                     mapping_stage=mapping_stage)
        if detection_status:
            self.pipeline.statuses["detection"].update(detection_status)
        if depth_stage is not None and getattr(depth_stage, "model_source", None):
            self.pipeline.statuses["depth"]["model_source"] = depth_stage.model_source
        if segmentation_stage is not None and getattr(segmentation_stage,
                                                      "model_source", None):
            self.pipeline.statuses["segmentation"]["model_source"] = \
                segmentation_stage.model_source
        self._modules_ready = True
        self._waiting_message = "modules ready, waiting for frames ..."
        self.reconstruction.begin()

    def _build_depth_stage(self):
        if self.depth_factory is not None:
            try:
                return self.depth_factory(self.config), None
            except Exception as error:
                return None, f"{type(error).__name__}: {error}"
        from prism.models.pretrained_paths import local_model_dir
        local = local_model_dir(DEPTH_MODEL_ID_FOR_LOADER, env_var="PRISM_DEPTH_MODEL_PATH",
                                environ=self.environ)
        if local is None and not _is_complete_model_dir_safe(self.config.depth_model):
            return None, (
                f"local depth weights not found under {self.config.depth_model}; "
                "run: python tools/download_prism_pretrained.py --only depth")
        try:
            from prism.models.depth import DepthStage
            model_path = str(local) if local is not None else str(self.config.depth_model)
            stage = DepthStage(device="cpu", fp16=False, model_path=model_path)
            return stage, None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    def _build_segmentation_stage(self):
        if self.segmentation_factory is not None:
            try:
                return self.segmentation_factory(self.config), None
            except Exception as error:
                return None, f"{type(error).__name__}: {error}"
        from prism.models.pretrained_paths import local_model_dir
        local = local_model_dir(SEGMENTATION_MODEL_ID_FOR_LOADER,
                                env_var="PRISM_SEGMENTATION_MODEL_PATH",
                                environ=self.environ)
        if local is None and not _is_complete_model_dir_safe(self.config.segmentation_model):
            return None, (
                f"local segmentation weights not found under "
                f"{self.config.segmentation_model}; "
                "run: python tools/download_prism_pretrained.py --only segmentation")
        try:
            from prism.models.segmentation import SegStage
            model_path = (local or self.config.segmentation_model)
            stage = SegStage(device="cpu", fp16=False, teacher_path=model_path)
            return stage, None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    def _pipeline_loop(self) -> None:
        try:
            self._pipeline_work()
        except Exception as error:
            self.errors.append(f"inference thread failed: {type(error).__name__}: {error}")
        finally:
            self._pipeline_done = True
            self.buffer.close()

    def _pipeline_work(self) -> None:
        while not self._stop.is_set():
            frame = self.buffer.consume(0.25)
            if frame is None:
                if self.buffer.closed:
                    return
                continue
            if not self._modules_ready:
                self._prepare_modules()
            if self.paused:
                self.paused_dropped += 1
                continue
            try:
                event = self.pipeline.process(frame)
            except Exception as error:
                self.errors.append(f"pipeline frame {frame.sequence}: "
                                   f"{type(error).__name__}: {error}")
                continue
            self.reconstruction.submit(frame.bgr, frame.capture_at_ns, frame.sequence)
            now = time.monotonic()
            if self._last_processed_mono is not None:
                gap = now - self._last_processed_mono
                if gap > 0:
                    self._fps_window.append(1.0 / gap)
                    if len(self._fps_window) > 30:
                        self._fps_window.pop(0)
            self._last_processed_mono = now
            self.detection_now_ms = (event or {}).get("detector_latency_ms")
            self.status_log.append({
                "frame_index": frame.sequence, "capture_at_ns": frame.capture_at_ns,
                "processed_at_mono": now,
                "detection_status": (self.pipeline.detection or {}).get("status"),
                "motion_status": (self.motion.last or {}).get("status"),
                "advisory": self.pipeline.advisory.get("disposition"),
                "reconstruction": self.reconstruction.status(),
                "buffer_capacity": self.buffer.capacity,
                "stale_dropped": self.buffer.stale_dropped,
                "frames_read": self.frames_read, "read_failures": self.read_failures,
            })
            self._write_recording_frame()
            if self.config.max_frames is not None and \
                    self.pipeline.processed >= int(self.config.max_frames):
                self.stop("max_frames reached")
                return

    # ----------------------------------------------------------- recording
    def _write_recording_frame(self) -> None:
        if not self.recording or self.pipeline is None:
            return
        snap = self.snapshot()
        image = compose_view(VIEW_LIVE, snap, self, DISPLAY_SIZE)
        size = (image.shape[1], image.shape[0])
        if self._writer is None:
            path = self.output / "recording.mp4"
            fps = max(1.0, min(30.0, self.processed_fps() or 10.0))
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                self.errors.append("recording unavailable: cv2.VideoWriter could not open "
                                   f"{path} with the mp4v codec")
                self.recording = False
                return
            self._writer, self._writer_path, self._writer_size = writer, path, size
        elif self._writer_size != size:
            self._close_writer()
            self._write_recording_frame()
            return
        self._writer.write(image)

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as error:
                self.errors.append(f"recording close failed: {type(error).__name__}")
        self._writer = None

    def toggle_recording(self) -> None:
        self.recording = not self.recording
        if not self.recording:
            self._close_writer()

    def save_snapshot(self) -> dict[str, Any]:
        snap = self.snapshot()
        index = int(snap.get("frame_index", -1))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"snapshot_{stamp}_frame{index:05d}"
        image = compose_view(self.view, snap, self, self.view_size())
        image_path = self.output / "snapshots" / f"{name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(image_path), image)
        json_path = image_path.with_suffix(".json")
        _write_json(json_path, {"captured_at_wall": stamp, "view": self.view,
                                "diagnostics": self._report_section(snap)})
        record = {"png": str(image_path) if written else None, "json": str(json_path),
                  "imwrite_returned": bool(written), "view": self.view, "frame_index": index}
        if written:
            self.snapshot_files.append(str(image_path))
        else:
            self.errors.append(f"snapshot imwrite returned false for {image_path}")
        return record

    # ---------------------------------------------------------------- keys
    def handle_key(self, key: int) -> bool:
        """Return False when the key asks the session to stop."""
        if key in (ord("q"), 27, ord("Q")):
            self.stop("operator key")
            return False
        if key in KEY_VIEWS:
            self.view = KEY_VIEWS[key]
            return True
        if key in (ord("r"), ord("R")):
            self.toggle_recording()
            return True
        if key in (ord("s"), ord("S")):
            self.save_snapshot()
            return True
        if key in (ord("m"), ord("M")):
            if self.motion is not None:
                self.motion.reset()
            if self.pipeline is not None:
                self.pipeline.trails.clear()
                self.pipeline.reset_perception_state()
                self.errors.append("motion reference, trails and perception state reset "
                                   "by operator")
            return True
        if key in (ord("h"), ord("H")):
            self.help = not self.help
            return True
        if key in (ord("c"), ord("C")):
            self.reconstruction.begin()
            return True
        if key in (ord("k"), ord("K")):
            self._start_reconstruction()
            return True
        if key in (ord("v"), ord("V")):
            self.open3d_result = self.reconstruction.launch_open3d_viewer()
            return True
        if key in (ord("o"), ord("O")):
            self.reconstruction.cycle_preset()
            return True
        if key in (ord("u"), ord("U")):
            self.active_url = self._resolve_url()
            self.errors.append(f"URL re-resolved to {RUNNER.redact_url(self.active_url)}")
            self._reopen()
            return True
        if key in (ord("p"), ord("P")):
            self.toggle_pause()
            return True
        return True

    def _start_reconstruction(self) -> None:
        if self._recon_thread is not None and self._recon_thread.is_alive():
            self.errors.append("reconstruction already running")
            return
        self.reconstruction.finish_capture()

        def worker() -> None:
            try:
                self.reconstruction.run()
            except Exception as error:
                self.errors.append(f"reconstruction worker: {type(error).__name__}: {error}")

        self._recon_thread = threading.Thread(target=worker, name="recon-worker", daemon=True)
        self._recon_thread.start()

    def _reopen(self) -> None:
        self.consecutive_read_failures = 0
        self.reopen_count += 1
        if self.motion is not None:
            self.motion.reset()

    def toggle_pause(self) -> None:
        """Freeze inference while the camera keeps running; frames are dropped, not queued."""
        self.paused = not self.paused
        if self.paused:
            self.errors.append("inference paused by operator: newest frames are dropped")

    # ------------------------------------------------------------- reporting
    def snapshot(self) -> dict[str, Any]:
        if self.pipeline is None:
            return {"frame": None, "frame_index": -1, "frame_capture_ns": 0,
                    "processed": 0, "detection": None, "motion": dict(
                        self.motion.last if self.motion else {}),
                    "depth_result": None, "segmentation": None, "terrain_result": None,
                    "yolop": None, "trails": {}, "advisory": {"disposition": None},
                    "adapter_gate": {"attempted": False}, "statuses": _initial_status_stubs(self),
                    "errors": list(self.errors[-5:]), "recon": self.reconstruction.status()}
        snap = self.pipeline.snapshot()
        snap["recon"] = self.reconstruction.status()
        return snap

    def waiting_message(self) -> str:
        if self.capture_errors:
            return (f"{self.source_label}: {self.capture_errors[-1]} | "
                    f"retries={self.consecutive_open_failures} "
                    f"reopens={self.reopen_count}")
        return self._waiting_message

    def state_label(self) -> str:
        if self._stop.is_set():
            return "stopping"
        if self.paused:
            return "paused (frames dropped, not queued)"
        if not self._modules_ready:
            return "initialising"
        if self.reconstruction.state == ReconstructionManager.RUNNING:
            return "COLMAP running (offline)"
        return "running"

    def source_display(self) -> str:
        if self.config.source == SOURCE_IPWEBCAM:
            return RUNNER.redact_url(self.active_url)
        return f"webcam index {self.config.camera_index}"

    def frame_size_text(self) -> str:
        if not self.frame_sizes:
            return "unknown"
        width, height = self.frame_sizes[-1]
        return f"{width}x{height}"

    def processed_fps(self) -> float:
        if not self._fps_window:
            return 0.0
        return sum(self._fps_window) / len(self._fps_window)

    def detector_p50(self) -> float | None:
        values = _latency(self.pipeline.detector_latencies) if self.pipeline else None
        return None if values is None else values["p50_ms"]

    def frame_age_ms(self) -> float:
        if self.pipeline is None:
            return 0.0
        ns = self.pipeline.current_capture_ns
        return 0.0 if not ns else (time.monotonic_ns() - ns) / 1e6

    def age_ms(self, capture_at_ns: Any) -> float:
        if not capture_at_ns:
            return 0.0
        return max(0.0, (time.monotonic_ns() - int(capture_at_ns)) / 1e6)

    def last_error_text(self) -> str:
        if not self.errors:
            return "no errors recorded"
        return str(self.errors[-1])[:110]

    def _run_config(self) -> dict[str, Any]:
        data = _json(dataclasses.asdict(self.config))
        if isinstance(data.get("url"), str):
            data["url"] = RUNNER.redact_url(data["url"])
        data["source_resolution_note"] = (
            "explicit --source wins; otherwise --url or a non-empty "
            f"{URL_ENV_VAR} implies ipwebcam, else webcam")
        data["tool_reuse_note"] = TOOL_NOTE
        data["opencv"] = {"version": cv2.__version__,
                          "gui_available": bool(getattr(self.gui, "available", False)),
                          "gui_reason": getattr(self.gui, "reason", None)}
        data["url_redacted"] = RUNNER.redact_url(self.active_url) if isinstance(
            self.active_url, str) else None
        return data

    def _report_section(self, snap: Mapping[str, Any]) -> dict[str, Any]:
        detection = snap.get("detection") or {}
        motion = snap.get("motion") or {}
        pipeline = self.pipeline
        return {
            "session": {
                "started_wall": self.started_wall.isoformat(),
                "elapsed_s": time.monotonic() - self.started_mono,
                "source": self.source_display(), "view": self.view,
                "gui": {"available": bool(getattr(self.gui, "available", False)),
                        "reason": getattr(self.gui, "reason", None)},
            },
            "capture": {
                "frames_read": self.frames_read, "read_failures": self.read_failures,
                "stale_frames_dropped": self.buffer.stale_dropped,
                "published": self.buffer.published, "reopen_count": self.reopen_count,
                "buffer_capacity": self.buffer.capacity,
                "frame_size": self.frame_size_text(),
                "capture_errors": list(self.capture_errors[-10:]),
                "connection_diagnostics": self.connection_diagnostics,
                "paused_frames_dropped": self.paused_dropped,
                "timestamp_note": ("capture_at_ns is time.monotonic_ns() at read time; a live "
                                   "MJPEG or webcam frame carries no hardware timestamp"),
            },
            "processing": {
                "frames_processed": snap.get("processed"),
                "processing_fps": self.processed_fps(),
                "detector_interval": self.config.detector_interval,
                "motion_interval": self.config.motion_interval,
                "detector_now_ms": self.detection_now_ms,
            },
            "detection": {
                "status": detection.get("status"), "count": detection.get("detection_count"),
                "latency_ms": detection.get("latency_ms"), "error": detection.get("error"),
                "frame_index": detection.get("frame_index"),
                "age_ms": self.age_ms(detection.get("capture_at_ns")),
                "runs": None if pipeline is None else pipeline.detector_runs,
                "errors": None if pipeline is None else pipeline.detector_errors,
            },
            "tracking": {"count": detection.get("track_count"),
                         "observed": detection.get("observed"),
                         "predicted": detection.get("predicted"),
                         "trails": len(snap.get("trails") or {}),
                         "frame_index": detection.get("frame_index")},
            "motion": {key: _json(value) for key, value in motion.items()
                       if key not in ("proc_bgr", "matches")},
            "depth": self._depth_report(snap),
            "segmentation": self._segmentation_report(snap),
            "terrain": self._terrain_report(snap),
            "mapping": self._mapping_report(snap),
            "yolop": snap.get("yolop"),
            "evidence": {
                "advisory": snap.get("advisory"), "adapter_gate": snap.get("adapter_gate"),
                "ledger_records": None if pipeline is None else len(pipeline.ledger.history),
                "evidence_errors": None if pipeline is None else pipeline.evidence_errors,
            },
            "statuses": snap.get("statuses"),
            "reconstruction": snap.get("recon"),
            "latency_ms": {
                "detection": None if pipeline is None else _latency(pipeline.detector_latencies),
                "motion": None if pipeline is None else _latency(pipeline.motion_latencies),
                "depth": None if pipeline is None else _latency(pipeline.depth_latencies),
                "segmentation": None if pipeline is None else _latency(pipeline.seg_latencies),
                "mapping": None if pipeline is None else _latency(pipeline.map_latencies),
                "yolop": None if pipeline is None else _latency(pipeline.yolop_latencies),
            },
            "errors": list(snap.get("errors") or []),
            "session_errors": list(self.errors[-20:]),
            "open3d_viewer": self.open3d_result,
        }

    def _depth_report(self, snap: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self.pipeline
        result = snap.get("depth_result") or {}
        status = (snap.get("statuses") or {}).get("depth") or {}
        return {
            "status": status.get("state", "UNAVAILABLE"),
            "model": "Depth Anything V2-Small (pretrained, local)",
            "model_source": status.get("model_source"),
            "semantics": "relative inverse depth; metres are aligned to the ASSUMED "
                         "0.12 m camera height and are not calibrated measurements",
            "runs": None if pipeline is None else pipeline.depth_runs,
            "errors": None if pipeline is None else pipeline.depth_errors,
            "interval_frames": self.config.depth_interval,
            "last": {
                "frame_index": result.get("frame_index"),
                "latency_ms": result.get("latency_ms"),
                "valid_fraction": result.get("valid_fraction"),
                "fit": result.get("fit"),
                "status": result.get("status"),
            },
            "drawn": result.get("rel_inv") is not None,
        }

    def _segmentation_report(self, snap: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self.pipeline
        result = snap.get("segmentation") or {}
        status = (snap.get("statuses") or {}).get("segmentation") or {}
        return {
            "status": status.get("state", "UNAVAILABLE"),
            "backend": status.get("backend"),
            "model": status.get("model"),
            "model_source": status.get("model_source"),
            "semantics": "pretrained ADE20K semantic segmentation mapped to prism-7; "
                         "not ground-truth traversability",
            "runs": None if pipeline is None else pipeline.seg_runs,
            "errors": None if pipeline is None else pipeline.seg_errors,
            "interval_frames": self.config.seg_interval,
            "last": {
                "frame_index": result.get("frame_index"),
                "latency_ms": result.get("latency_ms"),
                "drivable_fraction": result.get("drivable_fraction"),
                "prob_max_mean": result.get("prob_max_mean"),
                "entropy_mean": result.get("entropy_mean"),
            },
            "drawn": result.get("label") is not None,
        }

    def _terrain_report(self, snap: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self.pipeline
        result = snap.get("terrain_result") or {}
        status = (snap.get("statuses") or {}).get("terrain") or {}
        return {
            "status": status.get("state", "BLOCKED_MISSING_INPUT"),
            "label": "GEOMETRIC / ASSUMED SCALE / UNVALIDATED",
            "learned_traversability": learned_traversability_note(),
            "trav_source": result.get("trav_source"),
            "runs": None if pipeline is None else pipeline.map_runs,
            "errors": None if pipeline is None else pipeline.map_errors,
            "last_stats": result.get("stats"),
            "last_frame_index": result.get("frame_index"),
            "drawn": bool(result.get("bev") is not None),
            "live_gate": terrain_status(geometry_report=(snap.get("recon") or {})
                                        .get("summary") or {}),
        }

    def _mapping_report(self, snap: Mapping[str, Any]) -> dict[str, Any]:
        status = (snap.get("statuses") or {}).get("mapping") or {}
        return {
            "status": status.get("state", "BLOCKED_MISSING_INPUT"),
            "note": status.get("note", ""),
            "drawn": (snap.get("terrain_result") or {}).get("bev") is not None,
            "semantics": "camera-frame BEV accumulation from assumed-scale monocular "
                         "depth; no ego-motion warp because VO here is tracking-only",
        }

    def finalize(self) -> dict[str, Any]:
        snap = self.snapshot()
        section = self._report_section(snap)
        pipeline = self.pipeline
        self.output.mkdir(parents=True, exist_ok=True)
        if pipeline is not None:
            _write_jsonl(self.output / "events.jsonl", pipeline.events)
            _write_jsonl(self.output / "detections.jsonl", pipeline.detections_log)
            _write_jsonl(self.output / "tracks.jsonl", pipeline.tracks_log)
            _write_jsonl(self.output / "motion.jsonl", pipeline.motion_log)
            _write_jsonl(self.output / "evidence.jsonl", pipeline.evidence_log)
        _write_jsonl(self.output / "status.jsonl", self.status_log[-4000:])
        self._save_perception_artifacts(snap)
        report = {
            "schema": "prism-live-viewer/v1",
            "started_wall": self.started_wall.isoformat(),
            "ended_wall": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - self.started_mono,
            "exit_intent": "advisory display only",
            "stop_reason": self.stop_reason,
            "config": self._run_config(),
            "capture": section["capture"],
            "processing": section["processing"],
            "detection": section["detection"],
            "tracking": section["tracking"],
            "motion": section["motion"],
            "depth": section["depth"],
            "segmentation": section["segmentation"],
            "terrain": section["terrain"],
            "mapping": section["mapping"],
            "yolop": section["yolop"],
            "evidence": section["evidence"],
            "module_statuses": section["statuses"],
            "reconstruction": section["reconstruction"],
            "latency_ms": section["latency_ms"],
            "gui": section["session"]["gui"],
            "recording": {"enabled": bool(self._writer_path is not None),
                          "path": str(self._writer_path) if self._writer_path else None,
                          "size": list(self._writer_size) if self._writer_size else None},
            "snapshots": list(self.snapshot_files),
            "open3d_viewer": self.open3d_result,
            "errors": list(self.errors),
            "capture_errors": list(self.capture_errors),
            "honesty": {
                "live_camera_connected": bool(self.frames_read > 0),
                "frames_read_from_configured_source": int(self.frames_read),
                "camera_source_label": self.source_display(),
                "advisory_only_no_vehicle_command": True,
                "runtime_planner_supervisor_ros2_imported": False,
                "depth_relative_model_operational": bool(section["depth"]["drawn"]),
                "depth_metric_reported": False,
                "depth_scale_basis": "assumed 0.12 m camera height (CFG.cam), not a calibration",
                "segmentation_is_pretrained_ade20k_not_traversability": True,
                "terrain_reported": "experimental geometric grid, assumed scale, unvalidated",
                "terrain_or_traversability_reported": False,
                "learned_traversability_trained": False,
                "cloud_is_single_frame_assumed_scale": True,
                "active_map_updated": False,
                "metric_trajectory_reported": False,
                "imu_input_exists": False,
                "reconstruction_is_real_time": False,
                "fabricated_visuals_drawn": 0,
                "unverified_assumptions": [
                    "no calibration artifact was supplied, so camera intrinsics are either "
                    "missing (tracking-only motion) or assumed (flagged as unvalidated)",
                    "IP Webcam frames are read over plain HTTP MJPEG; no per-frame hardware "
                    "timestamp or exposure metadata is available",
                    "frame timestamps are this host's clock, not the camera's",
                    "live_camera_connected only means this process decoded frames from the "
                    "configured source; it is not an independent bench verification of a "
                    "particular phone or webcam",
                ],
                "limits": list(LIMITS),
            },
        }
        _write_json(self.output / "run_report.json", report)
        (self.output / "session.log").write_text(
            "\n".join(self.errors + self.capture_errors) + "\n", encoding="utf-8")
        self._last_report = report
        return report

    def _save_perception_artifacts(self, snap: Mapping[str, Any]) -> None:
        """Persist the last real depth/segmentation/BEV arrays for offline inspection."""
        depth = snap.get("depth_result") or {}
        seg = snap.get("segmentation") or {}
        terrain = snap.get("terrain_result") or {}
        bev = terrain.get("bev")
        arrays: dict[str, Any] = {}
        for key, value in (("rel_inv", depth.get("rel_inv")),
                           ("depth_m", depth.get("depth_m")),
                           ("depth_valid", depth.get("valid")),
                           ("seg_label", seg.get("label"))):
            if value is not None:
                arrays[key] = np.asarray(value)
        if bev is not None:
            arrays["bev_height"] = np.asarray(bev.height)
            if getattr(bev, "trav", None) is not None:
                arrays["bev_trav"] = np.asarray(bev.trav)
            if getattr(bev, "conf", None) is not None:
                arrays["bev_conf"] = np.asarray(bev.conf)
        if not arrays:
            return
        try:
            np.savez_compressed(self.output / "last_perception.npz", **arrays)
        except Exception as error:
            self.errors.append(f"perception artifact save failed: {type(error).__name__}")


class _NullTracker:
    """Keeps the tracking view honest when the real tracker could not be built."""

    backend = "unavailable"

    def update(self, detections, *, frame_index: int, capture_at_ns: int):
        return ()


def _initial_status_stubs(session: ViewerSession) -> dict[str, dict[str, Any]]:
    depth = (dict(DEPTH_BLOCK) if session.config.with_depth
             else {"state": "DISABLED", "drawn": False,
                   "note": "disabled by --no-depth"})
    segmentation = ({"state": "INITIALISING", "drawn": False}
                    if session.config.with_segmentation
                    else {"state": "DISABLED", "drawn": False,
                          "note": "disabled by --no-segmentation"})
    terrain = ({"state": "INITIALISING", "drawn": False,
                "note": "geometric fallback; no learned traversability model runs here"}
               if session.config.with_depth
               else dict(TERRAIN_BLOCK))
    mapping = ({"state": "INITIALISING", "drawn": False}
               if session.config.with_depth else dict(MAPPING_BLOCK))
    return {
        "detection": {"state": "INITIALISING", "drawn": False},
        "tracking": {"state": "INITIALISING", "drawn": False},
        "camera_motion": {"state": "INITIALISING", "drawn": False},
        "depth": depth, "segmentation": segmentation, "terrain": terrain,
        "mapping": mapping,
        "yolop": {"state": "NOT_REQUESTED", "drawn": False},
        "evidence": {"state": "INITIALISING", "drawn": False},
        "reconstruction": session.reconstruction.status(),
        "vehicle_control": {"state": "NOT_IMPORTED"},
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json(row), sort_keys=True) + "\n")


# ==========================================================================
# Command line
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_prism_live_viewer",
        description="Native P.R.I.S.M. vision workbench: runs the perception capabilities this "
                    "repository can actually execute on a live camera and reports the rest as "
                    "blocked. Advisory only: nothing here commands a vehicle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=(SOURCE_WEBCAM, SOURCE_IPWEBCAM), default=None,
                        help="camera source; when omitted, --url or "
                             f"{URL_ENV_VAR} implies ipwebcam, otherwise webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index")
    parser.add_argument("--url", default=None,
                        help=f"IP Webcam MJPEG URL (default from {URL_ENV_VAR} else "
                             f"{DEFAULT_URL}; /video is never doubled)")
    parser.add_argument("--url-file", default=None,
                        help="optional file holding the current stream URL; re-read on "
                             "reconnect and on the U key so a changed phone IP needs no code edit")
    parser.add_argument("--output", default=None,
                        help="run directory; default output/live_viewer_runs/<UTC>-<token>")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many captured frames (bounded/headless runs)")
    parser.add_argument("--buffer-frames", type=int, default=2,
                        help="bounded frame hand-off capacity; obsolete frames are dropped")
    parser.add_argument("--display-fps", type=float, default=30.0, help="window refresh rate")
    parser.add_argument("--detector-interval", type=int, default=1,
                        help="run detection every N frames")
    parser.add_argument("--motion-interval", type=int, default=1,
                        help="run motion diagnostics every N frames")
    parser.add_argument("--confidence", type=float, default=0.25, help="detector threshold")
    parser.add_argument("--no-depth", action="store_true",
                        help="skip the pretrained Depth Anything V2-Small relative-depth view")
    parser.add_argument("--depth-model", default=str(DEFAULT_DEPTH_DIR),
                        help="local Depth Anything V2-Small directory (never downloaded here)")
    parser.add_argument("--depth-interval", type=int, default=5,
                        help="run depth inference every N frames (CPU)")
    parser.add_argument("--no-segmentation", action="store_true",
                        help="skip the pretrained SegFormer-B0 semantic view")
    parser.add_argument("--segmentation-model", default=str(DEFAULT_SEGMENTATION_DIR),
                        help="local SegFormer-B0 directory (never downloaded here)")
    parser.add_argument("--seg-interval", type=int, default=5,
                        help="run segmentation inference every N frames (CPU)")
    parser.add_argument("--map-interval", type=int, default=5,
                        help="rebuild the geometric BEV map every N frames (CPU)")
    parser.add_argument("--cloud-stride", type=int, default=3,
                        help="point-cloud view subsampling stride (higher = faster)")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "model_assets" / "libreyolo"
                                                 / "LibreYOLO9t.pt"),
                       help="local checkpoint; never downloaded")
    parser.add_argument("--with-yolop", action="store_true",
                        help="also run the YOLOP ONNX provider (road-scene statistics, not terrain)")
    parser.add_argument("--yolop-interval", type=int, default=15)
    parser.add_argument("--yolop-onnx", default=str(YOLOP_ONNX))
    parser.add_argument("--trail-length", type=int, default=24)
    parser.add_argument("--evidence-ttl-ns", type=int, default=1_000_000_000,
                        help="BeliefLedger expiry horizon applied to recorded evidence")
    parser.add_argument("--open-retries", type=int, default=2)
    parser.add_argument("--read-retries", type=int, default=3)
    parser.add_argument("--open-timeout-ms", type=int, default=5000)
    parser.add_argument("--read-timeout-ms", type=int, default=5000)
    parser.add_argument("--calibration", default=None,
                        help="camera calibration artifact; without it motion is tracking-only")
    parser.add_argument("--allow-assumed-intrinsics", action="store_true",
                        help="use CFG.cam assumed intrinsics, reported as unvalidated motion")
    parser.add_argument("--recon-frames", type=int, default=40,
                        help="frames captured for one offline COLMAP sequence (COLMAP needs >=3 "
                             "registered views; the probe's own gate expects more)")
    parser.add_argument("--recon-interval", type=int, default=2,
                        help="submit every N processed frames to the sequence writer")
    parser.add_argument("--colmap-exe", default=str(DEFAULT_COLMAP_EXE),
                        help="local COLMAP binary; this project does not vendor or download one")
    parser.add_argument("--recon-threads", type=int, default=4)
    parser.add_argument("--recon-timeout-s", type=int, default=900)
    parser.add_argument("--auto-reconstruct", action="store_true",
                        help="run COLMAP as soon as a sequence is captured")
    parser.add_argument("--open3d-viewer", action="store_true",
                        help="open the native Open3D point-cloud window when a model is ready")
    parser.add_argument("--headless", action="store_true",
                        help="no window: bounded run, logs and report only")
    parser.add_argument("--record", action="store_true", help="start with recording enabled")
    return parser


def config_from_args(args: argparse.Namespace) -> ViewerConfig:
    source = resolve_source(args.source, args.url)
    if source == SOURCE_IPWEBCAM:
        url, url_source = RUNNER.resolve_url_with_source(args.url, os.environ)
    elif args.url:
        url, url_source = RUNNER.normalize_stream_url(args.url), "cli"
    else:
        url, url_source = DEFAULT_URL, "unused"
    return ViewerConfig(
        source=source, camera_index=args.camera_index, url=url, url_source=url_source,
        url_file=Path(args.url_file) if args.url_file else None,
        output=Path(args.output) if args.output else None, max_frames=args.max_frames,
        buffer_frames=args.buffer_frames, display_fps=args.display_fps,
        detector_interval=args.detector_interval, motion_interval=args.motion_interval,
        confidence=args.confidence, weights=Path(args.weights), with_yolop=args.with_yolop,
        yolop_interval=args.yolop_interval, yolop_onnx=Path(args.yolop_onnx),
        trail_length=args.trail_length, evidence_ttl_ns=args.evidence_ttl_ns,
        open_retries=args.open_retries, read_retries=args.read_retries,
        open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms,
        calibration_path=Path(args.calibration) if args.calibration else None,
        allow_assumed_intrinsics=args.allow_assumed_intrinsics,
        recon_frames=args.recon_frames, recon_interval=args.recon_interval,
        colmap_exe=Path(args.colmap_exe), recon_threads=args.recon_threads,
        recon_timeout_s=args.recon_timeout_s, auto_reconstruct=args.auto_reconstruct,
        open3d_viewer=args.open3d_viewer, headless=args.headless,
        record_on_start=args.record,
        with_depth=not args.no_depth, depth_model=Path(args.depth_model),
        depth_interval=args.depth_interval,
        with_segmentation=not args.no_segmentation,
        segmentation_model=Path(args.segmentation_model),
        seg_interval=args.seg_interval, map_interval=args.map_interval,
        cloud_stride=args.cloud_stride)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        config = config_from_args(args)
    except (ValueError, TypeError) as error:
        print(f"invalid configuration: {error}", file=sys.stderr)
        return 2
    session = ViewerSession(config, output_dir=config.output)
    if not getattr(session.gui, "available", False) and not config.headless:
        print(f"native window unavailable: {getattr(session.gui, 'reason', None)}; "
              "continuing headless (exit 4 to flag degraded mode)", file=sys.stderr)
    code = session.run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
