"""Injectable LibreYOLO9t provider for typed, camera-frame detections.

The provider is deliberately a narrow boundary around an optional third-party
detector. It produces only image-space detections; scores are not observation
confidence, depth, terrain, traversability, safety, or a driving decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .belief_ledger import EvidenceRelation
from .observations import (CoordinateFrame, EvidencePayload, ImageCoordinateConvention,
                           ObjectDetection, ObjectDetectionEvidence, Observation, Validity)
from .video_observation_replay import DecodedFrame, VideoReplayInput


class LibreYOLOProviderError(RuntimeError):
    """A model invocation or output could not be represented honestly."""


@dataclass(frozen=True, slots=True)
class LibreYOLOProviderConfig:
    """Explicit detector identity and replay policy for one source stream."""
    source: str
    hypothesis_id: str
    relation: EvidenceRelation
    expires_after_ns: int
    model_identifier: str
    model_version: str
    package_version: str
    weights_sha256: str
    taxonomy: str = "coco-2017@80"
    model_source_url: str = "https://huggingface.co/LibreYOLO/LibreYOLO9t"
    backend: str = "torch-cpu"
    input_size: int = 640
    confidence_threshold: float = 0.25
    preprocessing: str = "LibreYOLO default BGR image preprocessing"

    def __post_init__(self) -> None:
        for value, name in ((self.source, "source"), (self.hypothesis_id, "hypothesis_id"),
                            (self.model_identifier, "model_identifier"),
                            (self.model_version, "model_version"),
                            (self.package_version, "package_version"), (self.taxonomy, "taxonomy"),
                            (self.model_source_url, "model_source_url"), (self.backend, "backend"),
                            (self.preprocessing, "preprocessing")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.relation, EvidenceRelation):
            raise ValueError("relation must be an EvidenceRelation")
        if (isinstance(self.expires_after_ns, bool) or not isinstance(self.expires_after_ns, int)
                or self.expires_after_ns < 0):
            raise ValueError("expires_after_ns must be a non-negative integer")
        digest = self.weights_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("weights_sha256 must be a lowercase SHA-256 hexadecimal digest")
        object.__setattr__(self, "weights_sha256", digest)
        if isinstance(self.input_size, bool) or not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if (isinstance(self.confidence_threshold, bool)
                or not isinstance(self.confidence_threshold, (int, float))
                or not math.isfinite(float(self.confidence_threshold))
                or not 0.0 <= float(self.confidence_threshold) <= 1.0):
            raise ValueError("confidence_threshold must be finite and in [0, 1]")
        object.__setattr__(self, "confidence_threshold", float(self.confidence_threshold))


class LibreYOLODetectionProvider:
    """Turn one decoded BGR frame into detection-only replay input.

    Failures raise ``LibreYOLOProviderError`` for the video harness to record;
    they never become fabricated zero detections. A successful empty result is
    available ``ObjectDetectionEvidence(detections=())``.
    """
    def __init__(self, model: Callable[..., Any], config: LibreYOLOProviderConfig) -> None:
        if not callable(model):
            raise ValueError("model must be callable")
        if not isinstance(config, LibreYOLOProviderConfig):
            raise ValueError("config must be a LibreYOLOProviderConfig")
        self._model, self._config = model, config

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, source: str, hypothesis_id: str,
                        relation: EvidenceRelation, expires_after_ns: int,
                        taxonomy: str = "coco-2017@80", confidence_threshold: float = 0.25
                        ) -> "LibreYOLODetectionProvider":
        """Load a local checkpoint on CPU; this factory never downloads weights."""
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        try:
            from importlib.metadata import version
            from libreyolo import LibreYOLO
            model = LibreYOLO(str(path), device="cpu", task="detect")
        except Exception as error:  # optional runtime / third-party load
            raise LibreYOLOProviderError(f"LibreYOLO checkpoint load failed: {error}") from error
        return cls(model, LibreYOLOProviderConfig(
            source=source, hypothesis_id=hypothesis_id, relation=relation,
            expires_after_ns=expires_after_ns, model_identifier="LibreYOLO9t",
            model_version="LibreYOLO9t", package_version=version("libreyolo"),
            weights_sha256=_sha256(path), taxonomy=taxonomy,
            confidence_threshold=confidence_threshold))

    @property
    def config(self) -> LibreYOLOProviderConfig:
        return self._config

    def __call__(self, frame: DecodedFrame) -> VideoReplayInput:
        if not isinstance(frame, DecodedFrame):
            raise LibreYOLOProviderError("provider input must be a DecodedFrame")
        try:
            result = self._model(frame.bgr, conf=self._config.confidence_threshold,
                                 imgsz=self._config.input_size, device="cpu", batch=1,
                                 color_format="bgr")
        except Exception as error:
            raise LibreYOLOProviderError(f"LibreYOLO inference failed: {error}") from error
        evidence = self._evidence_from_result(result, frame.width, frame.height)
        observation = Observation(
            observation_id=f"{self._config.source}:frame-{frame.frame_index}:{frame.capture_at_ns}",
            source=self._config.source, captured_at_ns=frame.capture_at_ns,
            coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.VALID,
            provenance={
                "input_frame": {"frame_index": frame.frame_index, "capture_at_ns": frame.capture_at_ns,
                                "width": frame.width, "height": frame.height},
                "detector": {"implementation": "LibreYOLO", "package_version": self._config.package_version,
                             "model_identifier": self._config.model_identifier,
                             "model_version": self._config.model_version,
                             "model_source_url": self._config.model_source_url,
                             "weights_sha256": self._config.weights_sha256,
                             "backend": self._config.backend, "input_size": self._config.input_size,
                             "preprocessing": self._config.preprocessing,
                             "score_semantics": "per-detection detector score; not observation confidence"}},
            evidence=EvidencePayload(object_detections=evidence))
        return VideoReplayInput(observation, self._config.hypothesis_id, self._config.relation,
                                frame.capture_at_ns + self._config.expires_after_ns)

    def _evidence_from_result(self, result: Any, width: int, height: int) -> ObjectDetectionEvidence:
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise LibreYOLOProviderError("LibreYOLO must return exactly one result per frame")
            result = result[0]
        marker = object()
        boxes = getattr(result, "boxes", marker)
        if boxes is marker:
            raise LibreYOLOProviderError("LibreYOLO result has no boxes attribute")
        if boxes is None:
            detections: tuple[ObjectDetection, ...] = ()
        else:
            try:
                xyxy = _matrix(boxes.xyxy, "boxes.xyxy", 4)
                scores = _vector(boxes.conf, "boxes.conf")
                class_ids = _vector(boxes.cls, "boxes.cls")
            except (AttributeError, TypeError, ValueError) as error:
                raise LibreYOLOProviderError(f"malformed LibreYOLO detection output: {error}") from error
            if not (len(xyxy) == len(scores) == len(class_ids)):
                raise LibreYOLOProviderError("LibreYOLO box, score, and class counts differ")
            values = []
            for box, score, class_id in zip(xyxy, scores, class_ids):
                if not float(class_id).is_integer() or class_id < 0:
                    raise LibreYOLOProviderError("LibreYOLO emitted a non-integral or negative class ID")
                try:
                    values.append(ObjectDetection(int(class_id), self._config.taxonomy, float(score),
                                                  tuple(float(value) for value in box)))
                except ValueError as error:
                    raise LibreYOLOProviderError(f"invalid LibreYOLO detection: {error}") from error
            detections = tuple(values)
        try:
            return ObjectDetectionEvidence(self._config.taxonomy, width, height,
                                           ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE, detections)
        except ValueError as error:
            raise LibreYOLOProviderError(f"invalid LibreYOLO image-space evidence: {error}") from error


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach): value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu): value = cpu()
    as_numpy = getattr(value, "numpy", None)
    if callable(as_numpy): value = as_numpy()
    return np.asarray(value)


def _matrix(value: Any, name: str, columns: int) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (N, {columns})")
    return array.astype(float, copy=False)


def _vector(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (N,)")
    return array.astype(float, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
