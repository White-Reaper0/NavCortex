"""Immutable, backend-neutral spatial observations for P.R.I.S.M.

An observation records only evidence a producer actually supplied.  In
particular, ``relative_depth`` is an ordering-like depth signal with no metre
unit, while ``metric_depth_m`` is optical-axis depth in metres.  Consumers must
not reinterpret one as the other or manufacture metric depth from relative
depth without a separately recorded calibration/alignment procedure.

This module deliberately imports neither prism nor any neural runtime.  A
future adapter may translate validated observations into a backend-specific
packet without making P.R.I.S.M.'s reasoning layer depend on that backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class CoordinateFrame(str, Enum):
    """Frames currently accepted at the P.R.I.S.M. observation boundary."""

    CAMERA = "camera"
    VEHICLE = "vehicle"
    LOCAL_MAP = "local_map"


class Validity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class Staleness(str, Enum):
    FRESH = "fresh"
    AGED = "aged"
    EXPIRED = "expired"


class ConfidenceKind(str, Enum):
    """Meaning of a confidence value; it is never assumed calibrated."""

    REPORTED = "reported"
    CALIBRATED = "calibrated"
    HEURISTIC = "heuristic"


class ImageCoordinateConvention(str, Enum):
    """Image-space convention for object-detection bounding boxes.

    ``PIXEL_XYXY_EXCLUSIVE`` uses a top-left image origin and boxes ordered as
    ``(x_min, y_min, x_max, y_max)``.  Minimum edges are inclusive and maximum
    edges are exclusive, so a box must satisfy ``0 <= x_min < x_max <= width``
    and ``0 <= y_min < y_max <= height``.  Other conventions must be converted
    explicitly by the producer before entering this contract.
    """

    PIXEL_XYXY_EXCLUSIVE = "pixel_xyxy_exclusive"


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer nanosecond timestamp")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _readonly_array(value: np.ndarray, name: str, *, ndim: Optional[int] = None,
                    final_dim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if final_dim is not None and (array.ndim == 0 or array.shape[-1] != final_dim):
        raise ValueError(f"{name} must have final dimension {final_dim}")
    # Copy before marking read-only: callers retain no mutable alias.
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _freeze_metadata(value: Any) -> Any:
    """Copy recursively so provenance cannot be mutated through caller aliases."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(v) for v in value)
    if isinstance(value, np.ndarray):
        return _readonly_array(value, "provenance array")
    return value


@dataclass(frozen=True, slots=True)
class EgoMotionEvidence:
    """One reported ego-motion increment in the observation's coordinate frame."""

    translation_m: tuple[float, float, float]
    yaw_rad: float
    tracking_ok: bool
    quality: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.translation_m) != 3:
            raise ValueError("translation_m must contain exactly three values")
        translation = tuple(_finite(v, "translation_m") for v in self.translation_m)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "yaw_rad", _finite(self.yaw_rad, "yaw_rad"))
        if not isinstance(self.tracking_ok, bool):
            raise ValueError("tracking_ok must be bool")
        if self.quality is not None:
            quality = _finite(self.quality, "quality")
            if not 0.0 <= quality <= 1.0:
                raise ValueError("quality must be in [0, 1]")
            object.__setattr__(self, "quality", quality)


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    """One scored object box, with no implied depth, motion, or safety meaning."""

    class_id: str | int
    taxonomy: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        class_id = self.class_id
        if isinstance(class_id, bool) or not isinstance(class_id, (str, int)):
            raise ValueError("class_id must be a non-empty string or non-negative integer")
        if isinstance(class_id, str):
            object.__setattr__(self, "class_id", _require_nonempty(class_id, "class_id"))
        elif class_id < 0:
            raise ValueError("class_id must not be negative")
        object.__setattr__(self, "taxonomy", _require_nonempty(self.taxonomy, "taxonomy"))
        score = _finite(self.score, "detection score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("detection score must be in [0, 1]")
        object.__setattr__(self, "score", score)
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four coordinates")
        box = tuple(_finite(value, "bbox_xyxy") for value in self.bbox_xyxy)
        if not box[0] < box[2] or not box[1] < box[3]:
            raise ValueError("bbox_xyxy must have strictly ordered min/max edges")
        object.__setattr__(self, "bbox_xyxy", box)


@dataclass(frozen=True, slots=True)
class ObjectDetectionEvidence:
    """Available image-space detection output for one decoded image.

    An empty ``detections`` tuple means the frame was evaluated and yielded zero
    detections.  ``EvidencePayload.object_detections is None`` instead means no
    object-detection result is available.  Scores retain only the producing
    detector's class-score semantics; they are never general confidence.
    """

    taxonomy: str
    image_width: int
    image_height: int
    coordinate_convention: ImageCoordinateConvention
    detections: tuple[ObjectDetection, ...] = ()

    def __post_init__(self) -> None:
        taxonomy = _require_nonempty(self.taxonomy, "detection taxonomy")
        object.__setattr__(self, "taxonomy", taxonomy)
        for value, name in ((self.image_width, "image_width"),
                            (self.image_height, "image_height")):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        if self.coordinate_convention is not ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE:
            raise ValueError("coordinate_convention must be a supported ImageCoordinateConvention")
        detections = tuple(self.detections)
        for detection in detections:
            if not isinstance(detection, ObjectDetection):
                raise ValueError("detections must contain ObjectDetection values")
            if detection.taxonomy != taxonomy:
                raise ValueError("each detection taxonomy must match detection evidence taxonomy")
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            if not (0.0 <= x_min < x_max <= self.image_width
                    and 0.0 <= y_min < y_max <= self.image_height):
                raise ValueError("bbox_xyxy must lie within image bounds")
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True, slots=True)
class EvidencePayload:
    """Optional evidence with explicit units and taxonomy identifiers.

    ``relative_depth`` and ``metric_depth_m`` deliberately have distinct fields.
    Relative depth must never be presented as metric geometry.
    """

    relative_depth: Optional[np.ndarray] = None
    metric_depth_m: Optional[np.ndarray] = None
    depth_valid: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    semantic_labels: Optional[np.ndarray] = None
    semantic_taxonomy: Optional[str] = None
    traversability_probabilities: Optional[np.ndarray] = None
    traversability_labels: Optional[np.ndarray] = None
    traversability_classes: Optional[tuple[str, ...]] = None
    ego_motion: Optional[EgoMotionEvidence] = None
    object_detections: Optional[ObjectDetectionEvidence] = None

    def __post_init__(self) -> None:
        relative = None if self.relative_depth is None else _readonly_array(
            self.relative_depth, "relative_depth", ndim=2)
        metric = None if self.metric_depth_m is None else _readonly_array(
            self.metric_depth_m, "metric_depth_m", ndim=2)
        if relative is not None:
            object.__setattr__(self, "relative_depth", relative)
        if metric is not None:
            object.__setattr__(self, "metric_depth_m", metric)
        if self.depth_valid is not None:
            valid = _readonly_array(self.depth_valid, "depth_valid", ndim=2)
            depth = metric if metric is not None else relative
            if depth is None:
                raise ValueError("depth_valid requires relative_depth or metric_depth_m")
            if valid.shape != depth.shape:
                raise ValueError("depth_valid shape must match its depth evidence")
            object.__setattr__(self, "depth_valid", valid.astype(bool, copy=False))
        if self.points is not None:
            object.__setattr__(self, "points", _readonly_array(
                self.points, "points", final_dim=3))
        if self.semantic_labels is not None:
            _require_nonempty(self.semantic_taxonomy or "", "semantic_taxonomy")
            object.__setattr__(self, "semantic_labels", _readonly_array(
                self.semantic_labels, "semantic_labels", ndim=2))
        elif self.semantic_taxonomy is not None:
            raise ValueError("semantic_taxonomy requires semantic_labels")
        if self.traversability_probabilities is not None:
            probabilities = _readonly_array(
                self.traversability_probabilities, "traversability_probabilities", ndim=3)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("traversability_probabilities must be finite values in [0, 1]")
            if not self.traversability_classes:
                raise ValueError("traversability_probabilities requires traversability_classes")
            classes = tuple(_require_nonempty(c, "traversability class")
                            for c in self.traversability_classes)
            if probabilities.shape[0] != len(classes):
                raise ValueError("traversability class count must match probability channels")
            object.__setattr__(self, "traversability_probabilities", probabilities)
            object.__setattr__(self, "traversability_classes", classes)
        elif self.traversability_classes is not None:
            raise ValueError("traversability_classes requires traversability_probabilities")
        if self.traversability_labels is not None:
            object.__setattr__(self, "traversability_labels", _readonly_array(
                self.traversability_labels, "traversability_labels", ndim=2))
        if self.object_detections is not None and not isinstance(
                self.object_detections, ObjectDetectionEvidence):
            raise ValueError("object_detections must be an ObjectDetectionEvidence")

    @property
    def has_evidence(self) -> bool:
        return any(value is not None for value in (
            self.relative_depth, self.metric_depth_m, self.points,
            self.semantic_labels, self.traversability_probabilities,
            self.traversability_labels, self.ego_motion, self.object_detections,
        ))


@dataclass(frozen=True, slots=True)
class Observation:
    """A validated, immutable spatial observation from any perception backend."""

    observation_id: str
    source: str
    captured_at_ns: int
    coordinate_frame: CoordinateFrame
    validity: Validity
    received_at_ns: Optional[int] = None
    invalid_reason: Optional[str] = None
    confidence: Optional[float] = None
    confidence_kind: Optional[ConfidenceKind] = None
    staleness: Staleness = Staleness.FRESH
    age_ms: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    evidence: EvidencePayload = field(default_factory=EvidencePayload)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _require_nonempty(
            self.observation_id, "observation_id"))
        object.__setattr__(self, "source", _require_nonempty(self.source, "source"))
        captured = _timestamp(self.captured_at_ns, "captured_at_ns")
        object.__setattr__(self, "captured_at_ns", captured)
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise ValueError("coordinate_frame must be a supported CoordinateFrame")
        if not isinstance(self.validity, Validity):
            raise ValueError("validity must be a Validity value")
        if not isinstance(self.staleness, Staleness):
            raise ValueError("staleness must be a Staleness value")
        if self.received_at_ns is not None:
            received = _timestamp(self.received_at_ns, "received_at_ns")
            if received < captured:
                raise ValueError("received_at_ns cannot precede captured_at_ns")
            object.__setattr__(self, "received_at_ns", received)
        if self.invalid_reason is not None:
            object.__setattr__(self, "invalid_reason", _require_nonempty(
                self.invalid_reason, "invalid_reason"))
        if self.validity is Validity.VALID and self.invalid_reason is not None:
            raise ValueError("VALID observations cannot have invalid_reason")
        if self.validity is Validity.INVALID and self.invalid_reason is None:
            raise ValueError("INVALID observations require invalid_reason")
        if (self.confidence is None) != (self.confidence_kind is None):
            raise ValueError("confidence and confidence_kind must be supplied together")
        if self.confidence is not None:
            confidence = _finite(self.confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
            if not isinstance(self.confidence_kind, ConfidenceKind):
                raise ValueError("confidence_kind must be a ConfidenceKind value")
            object.__setattr__(self, "confidence", confidence)
        if self.age_ms is not None:
            age = _finite(self.age_ms, "age_ms")
            if age < 0:
                raise ValueError("age_ms must not be negative")
            object.__setattr__(self, "age_ms", age)
        if self.staleness is Staleness.FRESH and self.age_ms not in (None, 0.0):
            raise ValueError("FRESH observations must have no age or age_ms=0")
        if self.staleness in (Staleness.AGED, Staleness.EXPIRED) and (
                self.age_ms is None or self.age_ms <= 0):
            raise ValueError("AGED and EXPIRED observations require positive age_ms")
        if not isinstance(self.evidence, EvidencePayload):
            raise ValueError("evidence must be an EvidencePayload")
        if self.validity is Validity.UNAVAILABLE:
            if self.evidence.has_evidence:
                raise ValueError("UNAVAILABLE observations cannot contain evidence")
            if self.confidence is not None:
                raise ValueError("UNAVAILABLE observations cannot contain confidence")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        object.__setattr__(self, "provenance", _freeze_metadata(self.provenance))
