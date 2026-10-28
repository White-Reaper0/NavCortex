"""Safe conversion from P.R.I.S.M. observations to existing runtime packets.

The adapter is intentionally narrow.  It only transfers measurements that the
current mapping/runtime interface can consume without inferring a missing
calibration, coordinate transform, confidence map, or pose component.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from .config import N_TERRAIN, N_TRAV, TRAV_CLASSES
from .observations import CoordinateFrame, Observation, Staleness, Validity
from .types import (
    DepthResult,
    FramePacket,
    GeometryResult,
    OdometryResult,
    SegResult,
    TraversabilityResult,
    UncertaintyResult,
)


class ObservationAdapterError(ValueError):
    """Raised when an observation cannot safely become a runtime input."""


@dataclass(frozen=True, slots=True)
class AdapterAudit:
    """Immutable provenance retained outside the legacy ``FramePacket`` type."""

    observation_id: str
    source: str
    captured_at_ns: int
    received_at_ns: Optional[int]
    coordinate_frame: CoordinateFrame
    validity: Validity
    staleness: Staleness
    provenance: Mapping[str, Any]
    applied_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeAdapterResult:
    """A legacy runtime packet together with an auditable conversion record."""

    packet: FramePacket
    audit: AdapterAudit


def _copy_array(value: np.ndarray, dtype=None) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True)


def _dense_shape(observation: Observation) -> Optional[tuple[int, int]]:
    evidence = observation.evidence
    if evidence.metric_depth_m is not None:
        return tuple(evidence.metric_depth_m.shape)
    if evidence.points is not None:
        points = evidence.points
        if points.ndim != 3:
            raise ObservationAdapterError("runtime mapping requires dense (H, W, 3) points")
        return tuple(points.shape[:2])
    return None


def _require_dense_shape(name: str, value: np.ndarray,
                         expected: Optional[tuple[int, int]]) -> tuple[int, int]:
    if value.ndim < 2:
        raise ObservationAdapterError(f"{name} must be dense image evidence")
    shape = tuple(value.shape[-2:]) if value.ndim == 3 and name == "traversability_probabilities" \
        else tuple(value.shape[:2])
    if expected is None:
        raise ObservationAdapterError(f"{name} requires metric depth or vehicle points")
    if shape != expected:
        raise ObservationAdapterError(f"{name} shape {shape} does not match geometry shape {expected}")
    return shape


class RuntimeAdapter:
    """Convert one fresh, valid observation into a ``FramePacket``.

    Camera-frame observations can provide metric optical-axis depth. Vehicle-frame
    observations can provide already-transformed dense XYZ points.  The adapter
    never transforms camera points to vehicle points, calculates metric depth from
    relative depth, broadcasts scalar confidence into a map, or derives an absent
    traversability label/probability.
    """

    def to_packet(self, observation: Observation, *, packet_index: int,
                  clip_id: Optional[str] = None) -> RuntimeAdapterResult:
        if not isinstance(observation, Observation):
            raise ObservationAdapterError("observation must be an Observation")
        if not isinstance(packet_index, int) or isinstance(packet_index, bool) or packet_index < 0:
            raise ObservationAdapterError("packet_index must be a non-negative integer")
        if observation.validity is not Validity.VALID:
            raise ObservationAdapterError("only VALID observations may become runtime input")
        # The legacy map resets the age of any supplied geometry.  It has no field
        # to preserve external evidence age, so accepting AGED data would make it
        # falsely fresh; reject both non-fresh states instead.
        if observation.staleness is not Staleness.FRESH:
            raise ObservationAdapterError("only FRESH observations may become usable geometry")
        if observation.coordinate_frame is CoordinateFrame.LOCAL_MAP:
            raise ObservationAdapterError("local_map observations need a transform and are unsupported")

        evidence = observation.evidence
        if evidence.relative_depth is not None:
            raise ObservationAdapterError(
                "relative_depth cannot be converted to runtime metric geometry")
        if observation.coordinate_frame is CoordinateFrame.CAMERA and evidence.points is not None:
            raise ObservationAdapterError("camera-frame points need an explicit transform to vehicle frame")
        if observation.coordinate_frame is CoordinateFrame.VEHICLE and evidence.metric_depth_m is not None:
            raise ObservationAdapterError("metric_depth_m is optical-axis camera evidence, not vehicle-frame geometry")

        shape = _dense_shape(observation)
        if shape is None:
            raise ObservationAdapterError("runtime mapping requires metric_depth_m or vehicle-frame points")
        if evidence.depth_valid is not None and tuple(evidence.depth_valid.shape) != shape:
            raise ObservationAdapterError("depth_valid shape does not match runtime geometry")

        packet = FramePacket(
            clip_id=clip_id if clip_id is not None else observation.source,
            idx=packet_index,
            t=observation.captured_at_ns / 1_000_000_000.0,
        )
        applied: list[str] = []

        if evidence.metric_depth_m is not None:
            depth = _copy_array(evidence.metric_depth_m, np.float32)
            valid = (np.isfinite(depth) if evidence.depth_valid is None
                     else _copy_array(evidence.depth_valid, bool))
            # rel_inv is unavailable, so it remains None.  Mapping only consumes
            # depth_m and valid; downstream code must not treat rel_inv as measured.
            packet.depth = DepthResult(rel_inv=None, depth_m=depth, valid=valid)
            applied.append("metric_depth_m")

        if evidence.points is not None:
            points = _copy_array(evidence.points, np.float32)
            if points.shape != (*shape, 3):
                raise ObservationAdapterError("points must have shape (H, W, 3)")
            # Mapping consumes only points_veh.  All other GeometryResult products
            # are explicitly unavailable rather than calculated by this adapter.
            packet.geom = GeometryResult(
                points_cam=None,
                points_veh=points,
                height_above_ground=None,
                slope_deg=None,
                roughness=None,
                ground_normal=None,
            )
            applied.append("points")

        if evidence.semantic_labels is not None:
            _require_dense_shape("semantic_labels", evidence.semantic_labels, shape)
            labels = _copy_array(evidence.semantic_labels, np.uint8)
            if (labels >= N_TERRAIN).any():
                raise ObservationAdapterError("semantic_labels contain values outside the runtime taxonomy")
            packet.seg = SegResult(logits=None, label=labels, prob_max=None, entropy=None)
            applied.append("semantic_labels")

        if evidence.traversability_probabilities is not None:
            probabilities = evidence.traversability_probabilities
            _require_dense_shape("traversability_probabilities", probabilities, shape)
            if tuple(evidence.traversability_classes or ()) != tuple(TRAV_CLASSES):
                raise ObservationAdapterError("traversability_classes must match the runtime class order")
            if probabilities.shape[0] != N_TRAV:
                raise ObservationAdapterError("traversability probabilities need four runtime channels")
            packet.trav = TraversabilityResult(
                prob=_copy_array(probabilities, np.float32), label=None, risk=None)
            applied.append("traversability_probabilities")
        elif evidence.traversability_labels is not None:
            _require_dense_shape("traversability_labels", evidence.traversability_labels, shape)
            labels = _copy_array(evidence.traversability_labels, np.uint8)
            if (labels >= N_TRAV).any():
                raise ObservationAdapterError("traversability_labels contain values outside runtime classes")
            packet.trav = TraversabilityResult(prob=None, label=labels, risk=None)
            applied.append("traversability_labels")

        if observation.confidence is not None:
            # A scalar confidence is useful to the supervisor but is not a dense
            # confidence map.  Preserve that distinction by leaving all maps None.
            packet.unc = UncertaintyResult(
                depth_conf=None, seg_conf=None, fused_conf=None,
                mean_conf=observation.confidence,
            )
            applied.append("confidence")

        if evidence.ego_motion is not None:
            if observation.coordinate_frame is not CoordinateFrame.VEHICLE:
                raise ObservationAdapterError("ego_motion requires vehicle coordinate_frame")
            motion = evidence.ego_motion
            if abs(motion.translation_m[0]) > 1e-9 or abs(motion.translation_m[2]) > 1e-9:
                raise ObservationAdapterError(
                    "runtime odometry cannot represent lateral or vertical translation")
            if motion.quality is None:
                raise ObservationAdapterError("ego_motion quality is required for runtime odometry")
            packet.odom = OdometryResult(
                d_trans=motion.translation_m[1], d_yaw=motion.yaw_rad,
                speed_mps=None, n_matches=None, n_inliers=None,
                tracking_ok=motion.tracking_ok, track_quality=motion.quality,
            )
            applied.append("ego_motion")

        audit = AdapterAudit(
            observation_id=observation.observation_id,
            source=observation.source,
            captured_at_ns=observation.captured_at_ns,
            received_at_ns=observation.received_at_ns,
            coordinate_frame=observation.coordinate_frame,
            validity=observation.validity,
            staleness=observation.staleness,
            provenance=MappingProxyType(dict(observation.provenance)),
            applied_fields=tuple(applied),
        )
        return RuntimeAdapterResult(packet=packet, audit=audit)
