"""P.R.I.S.M. observation-to-runtime adapter tests; no models are loaded."""
import numpy as np
import pytest

from prism.observations import (
    ConfidenceKind,
    CoordinateFrame,
    EgoMotionEvidence,
    EvidencePayload,
    Observation,
    Staleness,
    Validity,
)
from prism.runtime_adapter import ObservationAdapterError, RuntimeAdapter


def observation(*, frame=CoordinateFrame.CAMERA, evidence=None, **changes):
    base = dict(
        observation_id="adapter-001",
        source="replay:clip_01",
        captured_at_ns=2_500_000_000,
        received_at_ns=2_600_000_000,
        coordinate_frame=frame,
        validity=Validity.VALID,
        provenance={"backend": "replay", "input": {"frame": 75}},
        evidence=evidence or EvidencePayload(metric_depth_m=np.ones((2, 3), np.float32)),
    )
    base.update(changes)
    return Observation(**base)


def test_valid_camera_metric_depth_conversion_preserves_audit_data():
    payload = EvidencePayload(
        metric_depth_m=np.full((2, 3), 2.0, np.float32),
        depth_valid=np.array([[1, 1, 0], [1, 0, 1]], bool),
        semantic_labels=np.ones((2, 3), np.uint8),
        semantic_taxonomy="prism-7",
        traversability_probabilities=np.full((4, 2, 3), 0.25, np.float32),
        traversability_classes=("safe", "risky", "obstacle", "unknown"),
    )
    result = RuntimeAdapter().to_packet(
        observation(evidence=payload, confidence=0.7, confidence_kind=ConfidenceKind.REPORTED),
        packet_index=4,
    )

    packet = result.packet
    assert packet.idx == 4 and packet.t == pytest.approx(2.5)
    assert packet.clip_id == "replay:clip_01"
    assert packet.depth.rel_inv is None
    assert np.array_equal(packet.depth.valid, payload.depth_valid)
    assert packet.seg.label.shape == (2, 3)
    assert packet.trav.prob.shape == (4, 2, 3)
    assert packet.unc.mean_conf == pytest.approx(0.7)
    assert packet.unc.fused_conf is None
    assert result.audit.observation_id == "adapter-001"
    assert result.audit.source == "replay:clip_01"
    assert result.audit.captured_at_ns == 2_500_000_000
    assert result.audit.provenance["input"]["frame"] == 75
    assert result.audit.applied_fields == (
        "metric_depth_m", "semantic_labels", "traversability_probabilities", "confidence")


def test_vehicle_points_and_supported_ego_motion_convert_without_transforming():
    points = np.zeros((2, 2, 3), np.float32)
    points[..., 1] = 1.0
    points[..., 2] = 0.1
    payload = EvidencePayload(
        points=points,
        ego_motion=EgoMotionEvidence((0.0, 0.25, 0.0), 0.02, True, 0.8),
    )
    result = RuntimeAdapter().to_packet(observation(frame=CoordinateFrame.VEHICLE, evidence=payload),
                                        packet_index=0, clip_id="external")

    assert result.packet.geom.points_veh.shape == (2, 2, 3)
    assert result.packet.geom.points_cam is None
    assert result.packet.odom.d_trans == pytest.approx(0.25)
    assert result.packet.odom.d_yaw == pytest.approx(0.02)
    assert result.packet.odom.track_quality == pytest.approx(0.8)
    assert result.audit.applied_fields == ("points", "ego_motion")


def test_unavailable_evidence_is_not_converted_or_fabricated():
    unavailable = Observation(
        observation_id="none",
        source="camera",
        captured_at_ns=0,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.UNAVAILABLE,
    )
    with pytest.raises(ObservationAdapterError, match="only VALID"):
        RuntimeAdapter().to_packet(unavailable, packet_index=0)

    with pytest.raises(ObservationAdapterError, match="relative_depth"):
        RuntimeAdapter().to_packet(
            observation(evidence=EvidencePayload(relative_depth=np.ones((2, 2), np.float32))),
            packet_index=0,
        )


@pytest.mark.parametrize("validity,staleness", [
    (Validity.INVALID, Staleness.FRESH),
    (Validity.VALID, Staleness.EXPIRED),
    (Validity.VALID, Staleness.AGED),
])
def test_invalid_or_non_fresh_input_cannot_become_usable_geometry(validity, staleness):
    kwargs = {"validity": validity, "staleness": staleness}
    if validity is Validity.INVALID:
        kwargs["invalid_reason"] = "sensor error"
    if staleness is not Staleness.FRESH:
        kwargs["age_ms"] = 10.0
    with pytest.raises(ObservationAdapterError):
        RuntimeAdapter().to_packet(observation(**kwargs), packet_index=0)


def test_incompatible_dense_shapes_and_unsupported_frames_are_rejected():
    payload = EvidencePayload(
        metric_depth_m=np.ones((2, 3), np.float32),
        semantic_labels=np.zeros((3, 2), np.uint8),
        semantic_taxonomy="prism-7",
    )
    with pytest.raises(ObservationAdapterError, match="does not match"):
        RuntimeAdapter().to_packet(observation(evidence=payload), packet_index=0)

    local = observation(frame=CoordinateFrame.LOCAL_MAP)
    with pytest.raises(ObservationAdapterError, match="local_map"):
        RuntimeAdapter().to_packet(local, packet_index=0)


def test_adapter_rejects_unsafe_motion_and_unknown_runtime_taxonomy():
    points = np.zeros((2, 2, 3), np.float32)
    bad_motion = EvidencePayload(
        points=points,
        ego_motion=EgoMotionEvidence((0.1, 0.0, 0.0), 0.0, True, 0.8),
    )
    with pytest.raises(ObservationAdapterError, match="lateral"):
        RuntimeAdapter().to_packet(
            observation(frame=CoordinateFrame.VEHICLE, evidence=bad_motion), packet_index=0)

    bad_classes = EvidencePayload(
        metric_depth_m=np.ones((2, 2), np.float32),
        traversability_probabilities=np.full((4, 2, 2), 0.25, np.float32),
        traversability_classes=("free", "risk", "blocked", "unknown"),
    )
    with pytest.raises(ObservationAdapterError, match="class order"):
        RuntimeAdapter().to_packet(observation(evidence=bad_classes), packet_index=0)
