"""Tests for the backend-neutral P.R.I.S.M. observation contract."""
from types import MappingProxyType

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


def valid_observation(**changes):
    base = dict(
        observation_id="obs-001",
        source="replay:clip_01",
        captured_at_ns=10,
        received_at_ns=12,
        coordinate_frame=CoordinateFrame.VEHICLE,
        validity=Validity.VALID,
        confidence=0.8,
        confidence_kind=ConfidenceKind.REPORTED,
        provenance={"clip": "clip_01", "nested": {"frame": 7}},
    )
    base.update(changes)
    return Observation(**base)


def test_valid_observation_keeps_available_evidence_immutable():
    depth = np.full((2, 3), 1.5, np.float32)
    points = np.zeros((2, 3, 3), np.float32)
    payload = EvidencePayload(
        metric_depth_m=depth,
        depth_valid=np.ones((2, 3), bool),
        points=points,
        semantic_labels=np.zeros((2, 3), np.uint8),
        semantic_taxonomy="example-v1",
        traversability_probabilities=np.full((2, 2, 3), 0.5, np.float32),
        traversability_classes=("safe", "unknown"),
        ego_motion=EgoMotionEvidence((0.0, 0.1, 0.0), 0.01, True, 0.9),
    )
    observation = valid_observation(evidence=payload)

    depth[0, 0] = 99.0
    points[0, 0, 0] = 99.0
    assert observation.evidence.metric_depth_m[0, 0] == 1.5
    assert observation.evidence.points[0, 0, 0] == 0.0
    assert not observation.evidence.metric_depth_m.flags.writeable
    with pytest.raises(ValueError):
        observation.evidence.metric_depth_m[0, 0] = 2.0


def test_missing_optional_evidence_is_not_fabricated():
    observation = valid_observation(confidence=None, confidence_kind=None)
    assert not observation.evidence.has_evidence
    assert observation.evidence.metric_depth_m is None
    assert observation.evidence.relative_depth is None
    assert observation.evidence.points is None
    assert observation.confidence is None


def test_metadata_is_recursively_immutable_and_copied():
    metadata = {"backend": "replay", "details": {"frames": [1, 2]}}
    observation = valid_observation(provenance=metadata)
    metadata["details"]["frames"].append(3)

    assert isinstance(observation.provenance, MappingProxyType)
    assert observation.provenance["details"]["frames"] == (1, 2)
    with pytest.raises(TypeError):
        observation.provenance["backend"] = "changed"


def test_explicit_unavailable_state_has_no_measurements():
    observation = Observation(
        observation_id="obs-unavailable",
        source="camera",
        captured_at_ns=0,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.UNAVAILABLE,
        invalid_reason="camera disconnected",
        staleness=Staleness.EXPIRED,
        age_ms=250.0,
    )
    assert observation.validity is Validity.UNAVAILABLE
    assert not observation.evidence.has_evidence

    with pytest.raises(ValueError, match="cannot contain evidence"):
        Observation(
            observation_id="bad-unavailable",
            source="camera",
            captured_at_ns=0,
            coordinate_frame=CoordinateFrame.CAMERA,
            validity=Validity.UNAVAILABLE,
            evidence=EvidencePayload(relative_depth=np.ones((1, 1))),
        )


@pytest.mark.parametrize("field,value", [
    ("observation_id", ""),
    ("source", " "),
    ("captured_at_ns", -1),
])
def test_required_identity_and_capture_timestamp_validation(field, value):
    with pytest.raises(ValueError):
        valid_observation(**{field: value})


def test_timestamp_validation_rejects_receive_before_capture():
    with pytest.raises(ValueError, match="cannot precede"):
        valid_observation(received_at_ns=9)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_confidence_must_be_finite_unit_interval(value):
    with pytest.raises(ValueError):
        valid_observation(confidence=value)


def test_validity_and_reason_rules_are_explicit():
    with pytest.raises(ValueError, match="cannot have"):
        valid_observation(invalid_reason="wrong")
    with pytest.raises(ValueError, match="require invalid_reason"):
        valid_observation(validity=Validity.INVALID)
    invalid = valid_observation(validity=Validity.INVALID, invalid_reason="checksum failed")
    assert invalid.invalid_reason == "checksum failed"


@pytest.mark.parametrize("staleness,age", [
    (Staleness.FRESH, 1.0),
    (Staleness.AGED, None),
    (Staleness.AGED, 0.0),
    (Staleness.EXPIRED, -1.0),
    (Staleness.EXPIRED, float("inf")),
])
def test_staleness_age_validation(staleness, age):
    with pytest.raises(ValueError):
        valid_observation(staleness=staleness, age_ms=age)


def test_depth_contract_keeps_relative_and_metric_measurements_distinct():
    payload = EvidencePayload(
        relative_depth=np.array([[0.2]], np.float32),
        metric_depth_m=np.array([[2.5]], np.float32),
        depth_valid=np.array([[True]]),
    )
    assert payload.relative_depth[0, 0] == pytest.approx(0.2)
    assert payload.metric_depth_m[0, 0] == pytest.approx(2.5)


def test_payload_validation_rejects_unpaired_or_inconsistent_evidence():
    with pytest.raises(ValueError, match="depth_valid requires"):
        EvidencePayload(depth_valid=np.ones((1, 1), bool))
    with pytest.raises(ValueError, match="shape must match"):
        EvidencePayload(metric_depth_m=np.ones((2, 2)), depth_valid=np.ones((1, 1), bool))
    with pytest.raises(ValueError, match="semantic_taxonomy"):
        EvidencePayload(semantic_labels=np.zeros((1, 1), np.uint8))
    with pytest.raises(ValueError, match="class count"):
        EvidencePayload(
            traversability_probabilities=np.full((2, 1, 1), 0.5),
            traversability_classes=("safe",),
        )
