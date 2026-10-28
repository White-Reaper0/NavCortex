"""Fixture-only tests for typed, backend-neutral object-detection evidence."""
import math

import numpy as np
import pytest

from prism.belief_ledger import EvidenceRelation
from prism.doubt_engine import DoubtReason
from prism.observation_replay import ObservationReplayPipeline, ReplayInput, ReplayOutcome
from prism.observations import (
    CoordinateFrame,
    EvidencePayload,
    ImageCoordinateConvention,
    ObjectDetection,
    ObjectDetectionEvidence,
    Observation,
    Validity,
)


TAXONOMY = "fixture-objects@1"


def detection(**changes):
    base = dict(class_id=1, taxonomy=TAXONOMY, score=0.8, bbox_xyxy=(10.0, 20.0, 50.0, 70.0))
    base.update(changes)
    return ObjectDetection(**base)


def detections(items=(None,)):
    items = (detection(),) if items == (None,) else items
    return ObjectDetectionEvidence(
        taxonomy=TAXONOMY, image_width=100, image_height=80,
        coordinate_convention=ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE,
        detections=items,
    )


def observation(evidence, **changes):
    base = dict(
        observation_id="objects-1", source="fixture-detector", captured_at_ns=100,
        received_at_ns=120, coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.VALID,
        provenance={"backend": "fixture", "model": "not-loaded"}, evidence=evidence,
    )
    base.update(changes)
    return Observation(**base)


def test_valid_detection_evidence_is_typed_immutable_and_not_general_confidence():
    payload = EvidencePayload(object_detections=detections())
    item = payload.object_detections.detections[0]
    result = observation(payload)

    assert payload.has_evidence
    assert item.class_id == 1 and item.score == pytest.approx(0.8)
    assert item.bbox_xyxy == (10.0, 20.0, 50.0, 70.0)
    assert result.confidence is None
    with pytest.raises(AttributeError):
        item.score = 0.2


def test_empty_detection_result_is_available_and_distinct_from_unavailable():
    available_empty = EvidencePayload(object_detections=detections(items=()))
    unavailable = EvidencePayload()

    assert available_empty.has_evidence
    assert available_empty.object_detections.detections == ()
    assert unavailable.object_detections is None
    assert not unavailable.has_evidence
    with pytest.raises(ValueError, match="cannot contain evidence"):
        Observation(
            observation_id="unavailable", source="camera", captured_at_ns=0,
            coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.UNAVAILABLE,
            evidence=available_empty,
        )


@pytest.mark.parametrize("box", [
    (10.0, 20.0, 10.0, 70.0),
    (10.0, 20.0, math.inf, 70.0),
])
def test_detection_rejects_malformed_boxes(box):
    with pytest.raises(ValueError):
        detection(bbox_xyxy=box)
    with pytest.raises(ValueError, match="image bounds"):
        detections(items=(detection(bbox_xyxy=(10.0, 20.0, 101.0, 70.0)),))


@pytest.mark.parametrize("score", [-0.01, 1.01, math.nan, math.inf])
def test_detection_score_must_be_finite_unit_interval(score):
    with pytest.raises(ValueError):
        detection(score=score)


def test_detection_taxonomy_must_match_its_enclosing_evidence():
    with pytest.raises(ValueError, match="taxonomy"):
        detections(items=(detection(taxonomy="another-taxonomy@1"),))


def test_detection_only_observation_preserves_audit_data_but_is_not_runtime_geometry():
    source = observation(EvidencePayload(object_detections=detections()))
    result = ObservationReplayPipeline().replay([
        ReplayInput(source, "pedestrian-present", EvidenceRelation.SUPPORTS, expires_at_ns=200)
    ], at_ns=150)

    record = result.records[0]
    assert record.outcome is ReplayOutcome.RUNTIME_REJECTED
    assert "requires metric_depth_m or vehicle-frame points" in record.adapter_error
    assert record.evidence_record is None  # incompatible runtime evidence stays replay-audited
    assert record.input.observation.source == "fixture-detector"
    assert record.input.observation.captured_at_ns == 100
    assert record.input.observation.provenance["model"] == "not-loaded"
    assert record.input.observation.evidence.object_detections.detections[0].class_id == 1
    assert result.assessment.items[0].reason is DoubtReason.UNKNOWN
