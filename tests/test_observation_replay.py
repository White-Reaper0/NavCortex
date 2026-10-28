"""Integration tests for the deterministic, non-executing replay pipeline."""
import numpy as np

from prism.belief_ledger import EvidenceRelation, EvidenceState, HypothesisStatus
from prism.doubt_engine import DoubtReason, RecommendationKind
from prism.observation_replay import (
    AdvisoryDisposition,
    ObservationReplayPipeline,
    ReplayInput,
    ReplayOutcome,
)
from prism.observations import CoordinateFrame, EvidencePayload, Observation, Staleness, Validity


def observation(observation_id: str, **changes) -> Observation:
    base = dict(
        observation_id=observation_id,
        source="recording:clip_01",
        captured_at_ns=1_000,
        received_at_ns=1_100,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.VALID,
        provenance={"clip": "clip_01", "frame": 3},
        evidence=EvidencePayload(metric_depth_m=np.full((2, 2), 2.0, np.float32)),
    )
    base.update(changes)
    return Observation(**base)


def entry(observation_value: Observation, hypothesis="path-clear", relation=EvidenceRelation.SUPPORTS,
          expiry=2_000) -> ReplayInput:
    return ReplayInput(observation_value, hypothesis, relation, expiry)


def test_valid_observation_adapts_and_preserves_audit_provenance_end_to_end():
    result = ObservationReplayPipeline().replay([entry(observation("valid"))], at_ns=1_500)

    record = result.records[0]
    assert record.outcome is ReplayOutcome.ADAPTED
    assert record.adapter_result.packet.depth.depth_m.shape == (2, 2)
    assert record.adapter_result.audit.provenance["clip"] == "clip_01"
    assert record.evidence_record.provenance["frame"] == 3
    assert record.evidence_record.captured_at_ns == 1_000
    assert result.assessment.items == ()
    assert result.advisory.disposition is AdvisoryDisposition.NO_ACTION
    assert "not a safe-to-move" in result.advisory.reason


def test_unavailable_observation_is_adapter_rejected_but_audited_as_unknown():
    unavailable = Observation(
        observation_id="offline", source="camera", captured_at_ns=1_000,
        coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.UNAVAILABLE,
        provenance={"cause": "offline"},
    )
    result = ObservationReplayPipeline().replay([entry(unavailable)], at_ns=1_500)

    record = result.records[0]
    assert record.outcome is ReplayOutcome.RUNTIME_REJECTED
    assert record.adapter_result is None and "only VALID" in record.adapter_error
    assert record.evidence_record.provenance["cause"] == "offline"
    assert result.assessment.items[0].reason is DoubtReason.UNKNOWN
    assert result.assessment.items[0].evidence[0].state is EvidenceState.UNAVAILABLE
    assert result.advisory.disposition is AdvisoryDisposition.CAUTIOUS_STOP


def test_contradictory_valid_evidence_never_produces_no_action_advisory():
    entries = [
        entry(observation("yes")),
        entry(observation("no"), relation=EvidenceRelation.CONTRADICTS),
    ]
    result = ObservationReplayPipeline().replay(entries, at_ns=1_500)

    assert all(record.outcome is ReplayOutcome.ADAPTED for record in result.records)
    assert result.assessment.items[0].status is HypothesisStatus.CONTRADICTORY
    assert result.assessment.items[0].reason is DoubtReason.CONTRADICTORY
    assert RecommendationKind.CAUTIOUS_STOP in [r.kind for r in result.assessment.recommendations]
    assert result.advisory.disposition is AdvisoryDisposition.CAUTIOUS_STOP


def test_stale_and_expired_inputs_are_rejected_by_adapter_and_remain_auditable():
    stale = observation("stale", staleness=Staleness.AGED, age_ms=1.0)
    expired = observation("expired")
    result = ObservationReplayPipeline().replay([
        entry(stale, hypothesis="lane"),
        entry(expired, hypothesis="obstacle", expiry=1_400),
    ], at_ns=1_500)

    assert result.records[0].outcome is ReplayOutcome.RUNTIME_REJECTED
    assert "only FRESH" in result.records[0].adapter_error
    assert result.records[1].outcome is ReplayOutcome.ADAPTED
    items = {item.hypothesis_id: item for item in result.assessment.items}
    assert items["lane"].reason is DoubtReason.STALE
    assert items["lane"].evidence[0].state is EvidenceState.STALE
    assert items["obstacle"].reason is DoubtReason.EXPIRED
    assert items["obstacle"].evidence[0].state is EvidenceState.EXPIRED
    assert result.advisory.disposition is AdvisoryDisposition.CAUTIOUS_STOP


def test_invalid_and_unsupported_frame_inputs_are_rejected_without_bypassing_audit():
    invalid = observation("invalid", validity=Validity.INVALID, invalid_reason="checksum failed")
    local_map = observation("local-map", coordinate_frame=CoordinateFrame.LOCAL_MAP)
    result = ObservationReplayPipeline().replay([
        entry(invalid, hypothesis="pose"),
        entry(local_map, hypothesis="map"),
    ], at_ns=1_500)

    assert all(record.outcome is ReplayOutcome.RUNTIME_REJECTED for record in result.records)
    assert "only VALID" in result.records[0].adapter_error
    assert "local_map" in result.records[1].adapter_error
    items = {item.hypothesis_id: item for item in result.assessment.items}
    assert items["pose"].reason is DoubtReason.INVALID
    assert items["map"].reason is DoubtReason.UNKNOWN
    assert result.records[1].evidence_record is None
    assert result.advisory.disposition is AdvisoryDisposition.CAUTIOUS_STOP


def test_replay_results_are_repeatable_with_explicit_timestamps():
    inputs = [entry(observation("one")), entry(observation("two"), hypothesis="other", expiry=1_400)]
    first = ObservationReplayPipeline().replay(inputs, at_ns=1_500)
    second = ObservationReplayPipeline().replay(inputs, at_ns=1_500)

    assert [record.outcome for record in first.records] == [record.outcome for record in second.records]
    assert first.assessment == second.assessment
    assert first.advisory == second.advisory
    assert first.ledger_history == second.ledger_history
