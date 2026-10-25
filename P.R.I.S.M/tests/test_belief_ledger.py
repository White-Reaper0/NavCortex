"""Focused tests for the backend-neutral P.R.I.S.M. Belief Ledger."""
import numpy as np
import pytest

from prism.belief_ledger import (
    BeliefLedger,
    EvidenceRelation,
    EvidenceState,
    HypothesisStatus,
    LedgerEventKind,
)
from prism.observations import (
    ConfidenceKind,
    CoordinateFrame,
    EvidencePayload,
    Observation,
    Staleness,
    Validity,
)


def observation(observation_id: str, **changes) -> Observation:
    base = dict(
        observation_id=observation_id,
        source="replay:clip_01",
        captured_at_ns=100,
        received_at_ns=120,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.VALID,
        confidence=0.75,
        confidence_kind=ConfidenceKind.REPORTED,
        provenance={"clip": "clip_01", "producer": {"frame": 3}},
        evidence=EvidencePayload(metric_depth_m=np.ones((2, 2), np.float32)),
    )
    base.update(changes)
    return Observation(**base)


def test_record_preserves_observation_provenance_and_optional_confidence():
    ledger = BeliefLedger()
    item = ledger.record("clear-path", observation("obs-1"), EvidenceRelation.SUPPORTS,
                         expires_at_ns=200)

    assert item.observation_id == "obs-1"
    assert item.source == "replay:clip_01"
    assert item.captured_at_ns == 100 and item.received_at_ns == 120
    assert item.coordinate_frame is CoordinateFrame.CAMERA
    assert item.validity is Validity.VALID
    assert item.confidence == pytest.approx(0.75)
    assert item.provenance["producer"]["frame"] == 3
    with pytest.raises(TypeError):
        item.provenance["clip"] = "changed"
    assert ledger.history[0].kind is LedgerEventKind.RECORDED
    assert ledger.history[0].at_ns == 120


def test_competing_hypotheses_and_conflicting_evidence_remain_explicit():
    ledger = BeliefLedger()
    ledger.record("path-clear", observation("support"), EvidenceRelation.SUPPORTS, expires_at_ns=200)
    ledger.record("path-clear", observation("contradict"), EvidenceRelation.CONTRADICTS,
                  expires_at_ns=200)
    ledger.record("path-blocked", observation("blocked"), EvidenceRelation.SUPPORTS,
                  expires_at_ns=200)

    assert ledger.hypotheses() == ("path-blocked", "path-clear")
    assert ledger.hypothesis_status("path-clear", at_ns=150) is HypothesisStatus.CONTRADICTORY
    assert ledger.hypothesis_status("path-blocked", at_ns=150) is HypothesisStatus.SUPPORTED
    assert [item.observation_id for item in ledger.evidence_for("path-clear")] == [
        "support", "contradict"]


def test_expiry_is_time_indexed_and_audited_without_remaining_current():
    ledger = BeliefLedger()
    item = ledger.record("dock-visible", observation("dock"), EvidenceRelation.SUPPORTS,
                         expires_at_ns=200)

    assert ledger.evidence_state(item, at_ns=199) is EvidenceState.CURRENT
    assert ledger.hypothesis_status("dock-visible", at_ns=199) is HypothesisStatus.SUPPORTED
    assert ledger.evidence_state(item, at_ns=200) is EvidenceState.EXPIRED
    assert ledger.hypothesis_status("dock-visible", at_ns=200) is HypothesisStatus.EXPIRED
    assert ledger.advance_to(200)[0].kind is LedgerEventKind.EXPIRED
    assert ledger.advance_to(250) == ()
    with pytest.raises(ValueError, match="backwards"):
        ledger.advance_to(199)


def test_stale_invalid_unavailable_and_missing_data_cannot_become_support():
    ledger = BeliefLedger()
    stale = observation("stale", staleness=Staleness.AGED, age_ms=5.0)
    invalid = observation("invalid", validity=Validity.INVALID, invalid_reason="checksum failed")
    unavailable = Observation(
        observation_id="unavailable", source="camera", captured_at_ns=100,
        coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.UNAVAILABLE,
    )
    missing = observation("missing", confidence=None, confidence_kind=None, evidence=EvidencePayload())

    stale_item = ledger.record("stale-hypothesis", stale, EvidenceRelation.SUPPORTS, expires_at_ns=200)
    invalid_item = ledger.record("invalid-hypothesis", invalid, EvidenceRelation.SUPPORTS, expires_at_ns=200)
    unavailable_item = ledger.record("unavailable-hypothesis", unavailable, EvidenceRelation.SUPPORTS,
                                     expires_at_ns=200)
    missing_item = ledger.record("missing-hypothesis", missing, EvidenceRelation.SUPPORTS, expires_at_ns=200)

    assert ledger.evidence_state(stale_item, at_ns=150) is EvidenceState.STALE
    assert ledger.evidence_state(invalid_item, at_ns=150) is EvidenceState.INVALID
    assert ledger.evidence_state(unavailable_item, at_ns=150) is EvidenceState.UNAVAILABLE
    assert ledger.evidence_state(missing_item, at_ns=150) is EvidenceState.MISSING
    assert ledger.hypothesis_status("stale-hypothesis", at_ns=150) is HypothesisStatus.STALE
    assert ledger.hypothesis_status("invalid-hypothesis", at_ns=150) is HypothesisStatus.INVALID
    assert ledger.hypothesis_status("unavailable-hypothesis", at_ns=150) is HypothesisStatus.UNKNOWN
    assert ledger.hypothesis_status("missing-hypothesis", at_ns=150) is HypothesisStatus.UNKNOWN


def test_queries_are_deterministic_and_reject_ambiguous_record_updates():
    ledger = BeliefLedger()
    first = ledger.record("h", observation("one"), EvidenceRelation.SUPPORTS, expires_at_ns=500)
    second = ledger.record("h", observation("two"), EvidenceRelation.SUPPORTS, expires_at_ns=400)

    assert ledger.evidence_for("h") == (first, second)
    assert ledger.hypothesis_status("h", at_ns=300) is HypothesisStatus.SUPPORTED
    assert ledger.hypothesis_status("h", at_ns=300) is HypothesisStatus.SUPPORTED
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record("h", observation("one"), EvidenceRelation.CONTRADICTS, expires_at_ns=500)
    with pytest.raises(ValueError, match="cannot precede"):
        ledger.record("late", observation("late"), EvidenceRelation.SUPPORTS, expires_at_ns=99)
