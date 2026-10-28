"""Focused tests for P.R.I.S.M.'s non-executing Doubt Engine."""
import numpy as np

from prism.belief_ledger import BeliefLedger, EvidenceRelation, EvidenceState, HypothesisStatus
from prism.doubt_engine import DoubtEngine, DoubtReason, RecommendationKind
from prism.observations import CoordinateFrame, EvidencePayload, Observation, Staleness, Validity


def observation(observation_id: str, **changes) -> Observation:
    base = dict(
        observation_id=observation_id,
        source="replay:clip_01",
        captured_at_ns=100,
        received_at_ns=120,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.VALID,
        provenance={"frame": 3},
        evidence=EvidencePayload(metric_depth_m=np.ones((2, 2), np.float32)),
    )
    base.update(changes)
    return Observation(**base)


def test_conflicting_evidence_is_auditable_and_recommends_bounded_caution():
    ledger = BeliefLedger()
    ledger.record("path-clear", observation("support"), EvidenceRelation.SUPPORTS, expires_at_ns=200)
    ledger.record("path-clear", observation("contradict"), EvidenceRelation.CONTRADICTS,
                  expires_at_ns=200)

    assessment = DoubtEngine().assess(ledger, at_ns=150)

    assert len(assessment.items) == 1
    item = assessment.items[0]
    assert item.status is HypothesisStatus.CONTRADICTORY
    assert item.reason is DoubtReason.CONTRADICTORY
    assert [ref.observation_id for ref in item.evidence] == ["support", "contradict"]
    assert item.missing_requirements[0].requirement == "fresh valid evidence that resolves the conflict"
    assert [recommendation.kind for recommendation in assessment.recommendations] == [
        RecommendationKind.REQUEST_ONE_OBSERVATION, RecommendationKind.CAUTIOUS_STOP]


def test_missing_or_unknown_evidence_is_not_treated_as_safe():
    ledger = BeliefLedger()
    ledger.record("visibility", observation("missing", evidence=EvidencePayload()),
                  EvidenceRelation.SUPPORTS, expires_at_ns=200)

    assessment = DoubtEngine().assess(ledger, at_ns=150, hypothesis_ids=["new-hypothesis", "visibility"])

    assert [item.hypothesis_id for item in assessment.items] == ["new-hypothesis", "visibility"]
    assert all(item.reason is DoubtReason.UNKNOWN for item in assessment.items)
    assert assessment.items[0].evidence == ()
    assert assessment.items[1].evidence[0].state is EvidenceState.MISSING
    assert all(recommendation.kind is not RecommendationKind.WAIT_FOR_ONE_FRESH_FRAME
               for recommendation in assessment.recommendations)


def test_stale_and_expired_evidence_request_fresh_data_with_references():
    ledger = BeliefLedger()
    ledger.record("lane", observation("stale", staleness=Staleness.AGED, age_ms=5.0),
                  EvidenceRelation.SUPPORTS, expires_at_ns=200)
    ledger.record("obstacle", observation("expired"), EvidenceRelation.SUPPORTS, expires_at_ns=140)

    assessment = DoubtEngine().assess(ledger, at_ns=150)
    items = {item.hypothesis_id: item for item in assessment.items}
    assert items["lane"].reason is DoubtReason.STALE
    assert items["lane"].evidence[0].state is EvidenceState.STALE
    assert items["obstacle"].reason is DoubtReason.EXPIRED
    assert items["obstacle"].evidence[0].state is EvidenceState.EXPIRED
    assert (RecommendationKind.WAIT_FOR_ONE_FRESH_FRAME in
            [r.kind for r in assessment.recommendations if r.hypothesis_id == "lane"])
    assert (RecommendationKind.REQUEST_ONE_OBSERVATION in
            [r.kind for r in assessment.recommendations if r.hypothesis_id == "obstacle"])


def test_invalid_evidence_is_exposed_and_cannot_supply_a_conclusion():
    ledger = BeliefLedger()
    invalid = observation("invalid", validity=Validity.INVALID, invalid_reason="checksum failed")
    ledger.record("pose", invalid, EvidenceRelation.SUPPORTS, expires_at_ns=200)

    assessment = DoubtEngine().assess(ledger, at_ns=150)

    assert assessment.items[0].status is HypothesisStatus.INVALID
    assert assessment.items[0].reason is DoubtReason.INVALID
    assert assessment.items[0].evidence[0].state is EvidenceState.INVALID
    assert assessment.recommendations[-1].kind is RecommendationKind.CAUTIOUS_STOP


def test_assessment_order_and_contents_are_deterministic():
    ledger = BeliefLedger()
    ledger.record("zeta", observation("zeta"), EvidenceRelation.SUPPORTS, expires_at_ns=120)
    ledger.record("alpha", observation("alpha", staleness=Staleness.AGED, age_ms=1.0),
                  EvidenceRelation.SUPPORTS, expires_at_ns=200)
    engine = DoubtEngine()

    first = engine.assess(ledger, at_ns=150)
    second = engine.assess(ledger, at_ns=150)

    assert first == second
    assert [item.hypothesis_id for item in first.items] == ["alpha", "zeta"]
    assert [item.reason for item in first.items] == [DoubtReason.STALE, DoubtReason.EXPIRED]
