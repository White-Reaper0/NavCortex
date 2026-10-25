"""Backend-neutral doubt assessment and conservative advisory policy.

This module evaluates a :mod:`prism.belief_ledger` at an explicitly supplied
timestamp.  Its outputs are records for audit and replay, not commands: no
vehicle, planner, supervisor, or perception backend is called from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .belief_ledger import (
    BeliefLedger,
    EvidenceRecord,
    EvidenceState,
    HypothesisStatus,
)


class DoubtReason(str, Enum):
    """Why a hypothesis cannot be treated as currently well-supported."""

    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


class RecommendationKind(str, Enum):
    """Bounded advisory actions; these values have no execution behavior."""

    REQUEST_ONE_OBSERVATION = "request_one_observation"
    WAIT_FOR_ONE_FRESH_FRAME = "wait_for_one_fresh_frame"
    CAUTIOUS_STOP = "cautious_stop"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """A compact, immutable reference to ledger evidence used in an assessment."""

    evidence_sequence: int
    observation_id: str
    source: str
    state: EvidenceState


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """A stated gap, not a claim that the required evidence exists."""

    hypothesis_id: str
    requirement: str


@dataclass(frozen=True, slots=True)
class DoubtItem:
    """One hypothesis that requires more evidence before relying on it."""

    hypothesis_id: str
    status: HypothesisStatus
    reason: DoubtReason
    evidence: tuple[EvidenceReference, ...]
    missing_requirements: tuple[EvidenceRequirement, ...]


@dataclass(frozen=True, slots=True)
class Recommendation:
    """An advisory recommendation only; producing one does not execute it."""

    hypothesis_id: str
    kind: RecommendationKind
    reason: DoubtReason


@dataclass(frozen=True, slots=True)
class DoubtAssessment:
    """Replayable assessment at one monotonic timestamp."""

    at_ns: int
    items: tuple[DoubtItem, ...]
    recommendations: tuple[Recommendation, ...]


class ConservativeDecisionPolicy:
    """Map doubt to bounded, non-executing conservative recommendations.

    Even an assessment without doubt items is *not* a clearance or a permission
    to move.  This policy only says when present evidence is insufficient.
    """

    def recommendations_for(self, item: DoubtItem) -> tuple[Recommendation, ...]:
        if item.reason is DoubtReason.STALE:
            return (
                Recommendation(item.hypothesis_id, RecommendationKind.WAIT_FOR_ONE_FRESH_FRAME,
                               item.reason),
                Recommendation(item.hypothesis_id, RecommendationKind.CAUTIOUS_STOP, item.reason),
            )
        if item.reason is DoubtReason.EXPIRED:
            return (
                Recommendation(item.hypothesis_id, RecommendationKind.REQUEST_ONE_OBSERVATION,
                               item.reason),
                Recommendation(item.hypothesis_id, RecommendationKind.CAUTIOUS_STOP, item.reason),
            )
        # Unknown, invalid, and contradictory evidence need another observation;
        # none of them is silently treated as a safe condition.
        return (
            Recommendation(item.hypothesis_id, RecommendationKind.REQUEST_ONE_OBSERVATION,
                           item.reason),
            Recommendation(item.hypothesis_id, RecommendationKind.CAUTIOUS_STOP, item.reason),
        )


class DoubtEngine:
    """Assess uncertainty in ledger hypotheses without selecting a conclusion."""

    def __init__(self, policy: Optional[ConservativeDecisionPolicy] = None) -> None:
        self._policy = policy if policy is not None else ConservativeDecisionPolicy()

    def assess(self, ledger: BeliefLedger, *, at_ns: int,
               hypothesis_ids: Optional[Iterable[str]] = None) -> DoubtAssessment:
        """Return a deterministic advisory assessment for ``at_ns``.

        When ``hypothesis_ids`` is omitted, every known hypothesis is assessed.
        Supplying an otherwise unknown identity explicitly yields an ``UNKNOWN``
        item, making missing evidence visible rather than implying safety.
        """
        if not isinstance(ledger, BeliefLedger):
            raise ValueError("ledger must be a BeliefLedger")
        # Delegate timestamp validation to the Ledger's existing contract.
        known = ledger.hypotheses()
        if hypothesis_ids is None:
            ids = known
        else:
            ids = self._normalise_hypothesis_ids(hypothesis_ids)
        items: list[DoubtItem] = []
        recommendations: list[Recommendation] = []
        for hypothesis_id in ids:
            status = ledger.hypothesis_status(hypothesis_id, at_ns=at_ns)
            reason = self._reason_for(status)
            if reason is None:
                continue
            records = ledger.evidence_for(hypothesis_id)
            item = DoubtItem(
                hypothesis_id=hypothesis_id,
                status=status,
                reason=reason,
                evidence=tuple(self._reference(ledger, record, at_ns) for record in records),
                missing_requirements=(EvidenceRequirement(
                    hypothesis_id, self._requirement_for(reason)),),
            )
            items.append(item)
            recommendations.extend(self._policy.recommendations_for(item))
        return DoubtAssessment(at_ns=at_ns, items=tuple(items),
                               recommendations=tuple(recommendations))

    @staticmethod
    def _normalise_hypothesis_ids(hypothesis_ids: Iterable[str]) -> tuple[str, ...]:
        ids = []
        for hypothesis_id in hypothesis_ids:
            if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
                raise ValueError("hypothesis_ids must contain non-empty strings")
            ids.append(hypothesis_id)
        return tuple(sorted(set(ids)))

    @staticmethod
    def _reference(ledger: BeliefLedger, record: EvidenceRecord, at_ns: int) -> EvidenceReference:
        return EvidenceReference(
            evidence_sequence=record.sequence,
            observation_id=record.observation_id,
            source=record.source,
            state=ledger.evidence_state(record, at_ns=at_ns),
        )

    @staticmethod
    def _reason_for(status: HypothesisStatus) -> Optional[DoubtReason]:
        reasons = {
            HypothesisStatus.UNKNOWN: DoubtReason.UNKNOWN,
            HypothesisStatus.CONTRADICTORY: DoubtReason.CONTRADICTORY,
            HypothesisStatus.STALE: DoubtReason.STALE,
            HypothesisStatus.EXPIRED: DoubtReason.EXPIRED,
            HypothesisStatus.INVALID: DoubtReason.INVALID,
        }
        return reasons.get(status)

    @staticmethod
    def _requirement_for(reason: DoubtReason) -> str:
        requirements = {
            DoubtReason.UNKNOWN: "fresh valid evidence",
            DoubtReason.CONTRADICTORY: "fresh valid evidence that resolves the conflict",
            DoubtReason.STALE: "fresh replacement evidence",
            DoubtReason.EXPIRED: "fresh replacement evidence",
            DoubtReason.INVALID: "fresh valid replacement evidence",
        }
        return requirements[reason]
