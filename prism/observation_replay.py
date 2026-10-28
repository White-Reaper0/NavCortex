"""Deterministic, non-executing observation replay pipeline for P.R.I.S.M.

This is the narrow integration seam between an immutable observation and the
reasoning records built around it:

``Observation -> RuntimeAdapter -> BeliefLedger -> DoubtEngine -> advisory``.

Adapter rejection prevents creation of a runtime packet.  Invalid, stale,
expired, unavailable, and missing observations can still be recorded as audit
evidence because the Ledger already represents those states.  A fresh but
runtime-incompatible observation remains in the replay audit only: the Ledger
has no state for runtime incompatibility and must not silently mark it current.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .belief_ledger import BeliefLedger, EvidenceRecord, EvidenceRelation
from .doubt_engine import DoubtAssessment, DoubtEngine, RecommendationKind
from .observations import Observation, Staleness, Validity
from .runtime_adapter import ObservationAdapterError, RuntimeAdapter, RuntimeAdapterResult


class ReplayOutcome(str, Enum):
    """Whether a replay entry produced a runtime packet or was rejected there."""

    ADAPTED = "adapted"
    RUNTIME_REJECTED = "runtime_rejected"


class AdvisoryDisposition(str, Enum):
    """Advisory-only replay outcome; neither value controls a vehicle."""

    NO_ACTION = "no_action"
    CAUTIOUS_STOP = "cautious_stop"


@dataclass(frozen=True, slots=True)
class ReplayInput:
    """One explicit evidence claim to replay at a declared expiry time."""

    observation: Observation
    hypothesis_id: str
    relation: EvidenceRelation
    expires_at_ns: int
    clip_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """Audit-preserving result for one replay input.

    ``adapter_result`` is absent exactly when adapter validation rejected the
    observation.  ``evidence_record`` references the original observation,
    never a reconstructed packet.  It is absent when a runtime-incompatible
    but otherwise fresh observation would be misleadingly current in the
    backend-neutral Ledger; the immutable ``input`` remains its audit record.
    """

    input: ReplayInput
    outcome: ReplayOutcome
    evidence_record: Optional[EvidenceRecord]
    adapter_result: Optional[RuntimeAdapterResult]
    adapter_error: Optional[str]


@dataclass(frozen=True, slots=True)
class AdvisoryDecision:
    """A replay recommendation summary, explicitly not a movement decision."""

    disposition: AdvisoryDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Replay result carrying all component-level audit objects."""

    records: tuple[ReplayRecord, ...]
    assessment: DoubtAssessment
    advisory: AdvisoryDecision
    ledger_history: tuple


class ObservationReplayPipeline:
    """Replay observations through existing validation and reasoning interfaces."""

    def __init__(self, adapter: Optional[RuntimeAdapter] = None,
                 doubt_engine: Optional[DoubtEngine] = None) -> None:
        self._adapter = adapter if adapter is not None else RuntimeAdapter()
        self._doubt_engine = doubt_engine if doubt_engine is not None else DoubtEngine()

    def replay(self, entries: Iterable[ReplayInput], *, at_ns: int) -> ReplayResult:
        """Process inputs in supplied order using only explicit replay time.

        The index handed to the adapter is the deterministic input position.  No
        camera, clock, model, network, planner, supervisor, or control endpoint
        is accessed.
        """
        entries = tuple(entries)
        ledger = BeliefLedger()
        records: list[ReplayRecord] = []
        hypothesis_ids: list[str] = []
        for packet_index, entry in enumerate(entries):
            self._validate_input(entry)
            hypothesis_ids.append(entry.hypothesis_id)
            try:
                adapter_result = self._adapter.to_packet(
                    entry.observation, packet_index=packet_index, clip_id=entry.clip_id)
                outcome = ReplayOutcome.ADAPTED
                adapter_error = None
            except ObservationAdapterError as error:
                adapter_result = None
                outcome = ReplayOutcome.RUNTIME_REJECTED
                adapter_error = str(error)
            # The Ledger represents invalid, unavailable, missing, stale, and
            # expired evidence explicitly.  It does not represent "valid but
            # incompatible with this runtime".  Do not ledger such a rejected
            # observation as current evidence; retain it in ReplayRecord instead.
            evidence_record = None
            if outcome is ReplayOutcome.ADAPTED or self._ledger_can_represent_rejection(
                    entry.observation):
                evidence_record = ledger.record(
                    entry.hypothesis_id, entry.observation, entry.relation,
                    expires_at_ns=entry.expires_at_ns)
            records.append(ReplayRecord(
                input=entry,
                outcome=outcome,
                evidence_record=evidence_record,
                adapter_result=adapter_result,
                adapter_error=adapter_error,
            ))
        # This creates auditable expiry events while DoubtEngine uses the same
        # explicit timestamp for its state references.
        ledger.advance_to(at_ns)
        assessment = self._doubt_engine.assess(ledger, at_ns=at_ns,
                                               hypothesis_ids=hypothesis_ids)
        return ReplayResult(
            records=tuple(records),
            assessment=assessment,
            advisory=self._advisory_for(assessment),
            ledger_history=ledger.history,
        )

    @staticmethod
    def _validate_input(entry: ReplayInput) -> None:
        if not isinstance(entry, ReplayInput):
            raise ValueError("entries must contain ReplayInput values")
        if not isinstance(entry.observation, Observation):
            raise ValueError("ReplayInput.observation must be an Observation")
        if not isinstance(entry.hypothesis_id, str) or not entry.hypothesis_id.strip():
            raise ValueError("ReplayInput.hypothesis_id must be a non-empty string")
        if not isinstance(entry.relation, EvidenceRelation):
            raise ValueError("ReplayInput.relation must be an EvidenceRelation")
        if isinstance(entry.expires_at_ns, bool) or not isinstance(entry.expires_at_ns, int):
            raise ValueError("ReplayInput.expires_at_ns must be an integer nanosecond timestamp")

    @staticmethod
    def _ledger_can_represent_rejection(observation: Observation) -> bool:
        """Whether existing Ledger states already describe the rejection safely."""
        return (observation.validity is not Validity.VALID
                or observation.staleness is not Staleness.FRESH
                or not observation.evidence.has_evidence)

    @staticmethod
    def _advisory_for(assessment: DoubtAssessment) -> AdvisoryDecision:
        if any(item.kind is RecommendationKind.CAUTIOUS_STOP
               for item in assessment.recommendations):
            return AdvisoryDecision(
                AdvisoryDisposition.CAUTIOUS_STOP,
                "doubt assessment recommends a cautious stop; recommendation not executed",
            )
        return AdvisoryDecision(
            AdvisoryDisposition.NO_ACTION,
            "no advisory was generated; this is not a safe-to-move conclusion",
        )
