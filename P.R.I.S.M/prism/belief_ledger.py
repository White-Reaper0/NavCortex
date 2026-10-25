"""Backend-neutral, append-only evidence ledger for P.R.I.S.M.

The ledger records validated :class:`~prism.observations.Observation` objects
without translating their measurements or assigning a confidence that was not
reported.  A hypothesis is assessed independently from every other hypothesis;
the ledger deliberately has no winner-selection policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .observations import Observation, Staleness, Validity


class EvidenceRelation(str, Enum):
    """How a producer says an observation bears on one hypothesis."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class EvidenceState(str, Enum):
    """Usability of one entry at a specified monotonic timestamp."""

    CURRENT = "current"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    MISSING = "missing"


class HypothesisStatus(str, Enum):
    """Status of a hypothesis at one explicitly supplied timestamp."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONTRADICTORY = "contradictory"
    STALE = "stale"
    EXPIRED = "expired"
    INVALID = "invalid"


class LedgerEventKind(str, Enum):
    RECORDED = "recorded"
    EXPIRED = "expired"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer nanosecond timestamp")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One immutable, append-only association between evidence and a hypothesis.

    ``observation`` remains the authoritative source for measurements,
    provenance, timestamps, frame, validity, and optional confidence.  The
    ledger intentionally does not copy arrays or invent a summary measurement.
    """

    sequence: int
    hypothesis_id: str
    relation: EvidenceRelation
    observation: Observation
    expires_at_ns: int

    @property
    def observation_id(self) -> str:
        return self.observation.observation_id

    @property
    def source(self) -> str:
        return self.observation.source

    @property
    def captured_at_ns(self) -> int:
        return self.observation.captured_at_ns

    @property
    def received_at_ns(self) -> Optional[int]:
        return self.observation.received_at_ns

    @property
    def coordinate_frame(self):
        return self.observation.coordinate_frame

    @property
    def validity(self) -> Validity:
        return self.observation.validity

    @property
    def confidence(self) -> Optional[float]:
        return self.observation.confidence

    @property
    def confidence_kind(self):
        return self.observation.confidence_kind

    @property
    def provenance(self):
        return self.observation.provenance


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """An immutable audit event.  Events are ordered by ledger sequence."""

    sequence: int
    kind: LedgerEventKind
    evidence_sequence: int
    hypothesis_id: str
    observation_id: str
    at_ns: int


class BeliefLedger:
    """Append-only evidence ledger with deterministic, time-indexed queries.

    Callers must provide ``expires_at_ns`` while recording and ``at_ns`` while
    querying.  This avoids reliance on wall-clock time and makes replay stable.
    An observation with missing payload, invalid validity, unavailable validity,
    or non-fresh staleness is retained for audit but cannot support or contradict
    a hypothesis.
    """

    def __init__(self) -> None:
        self._records: list[EvidenceRecord] = []
        self._events: list[LedgerEvent] = []
        self._record_keys: set[tuple[str, str]] = set()
        self._expired_record_sequences: set[int] = set()
        self._last_advanced_ns: Optional[int] = None

    @property
    def history(self) -> tuple[LedgerEvent, ...]:
        """Immutable ordered audit history of records and ledger expiry events."""
        return tuple(self._events)

    def record(self, hypothesis_id: str, observation: Observation,
               relation: EvidenceRelation, *, expires_at_ns: int) -> EvidenceRecord:
        """Append one observation without changing or interpreting its payload."""
        hypothesis_id = _nonempty(hypothesis_id, "hypothesis_id")
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation")
        if not isinstance(relation, EvidenceRelation):
            raise ValueError("relation must be an EvidenceRelation")
        expires_at_ns = _timestamp(expires_at_ns, "expires_at_ns")
        if expires_at_ns < observation.captured_at_ns:
            raise ValueError("expires_at_ns cannot precede captured_at_ns")
        key = (hypothesis_id, observation.observation_id)
        if key in self._record_keys:
            raise ValueError("observation is already recorded for this hypothesis")

        record = EvidenceRecord(
            sequence=len(self._records),
            hypothesis_id=hypothesis_id,
            relation=relation,
            observation=observation,
            expires_at_ns=expires_at_ns,
        )
        self._records.append(record)
        self._record_keys.add(key)
        self._append_event(LedgerEventKind.RECORDED, record, observation.received_at_ns
                           if observation.received_at_ns is not None
                           else observation.captured_at_ns)
        return record

    def evidence_for(self, hypothesis_id: str) -> tuple[EvidenceRecord, ...]:
        """Return evidence in append order, independent of query time."""
        hypothesis_id = _nonempty(hypothesis_id, "hypothesis_id")
        return tuple(record for record in self._records
                     if record.hypothesis_id == hypothesis_id)

    def hypotheses(self) -> tuple[str, ...]:
        """Return hypothesis identities in stable lexical order; no winner is chosen."""
        return tuple(sorted({record.hypothesis_id for record in self._records}))

    def evidence_state(self, record: EvidenceRecord, *, at_ns: int) -> EvidenceState:
        """Evaluate one record without mutating ledger state."""
        if not isinstance(record, EvidenceRecord):
            raise ValueError("record must be an EvidenceRecord")
        at_ns = _timestamp(at_ns, "at_ns")
        observation = record.observation
        if observation.validity is Validity.INVALID:
            return EvidenceState.INVALID
        if observation.validity is Validity.UNAVAILABLE:
            return EvidenceState.UNAVAILABLE
        if not observation.evidence.has_evidence:
            return EvidenceState.MISSING
        if observation.staleness is Staleness.EXPIRED or at_ns >= record.expires_at_ns:
            return EvidenceState.EXPIRED
        if observation.staleness is Staleness.AGED:
            return EvidenceState.STALE
        return EvidenceState.CURRENT

    def hypothesis_status(self, hypothesis_id: str, *, at_ns: int) -> HypothesisStatus:
        """Return the deterministic status of one hypothesis at ``at_ns``."""
        records = self.evidence_for(hypothesis_id)
        at_ns = _timestamp(at_ns, "at_ns")
        states = [(record, self.evidence_state(record, at_ns=at_ns)) for record in records]
        current_relations = {record.relation for record, state in states
                             if state is EvidenceState.CURRENT}
        if current_relations == {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}:
            return HypothesisStatus.CONTRADICTORY
        if current_relations == {EvidenceRelation.SUPPORTS}:
            return HypothesisStatus.SUPPORTED
        if current_relations == {EvidenceRelation.CONTRADICTS}:
            return HypothesisStatus.CONTRADICTED
        if any(state is EvidenceState.STALE for _, state in states):
            return HypothesisStatus.STALE
        if any(state is EvidenceState.EXPIRED for _, state in states):
            return HypothesisStatus.EXPIRED
        if any(state is EvidenceState.INVALID for _, state in states):
            return HypothesisStatus.INVALID
        return HypothesisStatus.UNKNOWN

    def advance_to(self, at_ns: int) -> tuple[LedgerEvent, ...]:
        """Record expiry events through ``at_ns`` and return newly added events.

        Advancing time is optional for status queries, but gives replays a stable,
        auditable event for every fresh, usable record whose declared expiry has
        been reached.  Time never moves backwards in the mutable audit log.
        """
        at_ns = _timestamp(at_ns, "at_ns")
        if self._last_advanced_ns is not None and at_ns < self._last_advanced_ns:
            raise ValueError("advance_to cannot move backwards")
        start = len(self._events)
        for record in self._records:
            if record.sequence in self._expired_record_sequences:
                continue
            if self.evidence_state(record, at_ns=record.expires_at_ns) is EvidenceState.EXPIRED \
                    and record.observation.validity is Validity.VALID \
                    and record.observation.staleness is Staleness.FRESH \
                    and record.observation.evidence.has_evidence \
                    and record.expires_at_ns <= at_ns:
                self._expired_record_sequences.add(record.sequence)
                self._append_event(LedgerEventKind.EXPIRED, record, record.expires_at_ns)
        self._last_advanced_ns = at_ns
        return tuple(self._events[start:])

    def _append_event(self, kind: LedgerEventKind, record: EvidenceRecord, at_ns: int) -> None:
        self._events.append(LedgerEvent(
            sequence=len(self._events),
            kind=kind,
            evidence_sequence=record.sequence,
            hypothesis_id=record.hypothesis_id,
            observation_id=record.observation_id,
            at_ns=at_ns,
        ))
