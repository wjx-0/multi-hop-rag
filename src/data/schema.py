"""Shared data schemas for the Phase 1 HotpotQA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A paragraph-level chunk used by retrieval."""

    doc_id: str
    title: str
    text: str
    sentences: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            text=data["text"],
            sentences=list(data.get("sentences", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class RetrievedDoc(Document):
    """A retrieved document with ranking metadata."""

    score: float = 0.0
    rank: int = 0
    retrieval_source: str = "unknown"

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        score: float,
        rank: int,
        retrieval_source: str,
    ) -> "RetrievedDoc":
        return cls(
            doc_id=document.doc_id,
            title=document.title,
            text=document.text,
            sentences=list(document.sentences),
            metadata=dict(document.metadata),
            score=float(score),
            rank=int(rank),
            retrieval_source=retrieval_source,
        )


@dataclass(slots=True)
class HotpotQASample:
    """A normalized HotpotQA sample."""

    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: list[list[Any]]
    context: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    """Common output record for pipeline predictions."""

    id: str
    question: str
    gold_answer: str
    pred_answer: str
    retrieved_docs: list[dict[str, Any]]
    pred_citations: list[dict[str, Any]]
    metrics: dict[str, Any]
    cost: dict[str, Any]
    route_initial: dict[str, Any] | None = None
    route_final: str | None = None
    was_upgraded: bool = False
    rewritten_docs: list[dict[str, Any]] = field(default_factory=list)
    agent_outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
