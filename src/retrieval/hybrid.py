"""Hybrid retrieval by fusing BM25 and dense ranked lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.data.schema import RetrievedDoc


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDoc]:
        ...


@dataclass(slots=True)
class HybridRetriever:
    bm25_retriever: Retriever
    dense_retriever: Retriever
    bm25_top_k: int = 50
    dense_top_k: int = 50
    final_top_k: int = 50
    rrf_k: int = 60

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedDoc]:
        bm25_docs = self.bm25_retriever.retrieve(query, top_k=self.bm25_top_k)
        dense_docs = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        return fuse_rrf_ranked_docs(
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            final_top_k=top_k or self.final_top_k,
            rrf_k=self.rrf_k,
        )


def fuse_rrf_ranked_docs(
    *,
    bm25_docs: list[RetrievedDoc],
    dense_docs: list[RetrievedDoc],
    final_top_k: int = 50,
    rrf_k: int = 60,
) -> list[RetrievedDoc]:
    """Fuse BM25 and dense ranked lists with reciprocal rank fusion."""

    candidates: dict[str, RetrievedDoc] = {}
    rrf_scores: dict[str, float] = {}

    _add_ranked_docs(
        candidates=candidates,
        rrf_scores=rrf_scores,
        docs=bm25_docs,
        source="bm25",
        rrf_k=rrf_k,
    )
    _add_ranked_docs(
        candidates=candidates,
        rrf_scores=rrf_scores,
        docs=dense_docs,
        source="dense",
        rrf_k=rrf_k,
    )

    ranked_docs = sorted(
        candidates.values(),
        key=lambda doc: (
            -rrf_scores[doc.doc_id],
            _best_source_rank(doc),
            doc.doc_id,
        ),
    )[:final_top_k]

    for rank, doc in enumerate(ranked_docs, start=1):
        doc.rank = rank
        doc.score = rrf_scores[doc.doc_id]
        doc.retrieval_source = "hybrid"
        doc.metadata["rrf_score"] = doc.score

    return ranked_docs


def _add_ranked_docs(
    *,
    candidates: dict[str, RetrievedDoc],
    rrf_scores: dict[str, float],
    docs: list[RetrievedDoc],
    source: str,
    rrf_k: int,
) -> None:
    for fallback_rank, doc in enumerate(docs, start=1):
        source_rank = doc.rank if doc.rank > 0 else fallback_rank
        if doc.doc_id not in candidates:
            candidates[doc.doc_id] = _copy_hybrid_doc(doc)
            rrf_scores[doc.doc_id] = 0.0

        candidate = candidates[doc.doc_id]
        sources = candidate.metadata.setdefault("hybrid_sources", [])
        if source not in sources:
            sources.append(source)
        candidate.metadata[f"{source}_rank"] = source_rank
        candidate.metadata[f"{source}_score"] = doc.score
        rrf_scores[doc.doc_id] += reciprocal_rank_score(source_rank, rrf_k=rrf_k)


def reciprocal_rank_score(rank: int, *, rrf_k: int = 60) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive.")
    return 1.0 / (rrf_k + rank)


def _copy_hybrid_doc(doc: RetrievedDoc) -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc.doc_id,
        title=doc.title,
        text=doc.text,
        sentences=list(doc.sentences),
        metadata=dict(doc.metadata),
        score=0.0,
        rank=0,
        retrieval_source="hybrid",
    )


def _best_source_rank(doc: RetrievedDoc) -> int:
    ranks = [
        value
        for key, value in doc.metadata.items()
        if key.endswith("_rank") and isinstance(value, int)
    ]
    return min(ranks) if ranks else doc.rank
