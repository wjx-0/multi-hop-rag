"""Hybrid retrieval by fusing BM25 and dense ranked lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.data.schema import RetrievedDoc
from src.utils.text import simple_tokenize


DEFAULT_TITLE_BOOST_WEIGHT = 0.0005
DEFAULT_TITLE_BOOST_MAX = 0.0015

TITLE_BOOST_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


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
    title_boost_weight: float = DEFAULT_TITLE_BOOST_WEIGHT

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedDoc]:
        bm25_docs = self.bm25_retriever.retrieve(query, top_k=self.bm25_top_k)
        dense_docs = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        return fuse_rrf_ranked_docs(
            query=query,
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            final_top_k=top_k or self.final_top_k,
            rrf_k=self.rrf_k,
            title_boost_weight=self.title_boost_weight,
        )


def fuse_rrf_ranked_docs(
    *,
    bm25_docs: list[RetrievedDoc],
    dense_docs: list[RetrievedDoc],
    final_top_k: int = 50,
    rrf_k: int = 60,
    query: str | None = None,
    title_boost_weight: float = 0.0,
    title_boost_max: float = DEFAULT_TITLE_BOOST_MAX,
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

    final_scores = dict(rrf_scores)
    _apply_title_boosts(
        query=query,
        candidates=candidates,
        scores=final_scores,
        title_boost_weight=title_boost_weight,
        title_boost_max=title_boost_max,
    )

    ranked_docs = sorted(
        candidates.values(),
        key=lambda doc: (
            -final_scores[doc.doc_id],
            _best_source_rank(doc),
            doc.doc_id,
        ),
    )[:final_top_k]

    for rank, doc in enumerate(ranked_docs, start=1):
        doc.rank = rank
        doc.score = final_scores[doc.doc_id]
        doc.retrieval_source = "hybrid"
        doc.metadata["rrf_score"] = rrf_scores[doc.doc_id]
        doc.metadata["hybrid_score"] = doc.score

    return ranked_docs


def _apply_title_boosts(
    *,
    query: str | None,
    candidates: dict[str, RetrievedDoc],
    scores: dict[str, float],
    title_boost_weight: float,
    title_boost_max: float,
) -> None:
    if not query or title_boost_weight <= 0 or title_boost_max <= 0:
        return

    query_tokens = _content_tokens(query)
    if not query_tokens:
        return

    for doc_id, doc in candidates.items():
        overlap = sorted(query_tokens & _content_tokens(doc.title))
        if not overlap:
            continue
        boost = min(title_boost_weight * len(overlap), title_boost_max)
        if boost <= 0:
            continue
        doc.metadata["score_before_title_boost"] = scores[doc_id]
        doc.metadata["title_boost"] = boost
        doc.metadata["title_overlap_tokens"] = overlap
        scores[doc_id] += boost


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in simple_tokenize(text)
        if len(token) > 1 and token not in TITLE_BOOST_STOPWORDS
    }


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
