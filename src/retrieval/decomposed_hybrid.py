"""Hybrid retrieval fusion across decomposed query result lists."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.schema import RetrievedDoc
from src.retrieval.hybrid import (
    DEFAULT_TITLE_BOOST_MAX,
    DEFAULT_TITLE_BOOST_WEIGHT,
    _content_tokens,
    reciprocal_rank_score,
)


@dataclass(slots=True)
class QueryRetrievedDocs:
    query_index: int
    query: str
    source: str
    docs: list[RetrievedDoc]


def fuse_decomposed_rrf_ranked_docs(
    *,
    query_results: list[QueryRetrievedDocs],
    original_query: str,
    final_top_k: int = 50,
    rrf_k: int = 60,
    title_boost_weight: float = DEFAULT_TITLE_BOOST_WEIGHT,
    title_boost_max: float = DEFAULT_TITLE_BOOST_MAX,
) -> list[RetrievedDoc]:
    """Fuse BM25/dense result lists from multiple decomposed queries."""

    candidates: dict[str, RetrievedDoc] = {}
    rrf_scores: dict[str, float] = {}

    for result in query_results:
        _add_query_result(
            candidates=candidates,
            rrf_scores=rrf_scores,
            result=result,
            rrf_k=rrf_k,
        )

    final_scores = dict(rrf_scores)
    _apply_title_boosts(
        query=original_query,
        candidates=candidates,
        scores=final_scores,
        title_boost_weight=title_boost_weight,
        title_boost_max=title_boost_max,
    )

    ranked_docs = sorted(
        candidates.values(),
        key=lambda doc: (
            -final_scores[doc.doc_id],
            _best_decomposed_source_rank(doc),
            doc.doc_id,
        ),
    )[:final_top_k]

    for rank, doc in enumerate(ranked_docs, start=1):
        doc.rank = rank
        doc.score = final_scores[doc.doc_id]
        doc.retrieval_source = "decomposed_hybrid"
        doc.metadata["rrf_score"] = rrf_scores[doc.doc_id]
        doc.metadata["hybrid_score"] = doc.score
        doc.metadata["decomposed_hybrid_score"] = doc.score

    return ranked_docs


def _add_query_result(
    *,
    candidates: dict[str, RetrievedDoc],
    rrf_scores: dict[str, float],
    result: QueryRetrievedDocs,
    rrf_k: int,
) -> None:
    for fallback_rank, doc in enumerate(result.docs, start=1):
        source_rank = doc.rank if doc.rank > 0 else fallback_rank
        if doc.doc_id not in candidates:
            candidates[doc.doc_id] = _copy_decomposed_doc(doc)
            rrf_scores[doc.doc_id] = 0.0

        candidate = candidates[doc.doc_id]
        source_key = f"q{result.query_index}_{result.source}"
        sources = candidate.metadata.setdefault("decomposition_sources", [])
        if source_key not in sources:
            sources.append(source_key)
        indexes = candidate.metadata.setdefault("decomposition_query_indexes", [])
        if result.query_index not in indexes:
            indexes.append(result.query_index)
        queries = candidate.metadata.setdefault("decomposition_queries", [])
        if result.query not in queries:
            queries.append(result.query)

        candidate.metadata[f"{source_key}_rank"] = source_rank
        candidate.metadata[f"{source_key}_score"] = doc.score
        rrf_scores[doc.doc_id] += reciprocal_rank_score(source_rank, rrf_k=rrf_k)


def _apply_title_boosts(
    *,
    query: str,
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


def _copy_decomposed_doc(doc: RetrievedDoc) -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc.doc_id,
        title=doc.title,
        text=doc.text,
        sentences=list(doc.sentences),
        metadata=dict(doc.metadata),
        score=0.0,
        rank=0,
        retrieval_source="decomposed_hybrid",
    )


def _best_decomposed_source_rank(doc: RetrievedDoc) -> int:
    ranks = [
        value
        for key, value in doc.metadata.items()
        if key.endswith("_rank") and isinstance(value, int)
    ]
    return min(ranks) if ranks else doc.rank
