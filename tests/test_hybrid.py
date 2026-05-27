import pytest

from src.data.schema import RetrievedDoc
from src.retrieval.hybrid import HybridRetriever, fuse_rrf_ranked_docs, reciprocal_rank_score


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def retrieve(self, query, *, top_k=5):
        self.calls.append((query, top_k))
        return self.docs[:top_k]


def test_reciprocal_rank_score_requires_positive_rank():
    assert reciprocal_rank_score(1, rrf_k=60) == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        reciprocal_rank_score(0)


def test_rrf_fusion_deduplicates_and_records_sources():
    bm25_docs = [
        _doc("d1", rank=1, source="bm25", score=10.0),
        _doc("d2", rank=2, source="bm25", score=8.0),
    ]
    dense_docs = [
        _doc("d2", rank=1, source="dense", score=0.9),
        _doc("d3", rank=2, source="dense", score=0.8),
    ]

    results = fuse_rrf_ranked_docs(
        bm25_docs=bm25_docs,
        dense_docs=dense_docs,
        final_top_k=10,
        rrf_k=60,
    )

    assert [doc.doc_id for doc in results] == ["d2", "d1", "d3"]
    assert len(results) == 3
    assert results[0].retrieval_source == "hybrid"
    assert results[0].metadata["hybrid_sources"] == ["bm25", "dense"]
    assert results[0].metadata["bm25_rank"] == 2
    assert results[0].metadata["dense_rank"] == 1
    assert results[0].metadata["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_fusion_keeps_bm25_only_and_dense_only_docs():
    results = fuse_rrf_ranked_docs(
        bm25_docs=[_doc("bm25-only", rank=1, source="bm25")],
        dense_docs=[_doc("dense-only", rank=1, source="dense")],
        final_top_k=10,
        rrf_k=60,
    )

    by_id = {doc.doc_id: doc for doc in results}
    assert by_id["bm25-only"].metadata["hybrid_sources"] == ["bm25"]
    assert by_id["dense-only"].metadata["hybrid_sources"] == ["dense"]


def test_hybrid_retriever_calls_both_retrievers_with_configured_top_k():
    bm25 = FakeRetriever([_doc("d1", rank=1, source="bm25")])
    dense = FakeRetriever([_doc("d2", rank=1, source="dense")])
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_top_k=7,
        dense_top_k=11,
        final_top_k=2,
    )

    results = retriever.retrieve("question")

    assert bm25.calls == [("question", 7)]
    assert dense.calls == [("question", 11)]
    assert len(results) == 2
    assert all(doc.retrieval_source == "hybrid" for doc in results)


def _doc(doc_id, *, rank, source, score=1.0):
    return RetrievedDoc(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Text {doc_id}",
        sentences=[f"Text {doc_id}"],
        metadata={"original": source},
        score=score,
        rank=rank,
        retrieval_source=source,
    )
