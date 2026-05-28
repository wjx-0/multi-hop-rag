from src.data.schema import RetrievedDoc
from src.retrieval.decomposed_hybrid import QueryRetrievedDocs, fuse_decomposed_rrf_ranked_docs


def test_fuse_decomposed_rrf_accumulates_scores_across_queries_and_sources():
    results = [
        QueryRetrievedDocs(
            query_index=0,
            query="Original question?",
            source="bm25",
            docs=[_doc("d1", "Title One", rank=1), _doc("d2", "Title Two", rank=2)],
        ),
        QueryRetrievedDocs(
            query_index=1,
            query="Bridge entity",
            source="dense",
            docs=[_doc("d2", "Title Two", rank=1), _doc("d3", "Title Three", rank=2)],
        ),
    ]

    docs = fuse_decomposed_rrf_ranked_docs(
        query_results=results,
        original_query="Original question?",
        final_top_k=3,
        rrf_k=60,
        title_boost_weight=0.0,
    )

    assert [doc.doc_id for doc in docs] == ["d2", "d1", "d3"]
    assert [doc.rank for doc in docs] == [1, 2, 3]
    assert all(doc.retrieval_source == "decomposed_hybrid" for doc in docs)
    assert docs[0].metadata["decomposition_query_indexes"] == [0, 1]
    assert docs[0].metadata["q0_bm25_rank"] == 2
    assert docs[0].metadata["q1_dense_rank"] == 1
    assert docs[0].metadata["decomposed_hybrid_score"] == docs[0].score


def _doc(doc_id, title, *, rank):
    return RetrievedDoc(
        doc_id=doc_id,
        title=title,
        text=f"Text {doc_id}",
        sentences=[f"Text {doc_id}"],
        metadata={},
        score=1.0 / rank,
        rank=rank,
        retrieval_source="test",
    )
