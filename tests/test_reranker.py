import pytest

from src.data.schema import RetrievedDoc
from src.retrieval.reranker import (
    DashScopeReranker,
    dashscope_response_to_reranked_docs,
    fallback_rerank_docs,
    reranker_document_text,
)


def test_dashscope_response_maps_indexes_to_original_docs():
    docs = [
        _doc("d1", rank=1, score=0.2),
        _doc("d2", rank=2, score=0.1),
        _doc("d3", rank=3, score=0.3),
    ]
    response = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.75},
        ]
    }

    results = dashscope_response_to_reranked_docs(
        docs,
        response,
        model="qwen3-rerank",
        top_n=2,
    )

    assert [doc.doc_id for doc in results] == ["d3", "d1"]
    assert [doc.rank for doc in results] == [1, 2]
    assert all(doc.retrieval_source == "reranker" for doc in results)
    assert results[0].score == pytest.approx(0.95)
    assert results[0].metadata["reranker_model"] == "qwen3-rerank"
    assert results[0].metadata["reranker_score"] == pytest.approx(0.95)
    assert results[0].metadata["reranker_rank"] == 1
    assert results[0].metadata["pre_rerank_rank"] == 3


def test_dashscope_reranker_reads_env_defaults(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=test-key",
                "DASHSCOPE_RERANK_MODEL=qwen3-rerank",
                "DASHSCOPE_RERANK_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_RERANK_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_RERANK_URL", raising=False)

    reranker = DashScopeReranker(env_path=env_file)

    assert reranker.api_key == "test-key"
    assert reranker.model == "qwen3-rerank"
    assert reranker.url == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


def test_dashscope_response_accepts_nested_output_results():
    docs = [_doc("d1", rank=1), _doc("d2", rank=2)]
    response = {"output": {"results": [{"index": 1, "relevance_score": 0.8}]}}

    results = dashscope_response_to_reranked_docs(
        docs,
        response,
        model="qwen3-rerank",
        top_n=1,
    )

    assert [doc.doc_id for doc in results] == ["d2"]
    assert results[0].metadata["pre_rerank_rank"] == 2


def test_dashscope_response_fills_missing_results_to_keep_top_n_length():
    docs = [_doc("d1", rank=1), _doc("d2", rank=2)]
    response = {"results": [{"index": 1, "relevance_score": 0.8}]}

    results = dashscope_response_to_reranked_docs(
        docs,
        response,
        model="qwen3-rerank",
        top_n=2,
    )

    assert [doc.doc_id for doc in results] == ["d2", "d1"]
    assert results[1].metadata["reranker_missing_result"] is True


def test_reranker_document_text_uses_title_and_text():
    doc = _doc("d1", rank=1, title="A Title", text="A paragraph.")

    assert reranker_document_text(doc) == "A Title\nA paragraph."


def test_fallback_rerank_docs_preserves_original_order_and_source():
    docs = [
        _doc("d1", rank=1, source="hybrid", score=0.5),
        _doc("d2", rank=2, source="hybrid", score=0.4),
    ]

    results = fallback_rerank_docs(docs, top_n=2)

    assert [doc.doc_id for doc in results] == ["d1", "d2"]
    assert [doc.rank for doc in results] == [1, 2]
    assert [doc.retrieval_source for doc in results] == ["hybrid", "hybrid"]
    assert results[0] is not docs[0]


def test_dashscope_response_requires_results_list():
    with pytest.raises(ValueError):
        dashscope_response_to_reranked_docs([_doc("d1", rank=1)], {}, model="qwen3-rerank")


def _doc(
    doc_id,
    *,
    rank,
    source="hybrid",
    score=1.0,
    title=None,
    text=None,
):
    return RetrievedDoc(
        doc_id=doc_id,
        title=title or f"Title {doc_id}",
        text=text or f"Text {doc_id}",
        sentences=[text or f"Text {doc_id}"],
        metadata={"original_rank": rank},
        score=score,
        rank=rank,
        retrieval_source=source,
    )
