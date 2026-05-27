from scripts.diagnose_hybrid_rerank import _run_rerank_diagnostic, parse_args
from src.data.schema import HotpotQASample, RetrievedDoc


def test_run_rerank_diagnostic_parse_args():
    args = parse_args(
        [
            "--limit",
            "20",
            "--bm25-top-k",
            "40",
            "--dense-top-k",
            "60",
            "--hybrid-top-k",
            "50",
            "--rerank-top-n",
            "25",
            "--api-min-request-interval-seconds",
            "0.5",
            "--api-max-retries",
            "2",
            "--api-retry-backoff-seconds",
            "0.1",
            "--output",
            "outputs/predictions/rerank.jsonl",
        ]
    )

    assert args.limit == 20
    assert args.bm25_top_k == 40
    assert args.dense_top_k == 60
    assert args.hybrid_top_k == 50
    assert args.rerank_top_n == 25
    assert args.api_min_request_interval_seconds == 0.5
    assert args.api_max_retries == 2
    assert args.api_retry_backoff_seconds == 0.1
    assert args.output == "outputs/predictions/rerank.jsonl"


def test_run_rerank_diagnostic_records_error_and_keeps_hybrid_order():
    sample = HotpotQASample(
        id="q1",
        question="Question?",
        answer="answer",
        type="bridge",
        level="medium",
        supporting_facts=[["BM25 Title", 0]],
        context=[],
    )

    records = list(
        _run_rerank_diagnostic(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever([_doc("bm25", "BM25 Title", rank=1, source="bm25")]),
            embedder=FakeEmbedder(),
            store=FakeDenseStore([_doc("dense", "Dense Title", rank=1, source="dense")]),
            reranker=FailingReranker(),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=2,
            rerank_top_n=2,
            rrf_k=60,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["metrics"]["rerank_error"] == 1.0
    assert record["cost"]["rerank_calls"] == 1
    assert [doc["doc_id"] for doc in record["retrieved_docs"]] == ["bm25", "dense"]
    assert record["retrieved_docs"][0]["retrieval_source"] == "hybrid"


class FakeBM25Retriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, *, top_k):
        return self.docs[:top_k]


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeDenseStore:
    def __init__(self, docs):
        self.docs = docs

    def search(self, embedding, *, top_k):
        return self.docs[:top_k]


class FailingReranker:
    model = "qwen3-rerank"

    def rerank(self, query, docs, *, top_n):
        raise RuntimeError("boom")


def _doc(doc_id, title, *, rank, source):
    return RetrievedDoc(
        doc_id=doc_id,
        title=title,
        text=f"Text {doc_id}",
        sentences=[f"Text {doc_id}"],
        metadata={},
        score=1.0,
        rank=rank,
        retrieval_source=source,
    )
