from scripts.run_hybrid_rerank_rag import _run_rerank_rag, parse_args
from src.data.schema import HotpotQASample, RetrievedDoc


def test_run_rerank_rag_parse_args():
    args = parse_args(
        [
            "--limit",
            "20",
            "--bm25-backend",
            "elasticsearch",
            "--elasticsearch-index",
            "hotpotqa_test",
            "--bm25-top-k",
            "40",
            "--dense-top-k",
            "60",
            "--hybrid-top-k",
            "50",
            "--rerank-top-n",
            "25",
            "--answer-top-k",
            "10",
            "--title-boost-weight",
            "0.0007",
            "--llm",
            "mock",
            "--output",
            "outputs/predictions/rerank_rag.jsonl",
        ]
    )

    assert args.limit == 20
    assert args.bm25_backend == "elasticsearch"
    assert args.elasticsearch_index == "hotpotqa_test"
    assert args.bm25_top_k == 40
    assert args.dense_top_k == 60
    assert args.hybrid_top_k == 50
    assert args.rerank_top_n == 25
    assert args.answer_top_k == 10
    assert args.title_boost_weight == 0.0007
    assert args.llm == "mock"
    assert args.output == "outputs/predictions/rerank_rag.jsonl"


def test_run_rerank_rag_falls_back_when_reranker_fails():
    sample = _sample(answer="final answer")

    records = list(
        _run_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever([_doc("bm25", "BM25 Title", rank=1, source="bm25")]),
            embedder=FakeEmbedder(),
            store=FakeDenseStore([_doc("dense", "Dense Title", rank=1, source="dense")]),
            reranker=FailingReranker(),
            llm_client=FakeLLMClient(answer="final answer"),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=2,
            rerank_top_n=2,
            answer_top_k=1,
            rrf_k=60,
            query_batch_size=1,
            progress_interval=0,
            title_boost_weight=0.0,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["pred_answer"] == "final answer"
    assert record["metrics"]["answer_em"] == 1.0
    assert record["metrics"]["rerank_error"] == 1.0
    assert record["cost"]["llm_calls"] == 1
    assert record["cost"]["rerank_calls"] == 1
    assert record["cost"]["answer_top_k"] == 1
    assert record["retrieved_docs"][0]["retrieval_source"] == "hybrid"


def test_run_rerank_rag_records_llm_errors_and_continues():
    sample = _sample(answer="final answer")

    records = list(
        _run_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever([_doc("bm25", "BM25 Title", rank=1, source="bm25")]),
            embedder=FakeEmbedder(),
            store=FakeDenseStore([]),
            reranker=PassThroughReranker(),
            llm_client=FailingLLMClient(),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=1,
            rerank_top_n=1,
            answer_top_k=1,
            rrf_k=60,
            query_batch_size=1,
            progress_interval=0,
            title_boost_weight=0.0,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["pred_answer"] == ""
    assert record["metrics"]["llm_error"] == 1.0
    assert record["metrics"]["rerank_error"] == 0.0
    assert record["cost"]["llm_calls"] == 1
    assert "temporary api failure" in record["agent_outputs"]["llm_error"]


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
        raise RuntimeError("reranker down")


class PassThroughReranker:
    model = "qwen3-rerank"

    def rerank(self, query, docs, *, top_n):
        return docs[:top_n]


class FakeLLMClient:
    model = "fake-model"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, messages, **kwargs):
        return self.answer

    def generate_json(self, messages, schema=None, **kwargs):
        return {}

    def answer_from_docs(self, question, docs):
        raise AssertionError("answer_from_docs should not be used for API-style fake LLM")


class FailingLLMClient:
    model = "fake-model"

    def generate(self, messages, **kwargs):
        raise RuntimeError("temporary api failure")

    def generate_json(self, messages, schema=None, **kwargs):
        return {}

    def answer_from_docs(self, question, docs):
        raise RuntimeError("temporary api failure")


def _sample(*, answer):
    return HotpotQASample(
        id="q1",
        question="Question?",
        answer=answer,
        type="bridge",
        level="medium",
        supporting_facts=[["BM25 Title", 0]],
        context=[],
    )


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
