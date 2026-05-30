from scripts.run_decomposed_hybrid_rerank_rag import (
    _run_decomposed_rerank_rag,
    _should_reuse_local_llm_for_decomposition,
    parse_args,
)
from src.data.schema import HotpotQASample, RetrievedDoc
from src.retrieval.query_decomposition import QueryDecompositionResult


def test_decomposed_rerank_rag_parse_args():
    args = parse_args(
        [
            "--sample-size",
            "100",
            "--sample-strategy",
            "uniform",
            "--bm25-backend",
            "elasticsearch",
            "--dense-backend",
            "faiss",
            "--faiss-index",
            "data/indexes/faiss.index",
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
            "--decomposition-model",
            "Qwen/Qwen3-8B",
            "--decomposition-backend",
            "local",
            "--decomposition-max-queries",
            "4",
            "--decomposition-cache",
            "outputs/cache/decomposed_test.jsonl",
            "--decomposition-query-mode",
            "generated_or_original",
            "--local-decomposition-device",
            "cuda",
            "--local-decomposition-dtype",
            "float16",
            "--local-decomposition-max-new-tokens",
            "128",
            "--reranker-backend",
            "local",
            "--local-reranker-device",
            "cuda",
            "--local-reranker-batch-size",
            "2",
            "--local-reranker-dtype",
            "float16",
            "--llm",
            "local",
            "--local-llm-model",
            "Qwen/Qwen3-8B",
            "--local-llm-device",
            "cuda",
            "--local-llm-dtype",
            "float16",
            "--local-llm-max-new-tokens",
            "32",
            "--output",
            "outputs/predictions/decomposed_rag.jsonl",
        ]
    )

    assert args.sample_size == 100
    assert args.sample_strategy == "uniform"
    assert args.bm25_backend == "elasticsearch"
    assert args.dense_backend == "faiss"
    assert args.faiss_index == "data/indexes/faiss.index"
    assert args.elasticsearch_index == "hotpotqa_test"
    assert args.bm25_top_k == 40
    assert args.dense_top_k == 60
    assert args.hybrid_top_k == 50
    assert args.rerank_top_n == 25
    assert args.answer_top_k == 10
    assert args.decomposition_backend == "local"
    assert args.decomposition_model == "Qwen/Qwen3-8B"
    assert args.decomposition_max_queries == 4
    assert args.decomposition_cache == "outputs/cache/decomposed_test.jsonl"
    assert args.decomposition_query_mode == "generated_or_original"
    assert args.local_decomposition_device == "cuda"
    assert args.local_decomposition_dtype == "float16"
    assert args.local_decomposition_max_new_tokens == 128
    assert args.reranker_backend == "local"
    assert args.local_reranker_device == "cuda"
    assert args.local_reranker_batch_size == 2
    assert args.local_reranker_dtype == "float16"
    assert args.llm == "local"
    assert args.local_llm_model == "Qwen/Qwen3-8B"
    assert args.local_llm_device == "cuda"
    assert args.local_llm_dtype == "float16"
    assert args.local_llm_max_new_tokens == 32
    assert args.output == "outputs/predictions/decomposed_rag.jsonl"


def test_decomposed_rerank_rag_reuses_matching_local_llm_for_decomposition():
    args = parse_args(
        [
            "--decomposition-backend",
            "local",
            "--llm",
            "local",
            "--local-llm-model",
            "Qwen/Qwen3-8B",
            "--local-decomposition-model",
            "Qwen/Qwen3-8B",
        ]
    )

    assert _should_reuse_local_llm_for_decomposition(args) is True


def test_decomposed_rerank_rag_does_not_reuse_different_local_decomposition_model():
    args = parse_args(
        [
            "--decomposition-backend",
            "local",
            "--llm",
            "local",
            "--local-llm-model",
            "Qwen/Qwen3-8B",
            "--local-decomposition-model",
            "Qwen/Qwen3-4B",
        ]
    )

    assert _should_reuse_local_llm_for_decomposition(args) is False


def test_decomposed_rerank_rag_records_decomposition_rerank_generation_and_cost():
    sample = _sample(answer="final answer")

    records = list(
        _run_decomposed_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FakeDecomposer(),
            cache=None,
            reranker=PassThroughReranker(),
            llm_client=FakeLLMClient(answer="final answer"),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=5,
            rerank_top_n=5,
            answer_top_k=5,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["pred_answer"] == "final answer"
    assert record["metrics"]["answer_em"] == 1.0
    assert record["metrics"]["evidence_full_hit@5"] == 1.0
    assert record["metrics"]["decomposition_error"] == 0.0
    assert record["metrics"]["rerank_error"] == 0.0
    assert record["cost"]["decomposition_llm_calls"] == 1
    assert record["cost"]["answer_llm_calls"] == 1
    assert record["cost"]["llm_calls"] == 2
    assert record["cost"]["bm25_query_count"] == 2
    assert record["cost"]["dense_query_count"] == 2
    assert record["cost"]["rerank_calls"] == 1
    assert record["agent_outputs"]["decomposition"]["queries"] == [
        "Question?",
        "Gold Title fact",
    ]
    assert record["agent_outputs"]["rerank"]["answer_doc_count"] == 2
    assert record["retrieval_source"] == "decomposed_hybrid_rerank_rag"


def test_decomposed_rerank_rag_cache_hit_does_not_count_decomposition_llm_call():
    sample = _sample(answer="final answer")

    records = list(
        _run_decomposed_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FailingDecomposer(),
            cache=FakeCache(),
            reranker=PassThroughReranker(),
            llm_client=FakeLLMClient(answer="final answer"),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=5,
            rerank_top_n=5,
            answer_top_k=5,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    assert records[0]["cost"]["decomposition_llm_calls"] == 0
    assert records[0]["cost"]["answer_llm_calls"] == 1
    assert records[0]["cost"]["llm_calls"] == 1
    assert records[0]["agent_outputs"]["decomposition"]["from_cache"] is True


def test_decomposed_rerank_rag_falls_back_when_reranker_fails():
    sample = _sample(answer="final answer")

    records = list(
        _run_decomposed_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FakeDecomposer(),
            cache=None,
            reranker=FailingReranker(),
            llm_client=FakeLLMClient(answer="final answer"),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=5,
            rerank_top_n=5,
            answer_top_k=5,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    record = records[0]
    assert record["metrics"]["rerank_error"] == 1.0
    assert record["retrieved_docs"][0]["retrieval_source"] == "decomposed_hybrid"
    assert "reranker down" in record["agent_outputs"]["rerank"]["error"]


def test_decomposed_rerank_rag_records_llm_errors_and_continues():
    sample = _sample(answer="final answer")

    records = list(
        _run_decomposed_rerank_rag(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FakeDecomposer(),
            cache=None,
            reranker=PassThroughReranker(),
            llm_client=FailingLLMClient(),
            bm25_top_k=1,
            dense_top_k=1,
            hybrid_top_k=5,
            rerank_top_n=5,
            answer_top_k=5,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    record = records[0]
    assert record["pred_answer"] == ""
    assert record["metrics"]["llm_error"] == 1.0
    assert record["cost"]["answer_llm_calls"] == 1
    assert record["cost"]["llm_calls"] == 2
    assert "temporary api failure" in record["agent_outputs"]["llm_error"]


class FakeDecomposer:
    model = "fake-decomposer"

    def decompose(self, *, sample_id, question):
        return QueryDecompositionResult(
            sample_id=sample_id,
            question=question,
            queries=[question, "Gold Title fact"],
            generated_queries=["Gold Title fact"],
            model=self.model,
        )


class FailingDecomposer:
    def decompose(self, *, sample_id, question):
        raise AssertionError("decomposer should not be called when cache hits")


class FakeCache:
    def get(self, *, sample_id, question, query_mode="original_plus_generated"):
        return QueryDecompositionResult(
            sample_id=sample_id,
            question=question,
            queries=[question, "Gold Title fact"],
            generated_queries=["Gold Title fact"],
            model="cached-decomposer",
            from_cache=True,
            query_mode=query_mode,
        )

    def put(self, result):
        raise AssertionError("cache put should not be called on a hit")


class FakeBM25Retriever:
    def retrieve(self, query, *, top_k):
        return [_doc("bm25", "Distractor", rank=1, source="bm25")]


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeDenseStore:
    def search(self, embedding, *, top_k):
        if embedding == [1.0]:
            return [_doc("dense", "Gold Title", rank=1, source="dense")]
        return []


class PassThroughReranker:
    model = "qwen3-rerank"
    backend = "local"

    def rerank(self, query, docs, *, top_n):
        return docs[:top_n]


class FailingReranker:
    model = "qwen3-rerank"
    backend = "local"

    def rerank(self, query, docs, *, top_n):
        raise RuntimeError("reranker down")


class FakeLLMClient:
    model = "fake-model"
    provider = "fake"

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
    provider = "fake"

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
        supporting_facts=[["Gold Title", 0]],
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
