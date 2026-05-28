from scripts.diagnose_decomposed_hybrid_retrieval import (
    _run_decomposed_hybrid_diagnostic,
    parse_args,
)
from src.data.schema import HotpotQASample, RetrievedDoc
from src.retrieval.query_decomposition import QueryDecompositionResult


def test_decomposed_hybrid_diagnostic_parse_args():
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
            "--bm25-top-k",
            "20",
            "--dense-top-k",
            "30",
            "--final-top-k",
            "50",
            "--decomposition-model",
            "Qwen/Qwen3-8B",
            "--decomposition-max-queries",
            "3",
            "--decomposition-cache",
            "outputs/cache/test_decomposition.jsonl",
            "--output",
            "outputs/predictions/decomposed.jsonl",
        ]
    )

    assert args.sample_size == 100
    assert args.sample_strategy == "uniform"
    assert args.bm25_backend == "elasticsearch"
    assert args.dense_backend == "faiss"
    assert args.bm25_top_k == 20
    assert args.dense_top_k == 30
    assert args.final_top_k == 50
    assert args.decomposition_model == "Qwen/Qwen3-8B"
    assert args.decomposition_max_queries == 3
    assert args.decomposition_cache == "outputs/cache/test_decomposition.jsonl"
    assert args.output == "outputs/predictions/decomposed.jsonl"


def test_decomposed_hybrid_diagnostic_records_decomposition_and_cost():
    sample = HotpotQASample(
        id="q1",
        question="Original question?",
        answer="answer",
        type="bridge",
        level="medium",
        supporting_facts=[["Gold Title", 0]],
        context=[],
    )

    records = list(
        _run_decomposed_hybrid_diagnostic(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FakeDecomposer(),
            cache=None,
            bm25_top_k=1,
            dense_top_k=1,
            final_top_k=5,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["cost"]["llm_calls"] == 1
    assert record["cost"]["decomposition_query_count"] == 2
    assert record["metrics"]["evidence_hit@5"] == 1.0
    assert record["metrics"]["evidence_full_hit@5"] == 1.0
    assert record["agent_outputs"]["decomposition"]["queries"] == [
        "Original question?",
        "Gold Title fact",
    ]
    assert record["retrieved_docs"][0]["retrieval_source"] == "decomposed_hybrid"


class FakeDecomposer:
    model = "fake-model"

    def decompose(self, *, sample_id, question):
        return QueryDecompositionResult(
            sample_id=sample_id,
            question=question,
            queries=[question, "Gold Title fact"],
            generated_queries=["Gold Title fact"],
            model=self.model,
        )


class FakeBM25Retriever:
    def retrieve(self, query, *, top_k):
        return [_doc("bm25", "Distractor", rank=1)]


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeDenseStore:
    def search(self, embedding, *, top_k):
        if embedding == [1.0]:
            return [_doc("dense", "Gold Title", rank=1)]
        return []


def _doc(doc_id, title, *, rank):
    return RetrievedDoc(
        doc_id=doc_id,
        title=title,
        text=f"Text {doc_id}",
        sentences=[f"Text {doc_id}"],
        metadata={},
        score=1.0,
        rank=rank,
        retrieval_source="test",
    )
