from scripts.diagnose_decomposed_fusion_stages import (
    _run_decomposed_fusion_stage_diagnostic,
    parse_args,
    parse_ks,
)
from src.data.schema import HotpotQASample, RetrievedDoc
from src.retrieval.query_decomposition import QueryDecompositionResult


def test_decomposed_fusion_stages_parse_args():
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
            "50",
            "--dense-top-k",
            "50",
            "--final-top-k",
            "100",
            "--fusion-ks",
            "5,10,50,100",
            "--decomposition-backend",
            "local",
            "--decomposition-query-mode",
            "generated_or_original",
            "--output",
            "outputs/predictions/stages.jsonl",
        ]
    )

    assert args.sample_size == 100
    assert args.sample_strategy == "uniform"
    assert args.bm25_backend == "elasticsearch"
    assert args.dense_backend == "faiss"
    assert args.bm25_top_k == 50
    assert args.dense_top_k == 50
    assert args.final_top_k == 100
    assert args.fusion_ks == "5,10,50,100"
    assert args.decomposition_backend == "local"
    assert args.decomposition_query_mode == "generated_or_original"
    assert args.output == "outputs/predictions/stages.jsonl"


def test_parse_ks_dedupes_and_keeps_order():
    assert parse_ks("5,10,10,50") == (5, 10, 50)


def test_decomposed_fusion_stage_diagnostic_compares_prefusion_and_fusion():
    sample = HotpotQASample(
        id="q1",
        question="Question?",
        answer="answer",
        type="bridge",
        level="medium",
        supporting_facts=[["Gold A", 0], ["Gold B", 1]],
        context=[],
    )

    records = list(
        _run_decomposed_fusion_stage_diagnostic(
            samples=[sample],
            bm25_retriever=FakeBM25Retriever(),
            embedder=FakeEmbedder(),
            store=FakeDenseStore(),
            decomposer=FakeDecomposer(),
            cache=None,
            bm25_top_k=2,
            dense_top_k=2,
            final_top_k=1,
            rrf_k=60,
            title_boost_weight=0.0,
            query_batch_size=1,
            progress_interval=0,
            fusion_ks=(1,),
        )
    )

    assert len(records) == 1
    record = records[0]
    metrics = record["metrics"]
    assert metrics["prefusion_bm25_pool_evidence_hit"] == 1.0
    assert metrics["prefusion_dense_pool_evidence_hit"] == 1.0
    assert metrics["prefusion_union_pool_evidence_full_hit"] == 1.0
    assert metrics["fusion_evidence_full_hit@1"] == 0.0
    assert record["cost"]["decomposition_query_count"] == 2
    assert record["cost"]["union_pool_doc_count"] == 3
    assert record["agent_outputs"]["prefusion"]["union_gold_titles"] == ["gold a", "gold b"]
    assert record["retrieved_docs"][0]["retrieval_source"] == "decomposed_hybrid"


class FakeDecomposer:
    model = "fake-decomposer"

    def decompose(self, *, sample_id, question):
        return QueryDecompositionResult(
            sample_id=sample_id,
            question=question,
            queries=[question, "second hop"],
            generated_queries=["second hop"],
            model=self.model,
        )


class FakeBM25Retriever:
    def retrieve(self, query, *, top_k):
        if query == "second hop":
            return [_doc("a", "Gold A", rank=1), _doc("x", "Distractor", rank=2)][:top_k]
        return [_doc("x", "Distractor", rank=1)][:top_k]


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeDenseStore:
    def search(self, embedding, *, top_k):
        if embedding == [1.0]:
            return [_doc("b", "Gold B", rank=1)][:top_k]
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
