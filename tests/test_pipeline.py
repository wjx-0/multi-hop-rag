from src.data.load_hotpotqa import load_hotpotqa
from src.data.schema import Document, HotpotQASample, RetrievedDoc
from src.pipeline.standard_rag import select_sentence_citations
from src.pipeline.standard_rag import StandardRAGPipeline


def test_standard_rag_pipeline_smoke():
    samples = load_hotpotqa("data/raw/hotpotqa/hotpot_dev_distractor_v1.json", limit=1)
    result = StandardRAGPipeline(top_k=3).run(samples[0])
    assert result.id == samples[0].id
    assert result.retrieved_docs
    assert "answer_f1" in result.metrics


def test_standard_rag_pipeline_global_mode_uses_shared_corpus_with_empty_context():
    documents = [
        Document(
            doc_id="d1",
            title="Arthur's Magazine",
            text="Arthur's Magazine was founded in 1844.",
            sentences=["Arthur's Magazine was founded in 1844."],
        ),
        Document(
            doc_id="d2",
            title="Other",
            text="A distractor sentence.",
            sentences=["A distractor sentence."],
        ),
    ]
    sample = HotpotQASample(
        id="q1",
        question="When was Arthur's Magazine founded?",
        answer="1844",
        type="bridge",
        level="easy",
        supporting_facts=[["Arthur's Magazine", 0]],
        context=[],
    )

    result = StandardRAGPipeline(top_k=1, documents=documents).run(sample)

    assert result.retrieved_docs[0]["doc_id"] == "d1"
    assert result.route_initial["retrieval_mode"] == "global"


def test_standard_rag_pipeline_records_llm_errors_and_continues():
    class FailingLLMClient:
        def answer_from_docs(self, question, docs):
            raise RuntimeError("temporary api failure")

        def generate(self, messages, **kwargs):
            return ""

        def generate_json(self, messages, schema=None, **kwargs):
            return {}

    sample = HotpotQASample(
        id="q1",
        question="Question?",
        answer="Answer",
        type="bridge",
        level="easy",
        supporting_facts=[],
        context=[["Title", ["Evidence sentence."]]],
    )

    result = StandardRAGPipeline(llm_client=FailingLLMClient()).run(sample)

    assert result.pred_answer == ""
    assert result.metrics["llm_error"] == 1.0
    assert "temporary api failure" in result.agent_outputs["llm_error"]


def test_select_sentence_citations_uses_best_sentence_not_always_zero():
    doc = RetrievedDoc.from_document(
        Document(
            doc_id="d1",
            title="Arthur's Magazine",
            text="Arthur's Magazine was a periodical. It was founded in 1844.",
            sentences=[
                "Arthur's Magazine was a periodical.",
                "It was founded in 1844.",
            ],
        ),
        score=1.0,
        rank=1,
        retrieval_source="bm25",
    )

    citations = select_sentence_citations(
        question="When was Arthur's Magazine founded?",
        answer="It was founded in 1844.",
        retrieved_docs=[doc],
        max_citations=1,
    )

    assert citations[0]["source_sentence_id"] == 1
    assert citations[0]["source_text"] == "It was founded in 1844."
