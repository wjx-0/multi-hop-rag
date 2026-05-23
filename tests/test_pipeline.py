from src.data.load_hotpotqa import load_hotpotqa
from src.data.schema import Document, RetrievedDoc
from src.pipeline.standard_rag import select_sentence_citations
from src.pipeline.standard_rag import StandardRAGPipeline


def test_standard_rag_pipeline_smoke():
    samples = load_hotpotqa("data/raw/hotpotqa/hotpot_dev_distractor_v1.json", limit=1)
    result = StandardRAGPipeline(top_k=3).run(samples[0])
    assert result.id == samples[0].id
    assert result.retrieved_docs
    assert "answer_f1" in result.metrics


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
