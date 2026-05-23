"""Standard RAG baseline for Phase 1."""

from __future__ import annotations

from src.data.build_corpus import context_to_documents
from src.data.schema import HotpotQASample, PipelineResult
from src.evaluation.answer_metrics import answer_metrics
from src.retrieval.bm25 import BM25Retriever
from src.utils.llm_client import MockLLMClient


class StandardRAGPipeline:
    def __init__(self, *, top_k: int = 5, llm_client: MockLLMClient | None = None) -> None:
        self.top_k = top_k
        self.llm_client = llm_client or MockLLMClient()

    def run(self, sample: HotpotQASample) -> PipelineResult:
        documents = context_to_documents(sample)
        retriever = BM25Retriever(documents)
        retrieved_docs = retriever.retrieve(sample.question, top_k=self.top_k)
        generation = self.llm_client.answer_from_docs(sample.question, retrieved_docs)

        citations = [
            {
                "source_doc_id": doc.doc_id,
                "source_title": doc.title,
                "source_sentence_id": 0,
            }
            for doc in retrieved_docs[:1]
        ]
        metrics = answer_metrics(generation.answer, sample.answer)
        return PipelineResult(
            id=sample.id,
            question=sample.question,
            gold_answer=sample.answer,
            pred_answer=generation.answer,
            retrieved_docs=[doc.to_dict() for doc in retrieved_docs],
            pred_citations=citations,
            metrics=metrics,
            cost=generation.cost,
            route_initial={"route": "simple", "confidence": 1.0, "reason": "phase1_standard_rag"},
            route_final="simple",
            was_upgraded=False,
        )
