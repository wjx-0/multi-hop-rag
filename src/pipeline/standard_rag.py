"""Standard RAG baseline for Phase 1."""

from __future__ import annotations

from src.data.build_corpus import context_to_documents
from src.data.schema import HotpotQASample, PipelineResult
from src.evaluation.answer_metrics import answer_metrics
from src.evaluation.evidence_metrics import evidence_metrics
from src.retrieval.bm25 import BM25Retriever
from src.utils.llm_client import LLMClient, MockLLMClient
from src.utils.text import simple_tokenize


class StandardRAGPipeline:
    def __init__(self, *, top_k: int = 5, llm_client: LLMClient | None = None) -> None:
        self.top_k = top_k
        self.llm_client = llm_client or MockLLMClient()

    def run(self, sample: HotpotQASample) -> PipelineResult:
        documents = context_to_documents(sample)
        retriever = BM25Retriever(documents)
        retrieved_docs = retriever.retrieve(sample.question, top_k=self.top_k)
        generation = self.llm_client.answer_from_docs(sample.question, retrieved_docs)

        citations = select_sentence_citations(
            question=sample.question,
            answer=generation.answer,
            retrieved_docs=retrieved_docs,
            max_citations=2,
        )
        retrieved_doc_dicts = [doc.to_dict() for doc in retrieved_docs]
        metrics = answer_metrics(generation.answer, sample.answer)
        metrics.update(
            evidence_metrics(
                retrieved_doc_dicts,
                sample.supporting_facts,
                predicted_supporting_facts=citations,
            )
        )
        return PipelineResult(
            id=sample.id,
            question=sample.question,
            gold_answer=sample.answer,
            gold_supporting_facts=sample.supporting_facts,
            pred_answer=generation.answer,
            retrieved_docs=retrieved_doc_dicts,
            pred_citations=citations,
            metrics=metrics,
            cost=generation.cost,
            route_initial={"route": "simple", "confidence": 1.0, "reason": "phase1_standard_rag"},
            route_final="simple",
            was_upgraded=False,
        )


def select_sentence_citations(
    *,
    question: str,
    answer: str,
    retrieved_docs,
    max_citations: int = 2,
) -> list[dict]:
    """Select sentence-level citations from retrieved docs using lexical overlap."""

    query_terms = set(simple_tokenize(f"{question} {answer}"))
    scored_sentences: list[tuple[float, int, int, object]] = []
    for doc_rank, doc in enumerate(retrieved_docs, start=1):
        for sentence_id, sentence in enumerate(doc.sentences):
            sentence_terms = set(simple_tokenize(sentence))
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap + 1.0 / (doc_rank + 1)
            scored_sentences.append((score, doc_rank, sentence_id, doc))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    citations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for score, _, sentence_id, doc in scored_sentences:
        if score <= 0:
            continue
        key = (doc.doc_id, sentence_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_doc_id": doc.doc_id,
                "source_title": doc.title,
                "source_sentence_id": sentence_id,
                "source_text": doc.sentences[sentence_id],
                "citation_score": score,
            }
        )
        if len(citations) >= max_citations:
            break

    return citations
