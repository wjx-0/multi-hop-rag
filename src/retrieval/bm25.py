"""A small dependency-free BM25 retriever for Phase 1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from src.data.schema import Document, RetrievedDoc
from src.utils.text import simple_tokenize


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        retrieval_source: str = "bm25",
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.retrieval_source = retrieval_source
        self.doc_tokens = [simple_tokenize(doc.text) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1
        total_docs = len(self.documents)
        return {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = simple_tokenize(query)
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index] or 1
            score = 0.0
            for term in query_terms:
                term_freq = frequencies.get(term, 0)
                if term_freq == 0:
                    continue
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_length / (self.avg_doc_length or 1)
                )
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores.append(score)
        return scores

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDoc]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedDoc.from_document(
                self.documents[index],
                score=score,
                rank=rank,
                retrieval_source=self.retrieval_source,
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]
