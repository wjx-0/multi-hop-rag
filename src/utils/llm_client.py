"""LLM client abstractions.

Phase 1 uses MockLLMClient so the data/retrieval/evaluation loop can run
without model credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from src.data.schema import RetrievedDoc
from src.utils.text import simple_tokenize


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class GenerationResult:
    answer: str
    cost: dict[str, Any]


class MockLLMClient:
    """A deterministic extractive placeholder for smoke tests."""

    def answer_from_docs(self, question: str, docs: list[RetrievedDoc]) -> GenerationResult:
        started = perf_counter()
        query_terms = set(simple_tokenize(question))
        best_sentence = ""
        best_overlap = -1
        for doc in docs:
            for sentence in doc.sentences:
                overlap = len(query_terms & set(simple_tokenize(sentence)))
                if overlap > best_overlap:
                    best_sentence = sentence
                    best_overlap = overlap

        answer = best_sentence.strip() or (docs[0].text.strip() if docs else "")
        latency = perf_counter() - started
        return GenerationResult(
            answer=answer,
            cost={
                "llm_calls": 0,
                "input_tokens": len(simple_tokenize(question)) + sum(len(simple_tokenize(doc.text)) for doc in docs),
                "output_tokens": len(simple_tokenize(answer)),
                "latency": latency,
                "mock_llm": True,
            },
        )

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return ""

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {}
