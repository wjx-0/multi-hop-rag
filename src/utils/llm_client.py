"""LLM client abstractions.

Phase 1 uses MockLLMClient so the data/retrieval/evaluation loop can run
without model credentials.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from src.data.schema import RetrievedDoc
from src.utils.text import simple_tokenize


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        ...

    def answer_from_docs(self, question: str, docs: list[RetrievedDoc]) -> "GenerationResult":
        ...


@dataclass(slots=True)
class GenerationResult:
    answer: str
    cost: dict[str, Any]


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


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


class AliyunDashScopeClient:
    """OpenAI-compatible Aliyun DashScope chat client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        env_path: str | Path = ".env",
    ) -> None:
        load_env_file(env_path)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("DASHSCOPE_TEMPERATURE", "0.0"))
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("DASHSCOPE_MAX_TOKENS", "512"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("DASHSCOPE_TIMEOUT_SECONDS", "60"))
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is empty. Fill it in .env before calling AliyunDashScopeClient.")

        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        payload.update(kwargs)

        request = urllib.request.Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope HTTP {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"DashScope request failed: {error}") from error

        data = json.loads(response_body)
        return data["choices"][0]["message"]["content"]

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        content = self.generate(messages, **kwargs)
        return json.loads(_strip_json_fence(content))

    def answer_from_docs(self, question: str, docs: list[RetrievedDoc]) -> GenerationResult:
        started = perf_counter()
        context = "\n\n".join(
            f"[{doc.rank}] title: {doc.title}\n{doc.text}"
            for doc in docs
        )
        answer = self.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a RAG answer generator. Answer the question using only "
                        "the provided context. If the context is insufficient, say you do not know. "
                        "Keep the answer concise and include citation markers like [1], [2] when possible."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
                },
            ]
        )
        latency = perf_counter() - started
        return GenerationResult(
            answer=answer.strip(),
            cost={
                "llm_calls": 1,
                "input_tokens": len(simple_tokenize(question)) + sum(len(simple_tokenize(doc.text)) for doc in docs),
                "output_tokens": len(simple_tokenize(answer)),
                "latency": latency,
                "mock_llm": False,
                "provider": "aliyun_dashscope",
                "model": self.model,
            },
        )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
