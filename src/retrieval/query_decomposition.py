"""LLM query decomposition helpers for multi-hop retrieval diagnostics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.utils.io import read_jsonl

DEFAULT_DECOMPOSITION_MAX_QUERIES = 4
DEFAULT_DECOMPOSITION_MAX_QUERY_CHARS = 200
DEFAULT_DECOMPOSITION_CACHE = "outputs/cache/decomposed_queries.jsonl"


class JsonLLMClient(Protocol):
    model: str

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class QueryDecompositionResult:
    sample_id: str
    question: str
    queries: list[str]
    generated_queries: list[str]
    model: str
    fallback: bool = False
    error: str | None = None
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, from_cache: bool = False) -> "QueryDecompositionResult":
        return cls(
            sample_id=str(data.get("sample_id", data.get("id", ""))),
            question=str(data.get("question", "")),
            queries=[str(query) for query in data.get("queries", [])],
            generated_queries=[str(query) for query in data.get("generated_queries", [])],
            model=str(data.get("model", "")),
            fallback=bool(data.get("fallback", False)),
            error=data.get("error"),
            from_cache=from_cache,
        )


class LLMQueryDecomposer:
    """Generate retrieval-oriented sub-queries with a JSON-speaking LLM."""

    def __init__(
        self,
        *,
        llm_client: JsonLLMClient,
        model: str | None = None,
        max_queries: int = DEFAULT_DECOMPOSITION_MAX_QUERIES,
        max_query_chars: int = DEFAULT_DECOMPOSITION_MAX_QUERY_CHARS,
    ) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")
        self.llm_client = llm_client
        self.model = model or getattr(llm_client, "model", "")
        self.max_queries = max_queries
        self.max_query_chars = max_query_chars

    def decompose(self, *, sample_id: str, question: str) -> QueryDecompositionResult:
        try:
            generation_kwargs: dict[str, Any] = {
                "temperature": 0,
                "max_tokens": 256,
            }
            if self.model:
                generation_kwargs["model"] = self.model
            response = self.llm_client.generate_json(
                decomposition_messages(question),
                **generation_kwargs,
            )
            generated_queries = extract_queries_from_response(response)
            queries = clean_decomposed_queries(
                question,
                generated_queries,
                max_queries=self.max_queries,
                max_query_chars=self.max_query_chars,
            )
            fallback = len(queries) == 1
            error = "no valid decomposed queries" if fallback and generated_queries else None
            return QueryDecompositionResult(
                sample_id=sample_id,
                question=question,
                queries=queries,
                generated_queries=[query for query in queries[1:]],
                model=self.model,
                fallback=fallback,
                error=error,
            )
        except Exception as error:  # noqa: BLE001 - decomposition fallback keeps diagnostics running.
            return fallback_decomposition_result(
                sample_id=sample_id,
                question=question,
                model=self.model,
                error=str(error),
            )


class QueryDecompositionCache:
    """Append-only JSONL cache keyed by HotpotQA sample id or question text."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, QueryDecompositionResult] = {}
        self._load()

    def get(self, *, sample_id: str, question: str) -> QueryDecompositionResult | None:
        for key in (_cache_key(sample_id), _cache_key(question)):
            if not key:
                continue
            if key in self._records:
                record = self._records[key]
                return QueryDecompositionResult(
                    sample_id=sample_id or record.sample_id,
                    question=question or record.question,
                    queries=list(record.queries),
                    generated_queries=list(record.generated_queries),
                    model=record.model,
                    fallback=record.fallback,
                    error=record.error,
                    from_cache=True,
                )
        return None

    def put(self, result: QueryDecompositionResult) -> None:
        if result.fallback or result.error:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            import json

            file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        self._store(result)

    def _load(self) -> None:
        if not self.path.exists():
            return
        for record in read_jsonl(self.path):
            result = QueryDecompositionResult.from_dict(record)
            if result.queries:
                self._store(result)

    def _store(self, result: QueryDecompositionResult) -> None:
        for key in (_cache_key(result.sample_id), _cache_key(result.question)):
            if key:
                self._records[key] = result


def decomposition_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You decompose HotpotQA multi-hop questions into retrieval queries. "
                "Return only JSON with a key named queries. "
                "Each query should be short, entity-focused, and useful for retrieving Wikipedia paragraphs."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{question}\n\n"
                'Return JSON like {"queries": ["query 1", "query 2", "query 3"]}.'
            ),
        },
    ]


def extract_queries_from_response(response: dict[str, Any]) -> list[str]:
    raw_queries = (
        response.get("queries")
        or response.get("sub_queries")
        or response.get("decomposed_queries")
        or []
    )
    if not isinstance(raw_queries, list):
        return []

    queries: list[str] = []
    for item in raw_queries:
        if isinstance(item, str):
            queries.append(item)
        elif isinstance(item, dict):
            value = item.get("query", item.get("question", item.get("text", "")))
            if isinstance(value, str):
                queries.append(value)
    return queries


def clean_decomposed_queries(
    original_question: str,
    generated_queries: list[str],
    *,
    max_queries: int,
    max_query_chars: int = DEFAULT_DECOMPOSITION_MAX_QUERY_CHARS,
) -> list[str]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for query in [original_question, *generated_queries]:
        normalized = _clean_query_text(query, max_chars=max_query_chars)
        key = _dedupe_key(normalized)
        if not normalized or key in seen:
            continue
        cleaned.append(normalized)
        seen.add(key)
        if len(cleaned) >= max_queries:
            break
    return cleaned or [original_question.strip()]


def fallback_decomposition_result(  #如果llm调用失败会会滚
    *,
    sample_id: str,
    question: str,
    model: str,
    error: str,
) -> QueryDecompositionResult:
    return QueryDecompositionResult(
        sample_id=sample_id,
        question=question,
        queries=[question.strip()],
        generated_queries=[],
        model=model,
        fallback=True,
        error=error,
    )


def _clean_query_text(query: str, *, max_chars: int) -> str:
    query = re.sub(r"^\s*[-*\d.)]+\s*", "", str(query).strip())
    query = re.sub(r"\s+", " ", query)
    if len(query) > max_chars:
        query = query[:max_chars].rstrip()
    return query


def _dedupe_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_key(value: str) -> str:
    return _dedupe_key(value)
