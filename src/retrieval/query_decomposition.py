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
DEFAULT_DECOMPOSITION_QUERY_MODE = "original_plus_generated"
DECOMPOSITION_QUERY_MODES = ("original_plus_generated", "generated_or_original")


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
    query_mode: str = DEFAULT_DECOMPOSITION_QUERY_MODE

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
            query_mode=str(data.get("query_mode", DEFAULT_DECOMPOSITION_QUERY_MODE)),
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
        pass_model_arg: bool = True,
        query_mode: str = DEFAULT_DECOMPOSITION_QUERY_MODE,
    ) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")
        if query_mode not in DECOMPOSITION_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {DECOMPOSITION_QUERY_MODES}.")
        self.llm_client = llm_client
        self.model = model or getattr(llm_client, "model", "")
        self.max_queries = max_queries
        self.max_query_chars = max_query_chars
        self.pass_model_arg = pass_model_arg
        self.query_mode = query_mode

    def decompose(self, *, sample_id: str, question: str) -> QueryDecompositionResult:
        try:
            generation_kwargs: dict[str, Any] = {
                "temperature": 0,
                "max_tokens": 256,
            }
            if self.model and self.pass_model_arg:
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
                query_mode=self.query_mode,
            )
            fallback = _is_fallback_query_set(
                original_question=question,
                queries=queries,
                generated_queries=generated_queries,
                query_mode=self.query_mode,
                max_query_chars=self.max_query_chars,
            )
            error = "no valid decomposed queries" if fallback and generated_queries else None
            return QueryDecompositionResult(
                sample_id=sample_id,
                question=question,
                queries=queries,
                generated_queries=_selected_generated_queries(queries, fallback=fallback, query_mode=self.query_mode),
                model=self.model,
                fallback=fallback,
                error=error,
                query_mode=self.query_mode,
            )
        except Exception as error:  # noqa: BLE001 - decomposition fallback keeps diagnostics running.
            return fallback_decomposition_result(
                sample_id=sample_id,
                question=question,
                model=self.model,
                error=str(error),
                query_mode=self.query_mode,
            )


class QueryDecompositionCache:
    """Append-only JSONL cache keyed by HotpotQA sample id or question text."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[tuple[str, str], QueryDecompositionResult] = {}
        self._load()

    def get(
        self,
        *,
        sample_id: str,
        question: str,
        query_mode: str = DEFAULT_DECOMPOSITION_QUERY_MODE,
    ) -> QueryDecompositionResult | None:
        for key in (_cache_key(sample_id), _cache_key(question)):
            if not key:
                continue
            cache_key = (key, query_mode)
            if cache_key in self._records:
                record = self._records[cache_key]
                return QueryDecompositionResult(
                    sample_id=sample_id or record.sample_id,
                    question=question or record.question,
                    queries=list(record.queries),
                    generated_queries=list(record.generated_queries),
                    model=record.model,
                    fallback=record.fallback,
                    error=record.error,
                    from_cache=True,
                    query_mode=record.query_mode,
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
                self._records[(key, result.query_mode)] = result


def decomposition_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You rewrite and decompose complex multi-hop questions into search queries for retrieval over "
                "Wikipedia-style passages.\n\n"
                'Return only valid JSON with a key named "queries".\n'
                "The value must be a list of 2 to 4 strings.\n\n"
                "Rules:\n"
                "- Each query must be short, specific, and entity-focused.\n"
                "- Keep original entity names exactly when possible.\n"
                "- Prefer Wikipedia-style search phrases over full natural-language questions.\n"
                "- Split the question into queries that can retrieve different pieces of evidence.\n"
                "- Include queries for the main entity, bridge entity, comparison target, or requested attribute "
                "when relevant.\n"
                "- Do not guess the final answer.\n"
                "- Do not explain your reasoning.\n"
                "- Do not include citations, markdown, or extra keys.\n\n"
                "Examples:\n\n"
                "Question:\n"
                "Which magazine was started first, Arthur's Magazine or First for Women?\n\n"
                "Return:\n"
                '{"queries": ["Arthur\'s Magazine founding date", "First for Women founding date", '
                '"Arthur\'s Magazine First for Women magazine"]}\n\n'
                "Question:\n"
                "The director of the romantic comedy Big Stone Gap is based in what New York city?\n\n"
                "Return:\n"
                '{"queries": ["Big Stone Gap romantic comedy director", '
                '"Adriana Trigiani based in New York city", "Adriana Trigiani"]}\n\n'
                "Question:\n"
                "What government position was held by the woman who portrayed Corliss Archer in the film "
                "Kiss and Tell?\n\n"
                "Return:\n"
                '{"queries": ["Kiss and Tell film Corliss Archer actress", '
                '"Shirley Temple government position", "Shirley Temple"]}\n\n'
                "Now decompose the question into retrieval queries."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question:\n"
                f"{question}\n\n"
                "Return JSON only."
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
    query_mode: str = DEFAULT_DECOMPOSITION_QUERY_MODE,
) -> list[str]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive.")
    if query_mode not in DECOMPOSITION_QUERY_MODES:
        raise ValueError(f"query_mode must be one of {DECOMPOSITION_QUERY_MODES}.")

    candidate_queries = (
        [original_question, *generated_queries]
        if query_mode == "original_plus_generated"
        else generated_queries
    )
    cleaned = _clean_query_candidates(
        candidate_queries,
        max_queries=max_queries,
        max_query_chars=max_query_chars,
    )
    return cleaned or [original_question.strip()]


def fallback_decomposition_result(  #如果llm调用失败会会滚
    *,
    sample_id: str,
    question: str,
    model: str,
    error: str,
    query_mode: str = DEFAULT_DECOMPOSITION_QUERY_MODE,
) -> QueryDecompositionResult:
    return QueryDecompositionResult(
        sample_id=sample_id,
        question=question,
        queries=[question.strip()],
        generated_queries=[],
        model=model,
        fallback=True,
        error=error,
        query_mode=query_mode,
    )


def _is_fallback_query_set(
    *,
    original_question: str,
    queries: list[str],
    generated_queries: list[str],
    query_mode: str,
    max_query_chars: int,
) -> bool:
    if query_mode == "original_plus_generated":
        return len(queries) == 1

    cleaned_generated = _clean_query_candidates(
        generated_queries,
        max_queries=len(generated_queries) or 1,
        max_query_chars=max_query_chars,
    )
    return not cleaned_generated


def _selected_generated_queries(
    queries: list[str],
    *,
    fallback: bool,
    query_mode: str,
) -> list[str]:
    if fallback:
        return []
    if query_mode == "generated_or_original":
        return list(queries)
    return [query for query in queries[1:]]


def _clean_query_text(query: str, *, max_chars: int) -> str:
    query = re.sub(r"^\s*[-*\d.)]+\s*", "", str(query).strip())
    query = re.sub(r"\s+", " ", query)
    if len(query) > max_chars:
        query = query[:max_chars].rstrip()
    return query


def _clean_query_candidates(
    queries: list[str],
    *,
    max_queries: int,
    max_query_chars: int,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _clean_query_text(query, max_chars=max_query_chars)
        key = _dedupe_key(normalized)
        if not normalized or key in seen:
            continue
        cleaned.append(normalized)
        seen.add(key)
        if len(cleaned) >= max_queries:
            break
    return cleaned


def _dedupe_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_key(value: str) -> str:
    return _dedupe_key(value)
