"""DashScope API reranking helpers."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

from src.data.schema import RetrievedDoc
from src.utils.llm_client import load_env_file

DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_RERANK_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."


class DashScopeReranker:
    """DashScope qwen3-rerank client for reranking retrieved paragraphs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        model: str | None = None,
        instruct: str = DEFAULT_RERANK_INSTRUCT,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
        env_path: str | Path = ".env",
    ) -> None:
        load_env_file(env_path)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.url = (url or os.getenv("DASHSCOPE_RERANK_URL", DEFAULT_RERANK_URL)).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self.instruct = instruct
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("DASHSCOPE_TIMEOUT_SECONDS", "60"))
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("DASHSCOPE_MAX_RETRIES", "3"))
        )
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else float(os.getenv("DASHSCOPE_RETRY_BACKOFF_SECONDS", "1.5"))
        )
        self.min_request_interval_seconds = (
            min_request_interval_seconds
            if min_request_interval_seconds is not None
            else float(os.getenv("DASHSCOPE_MIN_REQUEST_INTERVAL_SECONDS", "1.0"))
        )
        self._last_request_started_at = 0.0

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedDoc]:
        if not docs:
            return []
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is empty. Fill it in .env before calling DashScopeReranker.")

        effective_top_n = _validate_top_n(top_n, len(docs))
        payload = {
            "model": self.model,
            "query": query,
            "documents": [reranker_document_text(doc) for doc in docs],
            "top_n": effective_top_n,
            "instruct": self.instruct,
        }
        response_data = json.loads(self._post_json(payload))
        return dashscope_response_to_reranked_docs(
            docs,
            response_data,
            model=self.model,
            top_n=effective_top_n,
        )

    def _post_json(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._send_with_retries(request)

    def _send_with_retries(self, request: urllib.request.Request) -> str:
        attempts = self.max_retries + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._throttle()
            try:
                self._last_request_started_at = perf_counter()
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"DashScope rerank HTTP {error.code}: {error_body}")
                if not _is_retryable_http_status(error.code) or attempt == attempts:
                    raise last_error from error
            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
                last_error = RuntimeError(f"DashScope rerank request failed: {error}")
                if attempt == attempts:
                    raise last_error from error

            sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
            print(f"DashScope rerank failed; retrying in {sleep_seconds:.1f}s ({attempt}/{self.max_retries})")
            time.sleep(sleep_seconds)

        raise RuntimeError(f"DashScope rerank failed after retries: {last_error}")

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = perf_counter() - self._last_request_started_at
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)


def reranker_document_text(doc: RetrievedDoc) -> str:
    return f"{doc.title}\n{doc.text}".strip()


def dashscope_response_to_reranked_docs(
    docs: list[RetrievedDoc],
    response_data: dict[str, Any],
    *,
    model: str,
    top_n: int | None = None,
) -> list[RetrievedDoc]:
    max_results = _validate_top_n(top_n, len(docs))
    raw_results = _extract_results(response_data)
    reranked_docs: list[RetrievedDoc] = []
    used_indexes: set[int] = set()

    for result in raw_results:
        if len(reranked_docs) >= max_results:
            break
        index = result.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(docs) or index in used_indexes:
            continue
        score = _result_score(result)
        used_indexes.add(index)
        reranked_docs.append(
            _copy_reranked_doc(
                docs[index],
                model=model,
                score=score,
                reranker_rank=len(reranked_docs) + 1,
            )
        )

    # Keep diagnostic output length stable if the API returns fewer results than requested.
    for index, doc in enumerate(docs):
        if len(reranked_docs) >= max_results:
            break
        if index in used_indexes:
            continue
        reranked_docs.append(
            _copy_reranked_doc(
                doc,
                model=model,
                score=0.0,
                reranker_rank=len(reranked_docs) + 1,
                missing_result=True,
            )
        )

    return reranked_docs


def fallback_rerank_docs(
    docs: list[RetrievedDoc],
    *,
    top_n: int | None = None,
) -> list[RetrievedDoc]:
    """Return a copy of docs in their original order after a reranker failure."""

    max_results = _validate_top_n(top_n, len(docs))
    return [
        RetrievedDoc(
            doc_id=doc.doc_id,
            title=doc.title,
            text=doc.text,
            sentences=list(doc.sentences),
            metadata=dict(doc.metadata),
            score=doc.score,
            rank=rank,
            retrieval_source=doc.retrieval_source,
        )
        for rank, doc in enumerate(docs[:max_results], start=1)
    ]


def _extract_results(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    results = response_data.get("results")
    if results is None and isinstance(response_data.get("output"), dict):
        results = response_data["output"].get("results")
    if not isinstance(results, list):
        raise ValueError("DashScope rerank response missing results list.")
    return [result for result in results if isinstance(result, dict)]


def _result_score(result: dict[str, Any]) -> float:
    score = result.get("relevance_score", result.get("score", 0.0))
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _copy_reranked_doc(
    doc: RetrievedDoc,
    *,
    model: str,
    score: float,
    reranker_rank: int,
    missing_result: bool = False,
) -> RetrievedDoc:
    metadata = dict(doc.metadata)
    metadata["reranker_model"] = model
    metadata["reranker_score"] = score
    metadata["reranker_rank"] = reranker_rank
    metadata["pre_rerank_rank"] = doc.rank
    if missing_result:
        metadata["reranker_missing_result"] = True
    return RetrievedDoc(
        doc_id=doc.doc_id,
        title=doc.title,
        text=doc.text,
        sentences=list(doc.sentences),
        metadata=metadata,
        score=score,
        rank=reranker_rank,
        retrieval_source="reranker",
    )


def _validate_top_n(top_n: int | None, doc_count: int) -> int:
    if doc_count == 0:
        return 0
    if top_n is None:
        return doc_count
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    return min(top_n, doc_count)


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600
