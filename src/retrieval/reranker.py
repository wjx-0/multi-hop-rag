"""Reranking helpers for DashScope API and local Qwen3 models."""

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
DEFAULT_LOCAL_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_LOCAL_RERANK_BATCH_SIZE = 4
DEFAULT_LOCAL_RERANK_MAX_LENGTH = 8192
LOCAL_RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
LOCAL_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class DashScopeReranker:
    """DashScope qwen3-rerank client for reranking retrieved paragraphs."""

    backend = "dashscope"

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
                backend="dashscope",
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
                backend="dashscope",
            )
        )

    return reranked_docs


class LocalQwen3Reranker:
    """Local Qwen3-Reranker scorer using yes/no logits from AutoModelForCausalLM."""

    backend = "local"

    def __init__(
        self,
        *,
        model: str | None = None,
        instruct: str = DEFAULT_RERANK_INSTRUCT,
        device: str | None = None,
        batch_size: int = DEFAULT_LOCAL_RERANK_BATCH_SIZE,
        max_length: int = DEFAULT_LOCAL_RERANK_MAX_LENGTH,
        local_files_only: bool = True,
        torch_dtype: str | None = "auto",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LocalQwen3Reranker requires torch and transformers. "
                "Install project dependencies with `conda run -n qream-rag pip install -r requirements.txt`."
            ) from error

        self.model = model or os.getenv("LOCAL_RERANK_MODEL", DEFAULT_LOCAL_RERANK_MODEL)
        model_path = _resolve_local_model_path(self.model) if local_files_only else self.model
        self.instruct = instruct
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.device = _resolve_local_device(device, torch)
        self._torch = torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Local reranker tokenizer has no pad_token or eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        resolved_dtype = _resolve_torch_dtype(torch_dtype, torch, self.device)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        model_obj = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model_obj.eval()
        model_obj.to(self.device)
        if getattr(model_obj.config, "pad_token_id", None) is None:
            model_obj.config.pad_token_id = tokenizer.pad_token_id

        self._tokenizer = tokenizer
        self._model = model_obj
        self._token_false_id = _required_token_id(tokenizer, "no")
        self._token_true_id = _required_token_id(tokenizer, "yes")
        self._prefix_tokens = tokenizer.encode(_local_rerank_prefix(), add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(LOCAL_RERANK_SUFFIX, add_special_tokens=False)
        self._content_max_length = max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        if self._content_max_length <= 0:
            raise ValueError(
                "max_length is too small for the Qwen3 reranker chat template. "
                f"Use a value greater than {len(self._prefix_tokens) + len(self._suffix_tokens)}."
            )

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedDoc]:
        if not docs:
            return []
        scores = self.score(query, docs)
        return local_scores_to_reranked_docs(docs, scores, model=self.model, top_n=top_n)

    def score(self, query: str, docs: list[RetrievedDoc]) -> list[float]:
        """Return local reranker probabilities for query/document pairs."""

        scores: list[float] = []
        texts = [
            local_reranker_input_text(
                instruct=self.instruct,
                query=query,
                document=reranker_document_text(doc),
            )
            for doc in docs
        ]
        for start in range(0, len(texts), self.batch_size):
            scores.extend(self._score_texts(texts[start : start + self.batch_size]))
        return scores

    def _score_texts(self, texts: list[str]) -> list[float]:
        tokenized = self._tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._content_max_length,
        )
        input_ids = [
            self._prefix_tokens + ids + self._suffix_tokens
            for ids in tokenized["input_ids"]
        ]
        inputs = self._tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with self._torch.no_grad():
            logits = self._model(**inputs).logits[:, -1, :]
            false_logits = logits[:, self._token_false_id]
            true_logits = logits[:, self._token_true_id]
            yes_no_logits = self._torch.stack([false_logits, true_logits], dim=1)
            log_probs = self._torch.nn.functional.log_softmax(yes_no_logits, dim=1)
            return log_probs[:, 1].exp().detach().cpu().tolist()


def local_reranker_input_text(*, instruct: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {document}"


def local_scores_to_reranked_docs(
    docs: list[RetrievedDoc],
    scores: list[float],
    *,
    model: str,
    top_n: int | None = None,
) -> list[RetrievedDoc]:
    if len(docs) != len(scores):
        raise ValueError("docs and scores must have the same length.")

    max_results = _validate_top_n(top_n, len(docs))
    ranked_indexes = sorted(range(len(docs)), key=lambda index: (-scores[index], index))
    return [
        _copy_reranked_doc(
            docs[index],
            model=model,
            score=float(scores[index]),
            reranker_rank=rank,
            backend="local",
        )
        for rank, index in enumerate(ranked_indexes[:max_results], start=1)
    ]


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
    backend: str | None = None,
) -> RetrievedDoc:
    metadata = dict(doc.metadata)
    metadata["reranker_model"] = model
    metadata["reranker_score"] = score
    metadata["reranker_rank"] = reranker_rank
    metadata["pre_rerank_rank"] = doc.rank
    if backend is not None:
        metadata["reranker_backend"] = backend
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


def _local_rerank_prefix() -> str:
    return f"<|im_start|>system\n{LOCAL_RERANK_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"


def _resolve_local_device(device: str | None, torch_module: Any) -> str:
    if device:
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_local_model_path(model: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path)

    cache_root = _default_hf_hub_cache()
    cached_model_dir = cache_root / f"models--{model.replace('/', '--')}"
    if not cached_model_dir.exists():
        return model

    ref_path = cached_model_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cached_model_dir / "snapshots" / revision
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots_dir = cached_model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return model


def _default_hf_hub_cache() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.getenv("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub")
    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_torch_dtype(dtype_name: str | None, torch_module: Any, device: str) -> Any | None:
    if dtype_name is None or dtype_name == "":
        return None

    normalized = dtype_name.lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            return torch_module.float16
        return None

    dtype_by_name = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in dtype_by_name:
        raise ValueError("--local-reranker-dtype must be one of auto, float16, bfloat16, float32.")
    return dtype_by_name[normalized]


def _required_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int):
        raise ValueError(f"Local reranker tokenizer cannot encode required token: {token!r}")
    return token_id
