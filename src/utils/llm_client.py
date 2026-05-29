"""LLM client abstractions.

Phase 1 uses MockLLMClient so the data/retrieval/evaluation loop can run
without model credentials.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from src.data.schema import RetrievedDoc
from src.utils.text import simple_tokenize

DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS = 64
DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH = 4096


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

    provider = "mock"
    model = "mock"

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

    provider = "aliyun_dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
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

        response_body = self._send_with_retries(request)

        data = json.loads(response_body)
        return data["choices"][0]["message"]["content"]

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
                last_error = RuntimeError(f"DashScope HTTP {error.code}: {error_body}")
                if not _is_retryable_http_status(error.code) or attempt == attempts:
                    raise last_error from error
            except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
                last_error = RuntimeError(f"DashScope request failed: {error}")
                if attempt == attempts:
                    raise last_error from error

            sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
            print(f"DashScope request failed; retrying in {sleep_seconds:.1f}s ({attempt}/{self.max_retries})")
            time.sleep(sleep_seconds)

        raise RuntimeError(f"DashScope request failed after retries: {last_error}")

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = perf_counter() - self._last_request_started_at
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)

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


class LocalTransformersLLMClient:
    """Local Hugging Face causal LM client for answer generation."""

    provider = "local_transformers"

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = "auto",
        max_new_tokens: int = DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        max_input_length: int = DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH,
        local_files_only: bool = True,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if max_input_length <= 0:
            raise ValueError("max_input_length must be positive.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "LocalTransformersLLMClient requires torch and transformers. "
                "Install project dependencies with `python -m pip install -r requirements.txt`."
            ) from error

        self.model = model or os.getenv("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL)
        model_path = _resolve_local_model_path(self.model) if local_files_only else self.model
        self.device = _resolve_local_device(device, torch)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_length = max_input_length
        self.local_files_only = local_files_only
        self._torch = torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Local LLM tokenizer has no pad_token or eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
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

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        max_new_tokens = int(kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_new_tokens)))
        temperature = float(kwargs.pop("temperature", self.temperature))
        do_sample = bool(kwargs.pop("do_sample", temperature > 0))
        input_ids = self._encode_messages(messages)
        attention_mask = self._torch.ones_like(input_ids)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        generation_kwargs.update(kwargs)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        output_ids = generated_ids[0, input_ids.shape[-1]:]
        answer = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return _strip_qwen_thinking(answer).strip()

    def generate_json(self, messages: list[dict[str, str]], schema: Any = None, **kwargs: Any) -> dict[str, Any]:
        content = self.generate(messages, **kwargs)
        return json.loads(_strip_json_fence(content))

    def answer_from_docs(self, question: str, docs: list[RetrievedDoc]) -> GenerationResult:
        started = perf_counter()
        context = "\n\n".join(
            f"[{doc.rank}] Title: {doc.title}\nPassage: {doc.text}"
            for doc in docs
        )
        answer = self.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You answer HotpotQA questions using only the provided evidence. "
                        "Return the shortest answer span when possible. "
                        "For yes/no questions, answer only yes or no. "
                        "If the evidence is insufficient, answer unknown. "
                        "Do not include explanations or citation markers."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nEvidence:\n{context}\n\nAnswer:",
                },
            ],
        )
        latency = perf_counter() - started
        return GenerationResult(
            answer=answer,
            cost={
                "llm_calls": 1,
                "input_tokens": len(simple_tokenize(question)) + sum(len(simple_tokenize(doc.text)) for doc in docs),
                "output_tokens": len(simple_tokenize(answer)),
                "latency": latency,
                "mock_llm": False,
                "provider": self.provider,
                "model": self.model,
            },
        )

    def _encode_messages(self, messages: list[dict[str, str]]):
        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        input_ids = input_ids.to(self.device)
        if input_ids.shape[-1] > self.max_input_length:
            input_ids = input_ids[:, -self.max_input_length:]
        return input_ids


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


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _strip_qwen_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


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
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
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
        raise ValueError("local LLM dtype must be one of auto, float16, bfloat16, float32.")
    return dtype_by_name[normalized]
