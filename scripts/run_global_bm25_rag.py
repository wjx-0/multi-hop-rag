# 中文说明：运行 global BM25 Standard RAG baseline，支持 DashScope API 或 mock LLM。
"""Run Standard RAG over the default global HotpotQA corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import (
    iter_hotpotqa,
    iter_processed_hotpotqa_questions,
    load_documents_jsonl,
)
from src.pipeline.standard_rag import StandardRAGPipeline
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache
from src.utils.io import write_jsonl
from src.utils.llm_client import AliyunDashScopeClient, MockLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["global", "per-sample"], default="global")
    parser.add_argument("--input", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/standard_rag_global_bm25_api_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", choices=["aliyun", "mock"], default="aliyun")
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument("--api-retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--api-min-request-interval-seconds", type=float, default=None)
    return parser.parse_args()


def ensure_global_inputs_exist(corpus_path: Path, questions_path: Path) -> None:
    if corpus_path.exists() and questions_path.exists():
        return

    raise FileNotFoundError(
        "Global corpus files are missing. Run "
        "`conda run -n qream-rag python scripts/build_hotpotqa_indexes.py --mode global-corpus` first."
    )


def load_or_build_global_bm25(
    *,
    corpus_path: Path,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> BM25Retriever:
    if not rebuild_cache:
        cached_retriever = load_bm25_cache(cache_path, corpus_path=corpus_path)
        if cached_retriever is not None:
            print(f"loaded BM25 cache from {cache_path}")
            return cached_retriever

    documents = load_documents_jsonl(corpus_path)
    retriever = BM25Retriever(documents)
    save_bm25_cache(retriever, cache_path, corpus_path=corpus_path)
    print(f"wrote BM25 cache to {cache_path}")
    return retriever


def main() -> None:
    args = parse_args()
    llm_client = (
        AliyunDashScopeClient(
            max_retries=args.api_max_retries,
            retry_backoff_seconds=args.api_retry_backoff_seconds,
            min_request_interval_seconds=args.api_min_request_interval_seconds,
        )
        if args.llm == "aliyun"
        else MockLLMClient()
    )

    if args.mode == "global":
        corpus_path = Path(args.corpus)
        questions_path = Path(args.questions)
        cache_path = Path(args.bm25_cache)
        ensure_global_inputs_exist(corpus_path, questions_path)

        retriever = load_or_build_global_bm25(
            corpus_path=corpus_path,
            cache_path=cache_path,
            rebuild_cache=args.rebuild_bm25_cache,
        )
        pipeline = StandardRAGPipeline(
            top_k=args.top_k,
            llm_client=llm_client,
            retriever=retriever,
        )
        samples = iter_processed_hotpotqa_questions(questions_path, limit=args.limit)
    else:
        pipeline = StandardRAGPipeline(top_k=args.top_k, llm_client=llm_client)
        samples = iter_hotpotqa(args.input, limit=args.limit)

    results = (pipeline.run(sample).to_dict() for sample in samples)
    count = write_jsonl(results, args.output)
    print(f"wrote {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
