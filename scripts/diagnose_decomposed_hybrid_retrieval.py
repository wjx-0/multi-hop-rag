# 中文说明：批量诊断 LLM 查询分解 + Hybrid 检索效果，不调用回答 LLM。
"""Run query-decomposed BM25 + dense hybrid retrieval diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import load_documents_jsonl
from src.data.schema import HotpotQASample
from src.evaluation.evidence_metrics import evidence_metrics
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache
from src.retrieval.decomposed_hybrid import QueryRetrievedDocs, fuse_decomposed_rrf_ranked_docs
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.dense_backend import add_dense_backend_args, make_dense_store_from_args
from src.retrieval.elasticsearch_bm25 import (
    DEFAULT_ELASTICSEARCH_INDEX,
    DEFAULT_ELASTICSEARCH_URL,
    ElasticsearchBM25Retriever,
)
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT
from src.retrieval.query_decomposition import (
    DEFAULT_DECOMPOSITION_CACHE,
    DEFAULT_DECOMPOSITION_MAX_QUERIES,
    DEFAULT_DECOMPOSITION_MAX_QUERY_CHARS,
    DEFAULT_DECOMPOSITION_QUERY_MODE,
    DECOMPOSITION_QUERY_MODES,
    LLMQueryDecomposer,
    QueryDecompositionCache,
    QueryDecompositionResult,
)
from src.utils.io import write_jsonl
from src.utils.llm_client import (
    AliyunDashScopeClient,
    DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH,
    DEFAULT_LOCAL_LLM_MODEL,
    LLMClient,
    LocalTransformersLLMClient,
)

from scripts.diagnose_hybrid_retrieval import iter_batches, iter_sampled_questions

DEFAULT_LOCAL_DECOMPOSITION_MAX_NEW_TOKENS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-backend", choices=["rank_bm25", "elasticsearch"], default="rank_bm25")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/decomposed_hybrid_retrieval_top50.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-strategy", choices=["head", "uniform"], default="head")
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-boost-weight", type=float, default=DEFAULT_TITLE_BOOST_WEIGHT)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    add_dense_backend_args(parser)
    parser.add_argument("--elasticsearch-url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--elasticsearch-index", default=DEFAULT_ELASTICSEARCH_INDEX)
    add_decomposition_args(parser)
    return parser.parse_args(argv)


def add_decomposition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decomposition-backend", choices=["aliyun", "local"], default="aliyun")
    parser.add_argument("--decomposition-model", default=None)
    parser.add_argument("--decomposition-max-queries", type=int, default=DEFAULT_DECOMPOSITION_MAX_QUERIES)
    parser.add_argument("--decomposition-max-query-chars", type=int, default=DEFAULT_DECOMPOSITION_MAX_QUERY_CHARS)
    parser.add_argument(
        "--decomposition-query-mode",
        choices=DECOMPOSITION_QUERY_MODES,
        default=DEFAULT_DECOMPOSITION_QUERY_MODE,
    )
    parser.add_argument("--decomposition-cache", default=DEFAULT_DECOMPOSITION_CACHE)
    parser.add_argument("--local-decomposition-model", default=DEFAULT_LOCAL_LLM_MODEL)
    parser.add_argument("--local-decomposition-device", default=None)
    parser.add_argument("--local-decomposition-dtype", default="auto")
    parser.add_argument(
        "--local-decomposition-max-new-tokens",
        type=int,
        default=DEFAULT_LOCAL_DECOMPOSITION_MAX_NEW_TOKENS,
    )
    parser.add_argument("--local-decomposition-temperature", type=float, default=0.0)
    parser.add_argument("--local-decomposition-max-input-length", type=int, default=DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH)
    parser.add_argument("--local-decomposition-allow-download", action="store_true")
    parser.add_argument("--api-timeout-seconds", type=float, default=None)
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument("--api-retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--api-min-request-interval-seconds", type=float, default=None)


def main() -> None:
    args = parse_args()
    _validate_args(args)

    corpus_path = Path(args.corpus)
    questions_path = Path(args.questions)
    _ensure_global_inputs_exist(corpus_path, questions_path)

    bm25_retriever = _make_bm25_retriever(args, corpus_path)
    print(f"loading embedding model {args.embedding_model}")
    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        normalize=True,
        device=args.embedding_device,
    )
    print("embedding model loaded")
    store = make_dense_store_from_args(args)

    decomposer = _make_decomposer(args)
    cache = QueryDecompositionCache(args.decomposition_cache)

    samples = iter_sampled_questions(
        questions_path,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_strategy=args.sample_strategy,
    )
    records = _run_decomposed_hybrid_diagnostic(
        samples=samples,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        store=store,
        decomposer=decomposer,
        cache=cache,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        title_boost_weight=args.title_boost_weight,
        query_batch_size=args.query_batch_size,
        progress_interval=args.progress_interval,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} decomposed hybrid retrieval diagnostics to {args.output}")


def _run_decomposed_hybrid_diagnostic(
    *,
    samples: Iterable[HotpotQASample],
    bm25_retriever,
    embedder,
    store,
    decomposer: LLMQueryDecomposer,
    cache: QueryDecompositionCache | None,
    bm25_top_k: int,
    dense_top_k: int,
    final_top_k: int,
    rrf_k: int,
    title_boost_weight: float,
    query_batch_size: int,
    progress_interval: int,
) -> Iterator[dict]:
    processed = 0
    for batch in iter_batches(samples, query_batch_size):
        decompositions = [
            _decompose_sample(sample=sample, decomposer=decomposer, cache=cache)
            for sample in batch
        ]
        flattened_queries = [query for result in decompositions for query in result.queries]
        embeddings = embedder.encode_texts(flattened_queries)
        offset = 0

        for sample, decomposition in zip(batch, decompositions):
            query_embeddings = embeddings[offset : offset + len(decomposition.queries)]
            offset += len(decomposition.queries)

            query_results: list[QueryRetrievedDocs] = []
            for query_index, (query, embedding) in enumerate(zip(decomposition.queries, query_embeddings)):
                bm25_docs = bm25_retriever.retrieve(query, top_k=bm25_top_k)
                dense_docs = store.search(embedding, top_k=dense_top_k)
                query_results.append(
                    QueryRetrievedDocs(
                        query_index=query_index,
                        query=query,
                        source="bm25",
                        docs=bm25_docs,
                    )
                )
                query_results.append(
                    QueryRetrievedDocs(
                        query_index=query_index,
                        query=query,
                        source="dense",
                        docs=dense_docs,
                    )
                )

            retrieved_docs = fuse_decomposed_rrf_ranked_docs(
                query_results=query_results,
                original_query=sample.question,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
                title_boost_weight=title_boost_weight,
            )
            retrieved_doc_dicts = [doc.to_dict() for doc in retrieved_docs]
            metrics = evidence_metrics(retrieved_doc_dicts, sample.supporting_facts)
            metrics["decomposition_error"] = float(decomposition.error is not None)

            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} decomposed hybrid retrieval queries")

            llm_calls = 0 if decomposition.from_cache else 1
            yield {
                "id": sample.id,
                "question": sample.question,
                "type": sample.type,
                "level": sample.level,
                "gold_answer": sample.answer,
                "gold_supporting_facts": sample.supporting_facts,
                "pred_answer": "",
                "retrieved_docs": retrieved_doc_dicts,
                "pred_citations": [],
                "metrics": metrics,
                "cost": {
                    "llm_calls": llm_calls,
                    "decomposition_model": decomposition.model,
                    "decomposition_query_count": len(decomposition.queries),
                    "bm25_query_count": len(decomposition.queries),
                    "dense_query_count": len(decomposition.queries),
                },
                "route_initial": {
                    "route": "retrieval_diagnostic",
                    "confidence": 1.0,
                    "reason": "decomposed_global_hybrid_retrieval_diagnostic",
                    "retrieval_mode": "decomposed_global_hybrid",
                },
                "route_final": "retrieval_diagnostic",
                "was_upgraded": False,
                "agent_outputs": {
                    "decomposition": decomposition.to_dict(),
                },
            }


def _decompose_sample(
    *,
    sample: HotpotQASample,
    decomposer: LLMQueryDecomposer,
    cache: QueryDecompositionCache | None,
) -> QueryDecompositionResult:
    query_mode = getattr(decomposer, "query_mode", DEFAULT_DECOMPOSITION_QUERY_MODE)
    cached = (
        cache.get(
            sample_id=sample.id,
            question=sample.question,
            query_mode=query_mode,
        )
        if cache is not None
        else None
    )
    if cached is not None:
        return cached

    result = decomposer.decompose(sample_id=sample.id, question=sample.question)
    if cache is not None:
        cache.put(result)
    return result


def _make_decomposer(
    args: argparse.Namespace,
    *,
    local_llm_client: LLMClient | None = None,
) -> LLMQueryDecomposer:
    if args.decomposition_backend == "local":
        client = local_llm_client
        if client is None:
            model = args.decomposition_model or args.local_decomposition_model
            print(f"loading local decomposition LLM {model}")
            client = LocalTransformersLLMClient(
                model=model,
                device=args.local_decomposition_device,
                torch_dtype=args.local_decomposition_dtype,
                max_new_tokens=args.local_decomposition_max_new_tokens,
                temperature=args.local_decomposition_temperature,
                max_input_length=args.local_decomposition_max_input_length,
                local_files_only=not args.local_decomposition_allow_download,
            )
            print(f"local decomposition LLM loaded on {client.device}")

        return LLMQueryDecomposer(
            llm_client=client,
            model=args.decomposition_model or getattr(client, "model", ""),
            max_queries=args.decomposition_max_queries,
            max_query_chars=args.decomposition_max_query_chars,
            pass_model_arg=False,
            query_mode=args.decomposition_query_mode,
        )

    client = AliyunDashScopeClient(
        timeout_seconds=args.api_timeout_seconds,
        max_retries=args.api_max_retries,
        retry_backoff_seconds=args.api_retry_backoff_seconds,
        min_request_interval_seconds=args.api_min_request_interval_seconds,
    )
    return LLMQueryDecomposer(
        llm_client=client,
        model=args.decomposition_model,
        max_queries=args.decomposition_max_queries,
        max_query_chars=args.decomposition_max_query_chars,
        query_mode=args.decomposition_query_mode,
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "query_batch_size",
        "bm25_top_k",
        "dense_top_k",
        "final_top_k",
        "decomposition_max_queries",
        "decomposition_max_query_chars",
        "local_decomposition_max_new_tokens",
        "local_decomposition_max_input_length",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.title_boost_weight < 0:
        raise ValueError("--title-boost-weight must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")


def _ensure_global_inputs_exist(corpus_path: Path, questions_path: Path) -> None:
    if corpus_path.exists() and questions_path.exists():
        return
    raise FileNotFoundError(
        "Global corpus files are missing. Run "
        "`python scripts/build_hotpotqa_indexes.py --mode global-corpus` first."
    )


def _load_or_build_global_bm25(
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


def _make_bm25_retriever(args: argparse.Namespace, corpus_path: Path):
    if args.bm25_backend == "elasticsearch":
        print(f"using Elasticsearch BM25 index {args.elasticsearch_index}")
        return ElasticsearchBM25Retriever(
            url=args.elasticsearch_url,
            index_name=args.elasticsearch_index,
        )
    return _load_or_build_global_bm25(
        corpus_path=corpus_path,
        cache_path=Path(args.bm25_cache),
        rebuild_cache=args.rebuild_bm25_cache,
    )


if __name__ == "__main__":
    main()
