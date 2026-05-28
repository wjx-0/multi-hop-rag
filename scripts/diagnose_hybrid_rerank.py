# 中文说明：批量诊断 Hybrid top-k 经本地/DashScope reranker 重排后的证据召回。
"""Run hybrid retrieval plus rerank diagnostics without answer generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import (
    iter_processed_hotpotqa_questions,
    load_documents_jsonl,
)
from src.data.schema import HotpotQASample
from src.evaluation.evidence_metrics import evidence_metrics
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.elasticsearch_bm25 import (
    DEFAULT_ELASTICSEARCH_INDEX,
    DEFAULT_ELASTICSEARCH_URL,
    ElasticsearchBM25Retriever,
)
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT, fuse_rrf_ranked_docs
from src.retrieval.milvus_store import MilvusHotpotStore
from src.retrieval.reranker import (
    DEFAULT_LOCAL_RERANK_BATCH_SIZE,
    DEFAULT_LOCAL_RERANK_MAX_LENGTH,
    DEFAULT_LOCAL_RERANK_MODEL,
    DEFAULT_RERANK_INSTRUCT,
    DashScopeReranker,
    LocalQwen3Reranker,
    fallback_rerank_docs,
)
from src.utils.io import write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-backend", choices=["rank_bm25", "elasticsearch"], default="rank_bm25")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/rerank_hybrid_global_top50.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--hybrid-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-boost-weight", type=float, default=DEFAULT_TITLE_BOOST_WEIGHT)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
    parser.add_argument("--reranker-backend", choices=["local", "dashscope"], default="local")
    parser.add_argument("--reranker-url", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-instruct", default=DEFAULT_RERANK_INSTRUCT)
    parser.add_argument("--local-reranker-device", default=None)
    parser.add_argument("--local-reranker-batch-size", type=int, default=DEFAULT_LOCAL_RERANK_BATCH_SIZE)
    parser.add_argument("--local-reranker-max-length", type=int, default=DEFAULT_LOCAL_RERANK_MAX_LENGTH)
    parser.add_argument("--local-reranker-dtype", default="auto")
    parser.add_argument("--local-reranker-allow-download", action="store_true")
    parser.add_argument("--api-timeout-seconds", type=float, default=None)
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument("--api-retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--api-min-request-interval-seconds", type=float, default=None)
    parser.add_argument("--elasticsearch-url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--elasticsearch-index", default=DEFAULT_ELASTICSEARCH_INDEX)
    return parser.parse_args(argv)


def iter_batches(items: Iterable[HotpotQASample], batch_size: int) -> Iterator[list[HotpotQASample]]:
    batch: list[HotpotQASample] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    store = MilvusHotpotStore(
        uri=args.milvus_uri,
        token=args.milvus_token,
        collection_name=args.milvus_collection_name,
        dimension=args.embedding_dimension,
        metric_type=args.milvus_metric_type,
    )
    print(f"loading Milvus collection {args.milvus_collection_name}")
    store.load_collection()
    print("Milvus collection loaded")
    reranker = _make_reranker(args)

    samples = iter_processed_hotpotqa_questions(questions_path, limit=args.limit)
    records = _run_rerank_diagnostic(
        samples=samples,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        store=store,
        reranker=reranker,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        hybrid_top_k=args.hybrid_top_k,
        rerank_top_n=args.rerank_top_n,
        rrf_k=args.rrf_k,
        title_boost_weight=args.title_boost_weight,
        query_batch_size=args.query_batch_size,
        progress_interval=args.progress_interval,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} rerank diagnostics to {args.output}")


def _run_rerank_diagnostic(
    *,
    samples: Iterable[HotpotQASample],
    bm25_retriever: BM25Retriever,
    embedder,
    store,
    reranker,
    bm25_top_k: int,
    dense_top_k: int,
    hybrid_top_k: int,
    rerank_top_n: int,
    rrf_k: int,
    query_batch_size: int,
    progress_interval: int,
    title_boost_weight: float = DEFAULT_TITLE_BOOST_WEIGHT,
) -> Iterator[dict]:
    processed = 0
    for batch in iter_batches(samples, query_batch_size):
        embeddings = embedder.encode_texts([sample.question for sample in batch])
        for sample, embedding in zip(batch, embeddings):
            bm25_docs = bm25_retriever.retrieve(sample.question, top_k=bm25_top_k)
            dense_docs = store.search(embedding, top_k=dense_top_k)
            hybrid_docs = fuse_rrf_ranked_docs(
                query=sample.question,
                bm25_docs=bm25_docs,
                dense_docs=dense_docs,
                final_top_k=hybrid_top_k,
                rrf_k=rrf_k,
                title_boost_weight=title_boost_weight,
            )
            rerank_error = 0.0
            try:
                retrieved_docs = reranker.rerank(sample.question, hybrid_docs, top_n=rerank_top_n)
            except Exception as error:  # noqa: BLE001 - diagnostics should continue after one API failure.
                rerank_error = 1.0
                retrieved_docs = fallback_rerank_docs(hybrid_docs, top_n=rerank_top_n)
                print(f"rerank failed for sample {sample.id}: {error}")

            retrieved_doc_dicts = [doc.to_dict() for doc in retrieved_docs]
            metrics = evidence_metrics(
                retrieved_doc_dicts,
                sample.supporting_facts,
            )
            metrics["rerank_error"] = rerank_error
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} rerank queries")

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
                    "llm_calls": 0,
                    "rerank_calls": 1,
                    "reranker_model": getattr(reranker, "model", ""),
                    "reranker_backend": getattr(reranker, "backend", ""),
                },
                "route_initial": {
                    "route": "retrieval_diagnostic",
                    "confidence": 1.0,
                    "reason": "global_hybrid_rerank_diagnostic",
                    "retrieval_mode": "global_hybrid_rerank",
                },
                "route_final": "retrieval_diagnostic",
                "was_upgraded": False,
                "agent_outputs": {},
            }


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "query_batch_size",
        "bm25_top_k",
        "dense_top_k",
        "hybrid_top_k",
        "rerank_top_n",
        "local_reranker_batch_size",
        "local_reranker_max_length",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.title_boost_weight < 0:
        raise ValueError("--title-boost-weight must be non-negative.")


def _ensure_global_inputs_exist(corpus_path: Path, questions_path: Path) -> None:
    if corpus_path.exists() and questions_path.exists():
        return
    raise FileNotFoundError(
        "Global corpus files are missing. Run "
        "`conda run -n qream-rag python scripts/build_hotpotqa_indexes.py --mode global-corpus` first."
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


def _make_reranker(args: argparse.Namespace):
    if args.reranker_backend == "dashscope":
        reranker = DashScopeReranker(
            url=args.reranker_url,
            model=args.reranker_model,
            instruct=args.reranker_instruct,
            timeout_seconds=args.api_timeout_seconds,
            max_retries=args.api_max_retries,
            retry_backoff_seconds=args.api_retry_backoff_seconds,
            min_request_interval_seconds=args.api_min_request_interval_seconds,
        )
        if not reranker.api_key:
            raise ValueError("DASHSCOPE_API_KEY is empty. Fill it in .env before running DashScope rerank.")
        return reranker

    model = args.reranker_model or DEFAULT_LOCAL_RERANK_MODEL
    print(f"loading local reranker {model}")
    reranker = LocalQwen3Reranker(
        model=args.reranker_model,
        instruct=args.reranker_instruct,
        device=args.local_reranker_device,
        batch_size=args.local_reranker_batch_size,
        max_length=args.local_reranker_max_length,
        local_files_only=not args.local_reranker_allow_download,
        torch_dtype=args.local_reranker_dtype,
    )
    print(f"local reranker loaded on {reranker.device}")
    return reranker


if __name__ == "__main__":
    main()
