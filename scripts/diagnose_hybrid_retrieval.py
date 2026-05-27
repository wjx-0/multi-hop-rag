# 中文说明：批量诊断 BM25 + Dense Hybrid 检索效果，不调用 LLM。
"""Run BM25 + dense hybrid retrieval diagnostics without calling an LLM."""

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
    load_processed_hotpotqa_questions,
)
from src.data.schema import HotpotQASample
from src.evaluation.evidence_metrics import evidence_metrics
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT, fuse_rrf_ranked_docs
from src.retrieval.milvus_store import MilvusHotpotStore
from src.utils.io import write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/hybrid_retrieval_global_top50.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-strategy", choices=["head", "uniform"], default="head")
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-boost-weight", type=float, default=DEFAULT_TITLE_BOOST_WEIGHT)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
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


def iter_sampled_questions(
    path: Path,
    *,
    limit: int | None,
    sample_size: int | None,
    sample_strategy: str,
) -> Iterator[HotpotQASample]:
    sample_count = sample_size if sample_size is not None else limit
    if sample_strategy == "head":
        yield from iter_processed_hotpotqa_questions(path, limit=sample_count)
        return

    if sample_strategy != "uniform":
        raise ValueError(f"unsupported sample strategy: {sample_strategy}")
    if sample_count is None:
        yield from iter_processed_hotpotqa_questions(path)
        return

    questions = load_processed_hotpotqa_questions(path)
    indexes = uniform_sample_indexes(len(questions), sample_count)
    print(f"using uniform question sample: {len(indexes)} / {len(questions)}")
    for index in indexes:
        yield questions[index]


def uniform_sample_indexes(total_count: int, sample_size: int) -> list[int]:
    if total_count < 0:
        raise ValueError("total_count must be non-negative.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if total_count == 0:
        return []
    if sample_size >= total_count:
        return list(range(total_count))
    if sample_size == 1:
        return [total_count // 2]

    indexes: list[int] = []
    previous = -1
    for offset in range(sample_size):
        index = round(offset * (total_count - 1) / (sample_size - 1))
        if index <= previous:
            index = previous + 1
        indexes.append(index)
        previous = index
    return indexes


def main() -> None:
    args = parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive.")
    if args.title_boost_weight < 0:
        raise ValueError("--title-boost-weight must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    corpus_path = Path(args.corpus)
    questions_path = Path(args.questions)
    _ensure_global_inputs_exist(corpus_path, questions_path)

    bm25_retriever = _load_or_build_global_bm25(
        corpus_path=corpus_path,
        cache_path=Path(args.bm25_cache),
        rebuild_cache=args.rebuild_bm25_cache,
    )
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

    samples = iter_sampled_questions(
        questions_path,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_strategy=args.sample_strategy,
    )
    records = _run_hybrid_diagnostic(
        samples=samples,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        store=store,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        title_boost_weight=args.title_boost_weight,
        query_batch_size=args.query_batch_size,
        progress_interval=args.progress_interval,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} hybrid retrieval diagnostics to {args.output}")


def _run_hybrid_diagnostic(
    *,
    samples: Iterable[HotpotQASample],
    bm25_retriever: BM25Retriever,
    embedder,
    store,
    bm25_top_k: int,
    dense_top_k: int,
    final_top_k: int,
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
            retrieved_docs = fuse_rrf_ranked_docs(
                query=sample.question,
                bm25_docs=bm25_docs,
                dense_docs=dense_docs,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
                title_boost_weight=title_boost_weight,
            )
            retrieved_doc_dicts = [doc.to_dict() for doc in retrieved_docs]
            metrics = evidence_metrics(
                retrieved_doc_dicts,
                sample.supporting_facts,
            )
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} hybrid retrieval queries")

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
                "cost": {"llm_calls": 0},
                "route_initial": {
                    "route": "retrieval_diagnostic",
                    "confidence": 1.0,
                    "reason": "global_hybrid_retrieval_diagnostic",
                    "retrieval_mode": "global_hybrid",
                },
                "route_final": "retrieval_diagnostic",
                "was_upgraded": False,
                "agent_outputs": {},
            }


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


if __name__ == "__main__":
    main()
