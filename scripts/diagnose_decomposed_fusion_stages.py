# 中文说明：诊断查询分解后 BM25/Dense 原始候选池与 RRF 融合后候选池的证据召回差异。
"""Compare pre-fusion and post-fusion evidence recall for decomposed retrieval."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import HotpotQASample, RetrievedDoc
from src.evaluation.evidence_metrics import (
    evidence_full_hit_at_k,
    evidence_hit_at_k,
    evidence_metrics,
    evidence_recall_at_k,
    gold_evidence_titles,
)
from src.retrieval.decomposed_hybrid import QueryRetrievedDocs, fuse_decomposed_rrf_ranked_docs
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.dense_backend import add_dense_backend_args, make_dense_store_from_args
from src.retrieval.elasticsearch_bm25 import DEFAULT_ELASTICSEARCH_INDEX, DEFAULT_ELASTICSEARCH_URL
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT
from src.retrieval.query_decomposition import LLMQueryDecomposer, QueryDecompositionCache
from src.utils.io import write_jsonl

from scripts.diagnose_decomposed_hybrid_retrieval import (
    _decompose_sample,
    _ensure_global_inputs_exist,
    _make_bm25_retriever,
    _make_decomposer,
    add_decomposition_args,
)
from scripts.diagnose_hybrid_retrieval import iter_batches, iter_sampled_questions

DEFAULT_FUSION_KS = (5, 10, 20, 50, 100)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-backend", choices=["rank_bm25", "elasticsearch"], default="rank_bm25")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/decomposed_fusion_stages_top50.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-strategy", choices=["head", "uniform"], default="head")
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=50)
    parser.add_argument("--fusion-ks", default="5,10,20,50,100")
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


def main() -> None:
    args = parse_args()
    fusion_ks = parse_ks(args.fusion_ks)
    _validate_args(args, fusion_ks=fusion_ks)

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
    records = _run_decomposed_fusion_stage_diagnostic(
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
        fusion_ks=fusion_ks,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} decomposed fusion stage diagnostics to {args.output}")


def _run_decomposed_fusion_stage_diagnostic(
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
    fusion_ks: tuple[int, ...] = DEFAULT_FUSION_KS,
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

            bm25_pool = _unique_docs_in_order(
                result.docs for result in query_results if result.source == "bm25"
            )
            dense_pool = _unique_docs_in_order(
                result.docs for result in query_results if result.source == "dense"
            )
            union_pool = _unique_docs_in_order(result.docs for result in query_results)
            fused_docs = fuse_decomposed_rrf_ranked_docs(
                query_results=query_results,
                original_query=sample.question,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
                title_boost_weight=title_boost_weight,
            )

            metrics = {}
            metrics.update(_pool_metrics("prefusion_bm25_pool", bm25_pool, sample.supporting_facts))
            metrics.update(_pool_metrics("prefusion_dense_pool", dense_pool, sample.supporting_facts))
            metrics.update(_pool_metrics("prefusion_union_pool", union_pool, sample.supporting_facts))
            metrics.update(
                _prefix_metrics(
                    "fusion",
                    evidence_metrics(
                        [doc.to_dict() for doc in fused_docs],
                        sample.supporting_facts,
                        ks=fusion_ks,
                    ),
                )
            )
            metrics["decomposition_error"] = float(decomposition.error is not None)

            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} decomposed fusion stage queries")

            llm_calls = 0 if decomposition.from_cache else 1
            yield {
                "id": sample.id,
                "question": sample.question,
                "type": sample.type,
                "level": sample.level,
                "gold_answer": sample.answer,
                "gold_supporting_facts": sample.supporting_facts,
                "pred_answer": "",
                "retrieved_docs": [doc.to_dict() for doc in fused_docs],
                "pred_citations": [],
                "metrics": metrics,
                "cost": {
                    "llm_calls": llm_calls,
                    "decomposition_model": decomposition.model,
                    "decomposition_query_count": len(decomposition.queries),
                    "bm25_query_count": len(decomposition.queries),
                    "dense_query_count": len(decomposition.queries),
                    "bm25_pool_doc_count": len(bm25_pool),
                    "dense_pool_doc_count": len(dense_pool),
                    "union_pool_doc_count": len(union_pool),
                    "fusion_doc_count": len(fused_docs),
                },
                "route_initial": {
                    "route": "retrieval_diagnostic",
                    "confidence": 1.0,
                    "reason": "decomposed_fusion_stage_diagnostic",
                    "retrieval_mode": "decomposed_prefusion_vs_fusion",
                },
                "route_final": "retrieval_diagnostic",
                "was_upgraded": False,
                "agent_outputs": {
                    "decomposition": decomposition.to_dict(),
                    "prefusion": _prefusion_summary(
                        query_results=query_results,
                        gold_supporting_facts=sample.supporting_facts,
                        bm25_pool=bm25_pool,
                        dense_pool=dense_pool,
                        union_pool=union_pool,
                    ),
                },
            }


def parse_ks(value: str) -> tuple[int, ...]:
    ks: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        ks.append(int(part))
    if not ks:
        raise ValueError("--fusion-ks must contain at least one positive integer.")
    return tuple(dict.fromkeys(ks))


def _pool_metrics(prefix: str, docs: list[RetrievedDoc], gold_supporting_facts: list[list]) -> dict[str, float]:
    k = len(docs)
    return {
        f"{prefix}_doc_count": float(k),
        f"{prefix}_evidence_recall": evidence_recall_at_k(docs, gold_supporting_facts, k=k),
        f"{prefix}_evidence_hit": evidence_hit_at_k(docs, gold_supporting_facts, k=k),
        f"{prefix}_evidence_full_hit": evidence_full_hit_at_k(docs, gold_supporting_facts, k=k),
    }


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _unique_docs_in_order(doc_lists: Iterable[list[RetrievedDoc]]) -> list[RetrievedDoc]:
    docs: list[RetrievedDoc] = []
    seen: set[str] = set()
    for doc_list in doc_lists:
        for doc in doc_list:
            if doc.doc_id in seen:
                continue
            docs.append(doc)
            seen.add(doc.doc_id)
    return docs


def _prefusion_summary(
    *,
    query_results: list[QueryRetrievedDocs],
    gold_supporting_facts: list[list],
    bm25_pool: list[RetrievedDoc],
    dense_pool: list[RetrievedDoc],
    union_pool: list[RetrievedDoc],
) -> dict:
    gold_titles = gold_evidence_titles(gold_supporting_facts)
    return {
        "bm25_pool_doc_count": len(bm25_pool),
        "dense_pool_doc_count": len(dense_pool),
        "union_pool_doc_count": len(union_pool),
        "gold_titles": sorted(gold_titles),
        "bm25_gold_titles": _found_gold_titles(bm25_pool, gold_titles),
        "dense_gold_titles": _found_gold_titles(dense_pool, gold_titles),
        "union_gold_titles": _found_gold_titles(union_pool, gold_titles),
        "query_results": [
            {
                "query_index": result.query_index,
                "query": result.query,
                "source": result.source,
                "doc_count": len(result.docs),
                "gold_titles": _found_gold_titles(result.docs, gold_titles),
            }
            for result in query_results
        ],
    }


def _found_gold_titles(docs: list[RetrievedDoc], gold_titles: set[str]) -> list[str]:
    found = {doc.title.strip().lower() for doc in docs} & gold_titles
    return sorted(found)


def _validate_args(args: argparse.Namespace, *, fusion_ks: tuple[int, ...]) -> None:
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
    if any(k <= 0 for k in fusion_ks):
        raise ValueError("--fusion-ks values must be positive.")
    if args.title_boost_weight < 0:
        raise ValueError("--title-boost-weight must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")


if __name__ == "__main__":
    main()
