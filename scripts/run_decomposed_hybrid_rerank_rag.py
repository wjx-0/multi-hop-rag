# 中文说明：运行 LLM 查询分解 + Hybrid + Rerank + 生成答案的实验入口。
"""Run query-decomposed Hybrid + rerank RAG answer generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import HotpotQASample, PipelineResult
from src.evaluation.answer_metrics import answer_metrics
from src.evaluation.evidence_metrics import evidence_metrics
from src.pipeline.standard_rag import select_sentence_citations
from src.retrieval.decomposed_hybrid import QueryRetrievedDocs, fuse_decomposed_rrf_ranked_docs
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.dense_backend import add_dense_backend_args, make_dense_store_from_args
from src.retrieval.elasticsearch_bm25 import DEFAULT_ELASTICSEARCH_INDEX, DEFAULT_ELASTICSEARCH_URL
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT
from src.retrieval.query_decomposition import LLMQueryDecomposer, QueryDecompositionCache
from src.retrieval.reranker import (
    DEFAULT_LOCAL_RERANK_BATCH_SIZE,
    DEFAULT_LOCAL_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_INSTRUCT,
    fallback_rerank_docs,
)
from src.utils.io import write_jsonl
from src.utils.llm_client import (
    DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH,
    DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS,
    DEFAULT_LOCAL_LLM_MODEL,
    GenerationResult,
    LLMClient,
)

from scripts.diagnose_decomposed_hybrid_retrieval import (
    _decompose_sample,
    _make_decomposer,
    add_decomposition_args,
)
from scripts.diagnose_hybrid_retrieval import iter_batches, iter_sampled_questions
from scripts.run_hybrid_rerank_rag import (
    _ensure_global_inputs_exist,
    _make_bm25_retriever,
    _make_llm_client,
    _make_reranker,
    answer_from_reranked_docs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-backend", choices=["rank_bm25", "elasticsearch"], default="rank_bm25")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/decomposed_hybrid_rerank_rag_top10.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-strategy", choices=["head", "uniform"], default="head")
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--hybrid-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=50)
    parser.add_argument("--answer-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-boost-weight", type=float, default=DEFAULT_TITLE_BOOST_WEIGHT)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    add_dense_backend_args(parser)
    parser.add_argument("--reranker-backend", choices=["local", "dashscope"], default="local")
    parser.add_argument("--reranker-url", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-instruct", default=DEFAULT_RERANK_INSTRUCT)
    parser.add_argument("--local-reranker-device", default=None)
    parser.add_argument("--local-reranker-batch-size", type=int, default=DEFAULT_LOCAL_RERANK_BATCH_SIZE)
    parser.add_argument("--local-reranker-max-length", type=int, default=DEFAULT_LOCAL_RERANK_MAX_LENGTH)
    parser.add_argument("--local-reranker-dtype", default="auto")
    parser.add_argument("--local-reranker-allow-download", action="store_true")
    parser.add_argument("--llm", choices=["aliyun", "mock", "local"], default="aliyun")
    parser.add_argument("--local-llm-model", default=DEFAULT_LOCAL_LLM_MODEL)
    parser.add_argument("--local-llm-device", default=None)
    parser.add_argument("--local-llm-dtype", default="auto")
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS)
    parser.add_argument("--local-llm-temperature", type=float, default=0.0)
    parser.add_argument("--local-llm-max-input-length", type=int, default=DEFAULT_LOCAL_LLM_MAX_INPUT_LENGTH)
    parser.add_argument("--local-llm-allow-download", action="store_true")
    parser.add_argument("--elasticsearch-url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--elasticsearch-index", default=DEFAULT_ELASTICSEARCH_INDEX)
    add_decomposition_args(parser)
    return parser.parse_args(argv)


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

    reranker = _make_reranker(args)
    llm_client = _make_llm_client(args)
    decomposer = _make_decomposer(
        args,
        local_llm_client=llm_client if _should_reuse_local_llm_for_decomposition(args) else None,
    )
    cache = QueryDecompositionCache(args.decomposition_cache)

    samples = iter_sampled_questions(
        questions_path,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_strategy=args.sample_strategy,
    )
    records = _run_decomposed_rerank_rag(
        samples=samples,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        store=store,
        decomposer=decomposer,
        cache=cache,
        reranker=reranker,
        llm_client=llm_client,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        hybrid_top_k=args.hybrid_top_k,
        rerank_top_n=args.rerank_top_n,
        answer_top_k=args.answer_top_k,
        rrf_k=args.rrf_k,
        title_boost_weight=args.title_boost_weight,
        query_batch_size=args.query_batch_size,
        progress_interval=args.progress_interval,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} decomposed rerank RAG predictions to {args.output}")


def _run_decomposed_rerank_rag(
    *,
    samples: Iterable[HotpotQASample],
    bm25_retriever,
    embedder,
    store,
    decomposer: LLMQueryDecomposer,
    cache: QueryDecompositionCache | None,
    reranker,
    llm_client: LLMClient,
    bm25_top_k: int,
    dense_top_k: int,
    hybrid_top_k: int,
    rerank_top_n: int,
    answer_top_k: int,
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

            hybrid_docs = fuse_decomposed_rrf_ranked_docs(
                query_results=query_results,
                original_query=sample.question,
                final_top_k=hybrid_top_k,
                rrf_k=rrf_k,
                title_boost_weight=title_boost_weight,
            )

            rerank_error = 0.0
            rerank_error_message: str | None = None
            try:
                reranked_docs = reranker.rerank(sample.question, hybrid_docs, top_n=rerank_top_n)
            except Exception as error:  # noqa: BLE001 - keep long evaluation runs alive.
                rerank_error = 1.0
                rerank_error_message = str(error)
                reranked_docs = fallback_rerank_docs(hybrid_docs, top_n=rerank_top_n)
                print(f"rerank failed for sample {sample.id}: {error}")

            answer_docs = reranked_docs[:answer_top_k]
            generation_error: str | None = None
            try:
                generation = answer_from_reranked_docs(sample.question, answer_docs, llm_client)
            except Exception as error:  # noqa: BLE001 - keep long evaluation runs alive.
                generation_error = str(error)
                generation = GenerationResult(
                    answer="",
                    cost={
                        "llm_calls": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency": 0.0,
                        "error": generation_error,
                    },
                )
                print(f"answer generation failed for sample {sample.id}: {error}")

            citations = select_sentence_citations(
                question=sample.question,
                answer=generation.answer,
                retrieved_docs=answer_docs,
                max_citations=2,
            )
            retrieved_doc_dicts = [doc.to_dict() for doc in answer_docs]
            metrics = answer_metrics(generation.answer, sample.answer)
            metrics.update(
                evidence_metrics(
                    retrieved_doc_dicts,
                    sample.supporting_facts,
                    predicted_supporting_facts=citations,
                )
            )
            metrics["rerank_error"] = rerank_error
            metrics["decomposition_error"] = float(decomposition.error is not None)
            if generation_error is not None:
                metrics["llm_error"] = 1.0

            decomposition_llm_calls = 0 if decomposition.from_cache else 1
            answer_llm_calls = int(generation.cost.get("llm_calls", 0) or 0)
            cost = dict(generation.cost)
            cost["decomposition_llm_calls"] = decomposition_llm_calls
            cost["answer_llm_calls"] = answer_llm_calls
            cost["llm_calls"] = decomposition_llm_calls + answer_llm_calls
            cost["decomposition_model"] = decomposition.model
            cost["decomposition_query_count"] = len(decomposition.queries)
            cost["bm25_query_count"] = len(decomposition.queries)
            cost["dense_query_count"] = len(decomposition.queries)
            cost["rerank_calls"] = 1
            cost["reranker_model"] = getattr(reranker, "model", "")
            cost["reranker_backend"] = getattr(reranker, "backend", "")
            cost["answer_top_k"] = len(answer_docs)

            agent_outputs = {
                "decomposition": decomposition.to_dict(),
                "rerank": {
                    "hybrid_candidate_count": len(hybrid_docs),
                    "reranked_candidate_count": len(reranked_docs),
                    "answer_doc_count": len(answer_docs),
                },
            }
            if rerank_error_message is not None:
                agent_outputs["rerank"]["error"] = rerank_error_message
            if generation_error is not None:
                agent_outputs["llm_error"] = generation_error

            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} decomposed rerank RAG queries")

            record = PipelineResult(
                id=sample.id,
                question=sample.question,
                gold_answer=sample.answer,
                gold_supporting_facts=sample.supporting_facts,
                pred_answer=generation.answer,
                retrieved_docs=retrieved_doc_dicts,
                pred_citations=citations,
                metrics=metrics,
                cost=cost,
                route_initial={
                    "route": "complex",
                    "confidence": 1.0,
                    "reason": "decomposed_hybrid_rerank_rag",
                    "retrieval_mode": "decomposed_hybrid_rerank",
                },
                route_final="complex",
                was_upgraded=False,
                agent_outputs=agent_outputs,
            ).to_dict()
            record["type"] = sample.type
            record["level"] = sample.level
            record["retrieval_source"] = "decomposed_hybrid_rerank_rag"
            yield record


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "query_batch_size",
        "bm25_top_k",
        "dense_top_k",
        "hybrid_top_k",
        "rerank_top_n",
        "answer_top_k",
        "local_reranker_batch_size",
        "local_reranker_max_length",
        "local_llm_max_new_tokens",
        "local_llm_max_input_length",
        "decomposition_max_queries",
        "decomposition_max_query_chars",
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


def _should_reuse_local_llm_for_decomposition(args: argparse.Namespace) -> bool:
    if args.decomposition_backend != "local" or args.llm != "local":
        return False
    decomposition_model = args.decomposition_model or args.local_decomposition_model
    return decomposition_model == args.local_llm_model


if __name__ == "__main__":
    main()
