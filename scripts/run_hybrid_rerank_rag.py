# 中文说明：运行 Hybrid + DashScope rerank + DashScope 生成答案的主 baseline。
"""Run Hybrid + DashScope rerank RAG answer generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import (
    iter_processed_hotpotqa_questions,
    load_documents_jsonl,
)
from src.data.schema import HotpotQASample, PipelineResult, RetrievedDoc
from src.evaluation.answer_metrics import answer_metrics
from src.evaluation.evidence_metrics import evidence_metrics
from src.pipeline.standard_rag import select_sentence_citations
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.hybrid import DEFAULT_TITLE_BOOST_WEIGHT, fuse_rrf_ranked_docs
from src.retrieval.milvus_store import MilvusHotpotStore
from src.retrieval.reranker import (
    DEFAULT_RERANK_INSTRUCT,
    DashScopeReranker,
    fallback_rerank_docs,
)
from src.utils.io import write_jsonl
from src.utils.llm_client import (
    AliyunDashScopeClient,
    GenerationResult,
    LLMClient,
    MockLLMClient,
)
from src.utils.text import simple_tokenize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--corpus", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--bm25-cache", default="data/indexes/hotpotqa_global/bm25.pkl")
    parser.add_argument("--rebuild-bm25-cache", action="store_true")
    parser.add_argument("--output", default="outputs/predictions/rerank_rag_global_top10.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--hybrid-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=50)
    parser.add_argument("--answer-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--title-boost-weight", type=float, default=DEFAULT_TITLE_BOOST_WEIGHT)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
    parser.add_argument("--reranker-url", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--reranker-instruct", default=DEFAULT_RERANK_INSTRUCT)
    parser.add_argument("--llm", choices=["aliyun", "mock"], default="aliyun")
    parser.add_argument("--api-timeout-seconds", type=float, default=None)
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument("--api-retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--api-min-request-interval-seconds", type=float, default=None)
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

    reranker = DashScopeReranker(
        url=args.reranker_url,
        model=args.reranker_model,
        instruct=args.reranker_instruct,
        timeout_seconds=args.api_timeout_seconds,
        max_retries=args.api_max_retries,
        retry_backoff_seconds=args.api_retry_backoff_seconds,
        min_request_interval_seconds=args.api_min_request_interval_seconds,
    )
    llm_client = _make_llm_client(args)
    if not reranker.api_key:
        raise ValueError("DASHSCOPE_API_KEY is empty. Fill it in .env before running rerank RAG.")

    samples = iter_processed_hotpotqa_questions(questions_path, limit=args.limit)
    records = _run_rerank_rag(
        samples=samples,
        bm25_retriever=bm25_retriever,
        embedder=embedder,
        store=store,
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
    print(f"wrote {count} rerank RAG predictions to {args.output}")


def _run_rerank_rag(
    *,
    samples: Iterable[HotpotQASample],
    bm25_retriever: BM25Retriever,
    embedder,
    store,
    reranker,
    llm_client: LLMClient,
    bm25_top_k: int,
    dense_top_k: int,
    hybrid_top_k: int,
    rerank_top_n: int,
    answer_top_k: int,
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
                reranked_docs = reranker.rerank(sample.question, hybrid_docs, top_n=rerank_top_n)
            except Exception as error:  # noqa: BLE001 - a single reranker failure should not stop the run.
                rerank_error = 1.0
                reranked_docs = fallback_rerank_docs(hybrid_docs, top_n=rerank_top_n)
                print(f"rerank failed for sample {sample.id}: {error}")

            answer_docs = reranked_docs[:answer_top_k]
            generation_error: str | None = None
            try:
                generation = answer_from_reranked_docs(sample.question, answer_docs, llm_client)
            except Exception as error:  # noqa: BLE001 - a single LLM failure should not stop the run.
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
            agent_outputs = {
                "reranked_candidate_count": len(reranked_docs),
                "answer_doc_count": len(answer_docs),
            }
            if generation_error is not None:
                metrics["llm_error"] = 1.0
                agent_outputs["llm_error"] = generation_error

            cost = dict(generation.cost)
            cost["rerank_calls"] = 1
            cost["reranker_model"] = getattr(reranker, "model", "")
            cost["answer_top_k"] = len(answer_docs)

            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} rerank RAG queries")

            yield PipelineResult(
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
                    "route": "simple",
                    "confidence": 1.0,
                    "reason": "global_hybrid_rerank_rag",
                    "retrieval_mode": "global_hybrid_rerank",
                },
                route_final="simple",
                was_upgraded=False,
                agent_outputs=agent_outputs,
            ).to_dict()


def answer_from_reranked_docs(
    question: str,
    docs: list[RetrievedDoc],
    llm_client: LLMClient,
) -> GenerationResult:
    if isinstance(llm_client, MockLLMClient):
        return llm_client.answer_from_docs(question, docs)

    started = perf_counter()
    context = "\n\n".join(
        f"[{doc.rank}] Title: {doc.title}\nPassage: {doc.text}"
        for doc in docs
    )
    answer = llm_client.generate(
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
        ]
    )
    latency = perf_counter() - started
    answer = answer.strip()
    return GenerationResult(
        answer=answer,
        cost={
            "llm_calls": 1,
            "input_tokens": len(simple_tokenize(question)) + sum(len(simple_tokenize(doc.text)) for doc in docs),
            "output_tokens": len(simple_tokenize(answer)),
            "latency": latency,
            "mock_llm": False,
            "provider": "aliyun_dashscope",
            "model": getattr(llm_client, "model", ""),
        },
    )


def _make_llm_client(args: argparse.Namespace) -> LLMClient:
    if args.llm == "mock":
        return MockLLMClient()
    return AliyunDashScopeClient(
        timeout_seconds=args.api_timeout_seconds,
        max_retries=args.api_max_retries,
        retry_backoff_seconds=args.api_retry_backoff_seconds,
        min_request_interval_seconds=args.api_min_request_interval_seconds,
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "query_batch_size",
        "bm25_top_k",
        "dense_top_k",
        "hybrid_top_k",
        "rerank_top_n",
        "answer_top_k",
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


if __name__ == "__main__":
    main()
