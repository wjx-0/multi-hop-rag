# 中文说明：批量诊断 Milvus dense retrieval 的证据召回，不调用 LLM。
"""Run dense retrieval diagnostics without calling an LLM.只做检索不做llm调用"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import iter_processed_hotpotqa_questions
from src.data.schema import HotpotQASample
from src.evaluation.evidence_metrics import evidence_metrics
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.milvus_store import MilvusHotpotStore
from src.utils.io import write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--output", default="outputs/predictions/dense_retrieval_global_top50.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
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


def main() -> None:
    args = parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive.")

    embedder = SentenceTransformerEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        normalize=True,
        device=args.embedding_device,
    )
    store = MilvusHotpotStore(
        uri=args.milvus_uri,
        token=args.milvus_token,
        collection_name=args.milvus_collection_name,
        dimension=args.embedding_dimension,
        metric_type=args.milvus_metric_type,
    )
    store.load_collection()

    samples = iter_processed_hotpotqa_questions(args.questions, limit=args.limit)
    records = _run_dense_diagnostic(
        samples=samples,
        embedder=embedder,
        store=store,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
        progress_interval=args.progress_interval,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} dense retrieval diagnostics to {args.output}")


def _run_dense_diagnostic(
    *,
    samples: Iterable[HotpotQASample],
    embedder,
    store,
    top_k: int,
    query_batch_size: int,
    progress_interval: int,
) -> Iterator[dict]:
    processed = 0
    for batch in iter_batches(samples, query_batch_size):
        embeddings = embedder.encode_texts([sample.question for sample in batch])
        for sample, embedding in zip(batch, embeddings):
            retrieved_docs = store.search(embedding, top_k=top_k)
            retrieved_doc_dicts = [doc.to_dict() for doc in retrieved_docs]
            metrics = evidence_metrics(
                retrieved_doc_dicts,
                sample.supporting_facts,
            )
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                print(f"processed {processed} dense retrieval queries")

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
                    "reason": "global_dense_retrieval_diagnostic",
                    "retrieval_mode": "global_dense",
                },
                "route_final": "retrieval_diagnostic",
                "was_upgraded": False,
                "agent_outputs": {},
            }


if __name__ == "__main__":
    main()
