# 中文说明：对单个问题执行一次 Milvus dense retrieval 查询，用于 smoke test。
"""Run a dense retrieval query against the Milvus HotpotQA collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from src.retrieval.milvus_store import MilvusHotpotStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    retriever = DenseRetriever(embedder=embedder, store=store)
    results = retriever.retrieve(args.question, top_k=args.top_k)

    print("Question:")
    print(args.question)
    print()
    print("Top dense results:")
    for doc in results:
        print(f"{doc.rank}. {doc.title} | score={doc.score:.4f} | doc_id={doc.doc_id}")
        print(f"   {doc.text[:260]}...")


if __name__ == "__main__":
    main()
