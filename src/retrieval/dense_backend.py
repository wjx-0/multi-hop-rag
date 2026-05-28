"""Factory helpers for selectable dense retrieval stores."""

from __future__ import annotations

import argparse

from src.retrieval.faiss_store import (
    DEFAULT_FAISS_DOCSTORE_PATH,
    DEFAULT_FAISS_EF_SEARCH,
    DEFAULT_FAISS_INDEX_PATH,
    FAISSHotpotStore,
)
from src.retrieval.milvus_store import MilvusHotpotStore


def add_dense_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-backend", choices=["milvus", "faiss"], default="milvus")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
    parser.add_argument("--faiss-index", default=DEFAULT_FAISS_INDEX_PATH)
    parser.add_argument("--faiss-docstore", default=DEFAULT_FAISS_DOCSTORE_PATH)
    parser.add_argument("--faiss-metric-type", default="COSINE")
    parser.add_argument("--faiss-ef-search", type=int, default=DEFAULT_FAISS_EF_SEARCH)


def make_dense_store_from_args(args: argparse.Namespace):
    if args.dense_backend == "faiss":
        print(f"loading FAISS index {args.faiss_index}")
        store = FAISSHotpotStore(
            index_path=args.faiss_index,
            docstore_path=args.faiss_docstore,
            metric_type=args.faiss_metric_type,
            ef_search=args.faiss_ef_search,
        )
        print(f"FAISS index loaded with {store.count()} documents")
        return store

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
    return store
