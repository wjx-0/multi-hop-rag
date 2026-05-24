"""Build retrieval indexes and corpus artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_corpus import build_global_deduplicated_corpus
from src.data.load_hotpotqa import iter_hotpotqa
from src.retrieval.dense import SentenceTransformerEmbedder
from src.retrieval.index_builder import (
    build_milvus_dense_index,
    export_dense_embeddings,
    import_dense_embeddings_to_milvus,
)
from src.retrieval.milvus_store import MilvusHotpotStore
from src.utils.io import write_json, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "global-corpus",
            "milvus-dense",
            "dense-embeddings",
            "milvus-import-embeddings",
        ],
        default="global-corpus",
    )
    parser.add_argument("--train-input", default="data/raw/hotpotqa/hotpot_train_v1.1.json")
    parser.add_argument("--dev-input", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus-input", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--corpus-output", default="data/processed/hotpotqa/global/corpus.jsonl")
    parser.add_argument("--questions-output", default="data/processed/hotpotqa/global/questions_dev.jsonl")
    parser.add_argument("--title-index-output", default="data/processed/hotpotqa/global/title_to_doc_ids.json")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument(
        "--embedding-output-dir",
        default="data/indexes/hotpotqa_global/bge_m3_embeddings",
    )
    parser.add_argument(
        "--embedding-input-dir",
        default="data/indexes/hotpotqa_global/bge_m3_embeddings",
    )
    parser.add_argument("--embedding-shard-size", type=int, default=8192)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection-name", default="hotpotqa_global_chunks")
    parser.add_argument("--milvus-metric-type", default="COSINE")
    parser.add_argument("--milvus-insert-batch-size", type=int, default=1024)
    parser.add_argument("--dense-meta-output", default="data/indexes/hotpotqa_global/dense_build_meta.json")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "dense-embeddings":
        embedder = SentenceTransformerEmbedder(
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            normalize=True,
            device=args.embedding_device,
        )
        metadata = export_dense_embeddings(
            corpus_path=args.corpus_input,
            output_dir=args.embedding_output_dir,
            embedder=embedder,
            batch_size=args.embedding_batch_size,
            shard_size=args.embedding_shard_size,
            limit=args.limit,
            overwrite=args.drop_existing,
            embedding_model=args.embedding_model,
            dimension=args.embedding_dimension,
        )
        print(
            "exported "
            f"{metadata['exported_count']} dense embeddings to {args.embedding_output_dir}"
        )
        return

    if args.mode == "milvus-import-embeddings":
        store = MilvusHotpotStore(
            uri=args.milvus_uri,
            token=args.milvus_token,
            collection_name=args.milvus_collection_name,
            dimension=args.embedding_dimension,
            metric_type=args.milvus_metric_type,
        )
        metadata = import_dense_embeddings_to_milvus(
            corpus_path=args.corpus_input,
            embeddings_dir=args.embedding_input_dir,
            store=store,
            insert_batch_size=args.milvus_insert_batch_size,
            limit=args.limit,
            drop_existing=args.drop_existing,
            metadata_output_path=args.dense_meta_output,
        )
        print(
            "imported "
            f"{metadata['inserted_count']} exported embeddings into {args.milvus_collection_name}"
        )
        return

    if args.mode == "milvus-dense":
        store = MilvusHotpotStore(
            uri=args.milvus_uri,
            token=args.milvus_token,
            collection_name=args.milvus_collection_name,
            dimension=args.embedding_dimension,
            metric_type=args.milvus_metric_type,
        )
        embedder = SentenceTransformerEmbedder(
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            normalize=True,
            device=args.embedding_device,
        )
        metadata = build_milvus_dense_index(
            corpus_path=args.corpus_input,
            store=store,
            embedder=embedder,
            batch_size=args.embedding_batch_size,
            limit=args.limit,
            drop_existing=args.drop_existing,
            metadata_output_path=args.dense_meta_output,
            embedding_model=args.embedding_model,
            dimension=args.embedding_dimension,
        )
        print(f"built dense Milvus index with {metadata['inserted_count']} vectors")
        return

    documents, dev_questions, title_to_doc_ids = build_global_deduplicated_corpus(
        train_samples=iter_hotpotqa(args.train_input, limit=args.limit_train),
        dev_samples=iter_hotpotqa(args.dev_input, limit=args.limit_dev),
    )

    corpus_count = write_jsonl((document.to_dict() for document in documents), args.corpus_output)
    question_count = write_jsonl(dev_questions, args.questions_output)
    write_json(title_to_doc_ids, args.title_index_output)

    print(f"wrote {corpus_count} global corpus documents to {args.corpus_output}")
    print(f"wrote {question_count} dev questions to {args.questions_output}")
    print(f"wrote title index with {len(title_to_doc_ids)} titles to {args.title_index_output}")


if __name__ == "__main__":
    main()
