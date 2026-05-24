"""Index building helpers for retrieval backends."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

from src.data.load_hotpotqa import iter_documents_jsonl
from src.data.schema import Document
from src.retrieval.bm25 import corpus_fingerprint
from src.retrieval.dense import Embedder, document_embedding_text
from src.retrieval.milvus_store import MilvusHotpotStore
from src.utils.io import write_json

EMBEDDING_MANIFEST_NAME = "manifest.json"


def iter_batches(items: Iterable[Document], batch_size: int) -> Iterator[list[Document]]:
    batch: list[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_milvus_dense_index(
    *,
    corpus_path: str | Path,
    store: MilvusHotpotStore,
    embedder: Embedder,
    batch_size: int = 16,
    limit: int | None = None,
    drop_existing: bool = False,
    metadata_output_path: str | Path = "data/indexes/hotpotqa_global/dense_build_meta.json",
    embedding_model: str = "BAAI/bge-m3",
    dimension: int = 1024,
) -> dict[str, Any]:
    started = perf_counter()
    store.create_collection(drop_existing=drop_existing)

    inserted_count = 0
    documents = iter_documents_jsonl(corpus_path, limit=limit)
    for batch in iter_batches(documents, batch_size):
        texts = [document_embedding_text(doc.title, doc.text) for doc in batch]
        embeddings = embedder.encode_texts(texts)
        inserted_count += store.insert_documents(batch, embeddings)
        print(f"inserted {inserted_count} dense vectors")

    store.flush()
    metadata = {
        "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "embedding_model": embedding_model,
        "dimension": dimension,
        "collection_name": store.collection_name,
        "inserted_count": inserted_count,
        "batch_size": batch_size,
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(metadata, metadata_output_path)
    return metadata


def export_dense_embeddings(
    *,
    corpus_path: str | Path,
    output_dir: str | Path,
    embedder: Embedder,
    batch_size: int = 16,
    shard_size: int = 8192,
    limit: int | None = None,
    overwrite: bool = False,
    embedding_model: str = "BAAI/bge-m3",
    dimension: int = 1024,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    started = perf_counter()
    output_path = Path(output_dir)
    _prepare_embedding_output_dir(output_path, overwrite=overwrite)

    exported_count = 0
    shard_index = 0
    shard_doc_ids: list[str] = []
    shard_embeddings: list[list[float]] = []
    shards: list[dict[str, Any]] = []

    documents = iter_documents_jsonl(corpus_path, limit=limit)
    for batch in iter_batches(documents, batch_size):
        texts = [document_embedding_text(doc.title, doc.text) for doc in batch]
        embeddings = embedder.encode_texts(texts)
        if len(batch) != len(embeddings):
            raise ValueError("embedder returned a different number of embeddings.")

        for document, embedding in zip(batch, embeddings):
            shard_doc_ids.append(document.doc_id)
            shard_embeddings.append(embedding)
            exported_count += 1

            if len(shard_doc_ids) >= shard_size:
                shard = _write_embedding_shard(
                    output_dir=output_path,
                    shard_index=shard_index,
                    doc_ids=shard_doc_ids,
                    embeddings=shard_embeddings,
                    dimension=dimension,
                )
                shards.append(shard)
                shard_index += 1
                shard_doc_ids = []
                shard_embeddings = []
                print(f"exported {exported_count} dense embeddings")

    if shard_doc_ids:
        shard = _write_embedding_shard(
            output_dir=output_path,
            shard_index=shard_index,
            doc_ids=shard_doc_ids,
            embeddings=shard_embeddings,
            dimension=dimension,
        )
        shards.append(shard)
        print(f"exported {exported_count} dense embeddings")

    metadata = {
        "corpus_fingerprint": portable_corpus_fingerprint(corpus_path),
        "embedding_model": embedding_model,
        "dimension": dimension,
        "exported_count": exported_count,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "shards": shards,
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(metadata, output_path / EMBEDDING_MANIFEST_NAME)
    return metadata


def import_dense_embeddings_to_milvus(
    *,
    corpus_path: str | Path,
    embeddings_dir: str | Path,
    store: MilvusHotpotStore,
    insert_batch_size: int = 1024,
    limit: int | None = None,
    drop_existing: bool = False,
    metadata_output_path: str | Path = "data/indexes/hotpotqa_global/dense_build_meta.json",
) -> dict[str, Any]:
    if insert_batch_size <= 0:
        raise ValueError("insert_batch_size must be positive.")

    started = perf_counter()
    embeddings_path = Path(embeddings_dir)
    manifest = _load_embedding_manifest(embeddings_path)
    expected_fingerprint = manifest.get("corpus_fingerprint")
    actual_fingerprint = portable_corpus_fingerprint(corpus_path)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Embedding manifest corpus fingerprint does not match corpus input. "
            "Use the same corpus.jsonl that was used to export embeddings."
        )

    exported_count = int(manifest.get("exported_count", 0))
    target_count = min(exported_count, limit) if limit is not None else exported_count
    store.create_collection(drop_existing=drop_existing)

    inserted_count = 0
    remaining = target_count
    documents = iter(iter_documents_jsonl(corpus_path, limit=target_count))
    for shard in manifest.get("shards", []):
        if remaining <= 0:
            break

        shard_doc_ids, shard_embeddings = _read_embedding_shard(embeddings_path, shard)
        take_count = min(len(shard_doc_ids), remaining)
        shard_doc_ids = shard_doc_ids[:take_count]
        shard_embeddings = shard_embeddings[:take_count]

        for offset in range(0, take_count, insert_batch_size):
            end = min(offset + insert_batch_size, take_count)
            doc_ids_batch = shard_doc_ids[offset:end]
            embeddings_batch = shard_embeddings[offset:end]
            docs_batch = _next_documents(documents, len(doc_ids_batch))
            _validate_doc_id_alignment(docs_batch, doc_ids_batch, inserted_count)
            inserted_count += store.insert_documents(docs_batch, embeddings_batch.tolist())
            print(f"inserted {inserted_count} dense vectors from exported embeddings")

        remaining -= take_count

    if inserted_count != target_count:
        raise ValueError(
            f"Imported {inserted_count} embeddings, expected {target_count} from manifest."
        )

    store.flush()
    manifest_dimension = manifest.get("dimension")
    metadata = {
        "corpus_fingerprint": actual_fingerprint,
        "embedding_model": manifest.get("embedding_model", ""),
        "dimension": manifest_dimension if manifest_dimension is not None else store.dimension,
        "collection_name": store.collection_name,
        "inserted_count": inserted_count,
        "source_embedding_manifest": str((embeddings_path / EMBEDDING_MANIFEST_NAME).resolve()),
        "elapsed_seconds": perf_counter() - started,
    }
    write_json(metadata, metadata_output_path)
    return metadata


def portable_corpus_fingerprint(path: str | Path) -> dict[str, Any]:
    corpus_path = Path(path)
    stat = corpus_path.stat()
    return {
        "name": corpus_path.name,
        "size": stat.st_size,
        "sha1": _file_sha1(corpus_path),
    }


def _prepare_embedding_output_dir(output_path: Path, *, overwrite: bool) -> None:
    manifest_path = output_path / EMBEDDING_MANIFEST_NAME
    existing_shards = list(output_path.glob("shard_*.npz")) if output_path.exists() else []
    if (manifest_path.exists() or existing_shards) and not overwrite:
        raise FileExistsError(
            f"{output_path} already contains exported embeddings. "
            "Pass --drop-existing to overwrite them."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if manifest_path.exists():
            manifest_path.unlink()
        for shard_path in existing_shards:
            shard_path.unlink()


def _write_embedding_shard(
    *,
    output_dir: Path,
    shard_index: int,
    doc_ids: list[str],
    embeddings: list[list[float]],
    dimension: int,
) -> dict[str, Any]:
    import numpy as np

    embeddings_array = np.asarray(embeddings, dtype="float32")
    if embeddings_array.ndim != 2 or embeddings_array.shape[1] != dimension:
        raise ValueError(
            f"Expected embeddings with dimension {dimension}, got shape {embeddings_array.shape}."
        )

    filename = f"shard_{shard_index:06d}.npz"
    np.savez(
        output_dir / filename,
        doc_ids=np.asarray(doc_ids),
        embeddings=embeddings_array,
    )
    return {
        "file": filename,
        "count": len(doc_ids),
        "first_doc_id": doc_ids[0] if doc_ids else "",
        "last_doc_id": doc_ids[-1] if doc_ids else "",
    }


def _load_embedding_manifest(embeddings_dir: Path) -> dict[str, Any]:
    manifest_path = embeddings_dir / EMBEDDING_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing embedding manifest: {manifest_path}")

    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid embedding manifest: {manifest_path}")
    return manifest


def _read_embedding_shard(embeddings_dir: Path, shard: dict[str, Any]):
    import numpy as np

    shard_path = embeddings_dir / str(shard["file"])
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing embedding shard: {shard_path}")

    data = np.load(shard_path, allow_pickle=False)
    doc_ids = [str(doc_id) for doc_id in data["doc_ids"].tolist()]
    embeddings = data["embeddings"].astype("float32", copy=False)
    expected_count = int(shard.get("count", len(doc_ids)))
    if len(doc_ids) != expected_count or embeddings.shape[0] != expected_count:
        raise ValueError(f"Embedding shard count mismatch: {shard_path}")
    return doc_ids, embeddings


def _next_documents(documents: Iterator[Document], count: int) -> list[Document]:
    batch: list[Document] = []
    for _ in range(count):
        try:
            batch.append(next(documents))
        except StopIteration as error:
            raise ValueError("Corpus ended before all embeddings were imported.") from error
    return batch


def _validate_doc_id_alignment(
    documents: list[Document],
    expected_doc_ids: list[str],
    imported_before_batch: int,
) -> None:
    for offset, (document, expected_doc_id) in enumerate(zip(documents, expected_doc_ids)):
        if document.doc_id != expected_doc_id:
            row_number = imported_before_batch + offset
            raise ValueError(
                "Embedding shard doc_id mismatch at row "
                f"{row_number}: corpus has {document.doc_id}, shard has {expected_doc_id}."
            )


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
