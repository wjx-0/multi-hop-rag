import json

import pytest

from src.data.schema import Document
from src.retrieval.index_builder import (
    build_faiss_dense_index,
    build_milvus_dense_index,
    export_dense_embeddings,
    import_dense_embeddings_to_faiss,
    import_dense_embeddings_to_milvus,
    iter_batches,
)


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class FakeStore:
    collection_name = "fake_collection"

    def __init__(self):
        self.created = False
        self.drop_existing = None
        self.rows = []

    def create_collection(self, *, drop_existing=False):
        self.created = True
        self.drop_existing = drop_existing

    def insert_documents(self, documents, embeddings):
        self.rows.extend(zip(documents, embeddings))
        return len(documents)

    def flush(self):
        pass


def test_iter_batches_keeps_final_partial_batch():
    assert list(iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_build_milvus_dense_index_with_fake_store(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    meta_path = tmp_path / "dense_meta.json"
    records = [
        Document("d1", "Title 1", "Text 1", ["Text 1"]).to_dict(),
        Document("d2", "Title 2", "Text 2", ["Text 2"]).to_dict(),
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    store = FakeStore()

    metadata = build_milvus_dense_index(
        corpus_path=corpus_path,
        store=store,
        embedder=FakeEmbedder(),
        batch_size=1,
        limit=None,
        drop_existing=True,
        metadata_output_path=meta_path,
        embedding_model="fake",
        dimension=2,
    )

    assert store.created is True
    assert store.drop_existing is True
    assert len(store.rows) == 2
    assert metadata["inserted_count"] == 2
    assert metadata["embedding_model"] == "fake"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["dimension"] == 2


def test_export_dense_embeddings_writes_shards_and_manifest(tmp_path):
    corpus_path = _write_test_corpus(tmp_path)
    output_dir = tmp_path / "embeddings"

    metadata = export_dense_embeddings(
        corpus_path=corpus_path,
        output_dir=output_dir,
        embedder=FakeEmbedder(),
        batch_size=2,
        shard_size=2,
        limit=None,
        overwrite=False,
        embedding_model="fake",
        dimension=2,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["exported_count"] == 3
    assert manifest["exported_count"] == 3
    assert [shard["count"] for shard in manifest["shards"]] == [2, 1]
    assert (output_dir / "shard_000000.npz").exists()
    assert (output_dir / "shard_000001.npz").exists()


def test_import_dense_embeddings_to_milvus_uses_exported_doc_order(tmp_path):
    corpus_path = _write_test_corpus(tmp_path)
    output_dir = tmp_path / "embeddings"
    meta_path = tmp_path / "dense_meta.json"
    export_dense_embeddings(
        corpus_path=corpus_path,
        output_dir=output_dir,
        embedder=FakeEmbedder(),
        batch_size=2,
        shard_size=2,
        overwrite=False,
        embedding_model="fake",
        dimension=2,
    )
    store = FakeStore()

    metadata = import_dense_embeddings_to_milvus(
        corpus_path=corpus_path,
        embeddings_dir=output_dir,
        store=store,
        insert_batch_size=2,
        drop_existing=True,
        metadata_output_path=meta_path,
    )

    assert store.created is True
    assert store.drop_existing is True
    assert [document.doc_id for document, _ in store.rows] == ["d1", "d2", "d3"]
    assert metadata["inserted_count"] == 3
    assert json.loads(meta_path.read_text(encoding="utf-8"))["embedding_model"] == "fake"


def test_build_faiss_dense_index_writes_index_and_docstore(tmp_path):
    pytest.importorskip("faiss")
    corpus_path = _write_test_corpus(tmp_path)
    index_path = tmp_path / "faiss.index"
    docstore_path = tmp_path / "faiss_docs.jsonl"
    meta_path = tmp_path / "faiss_meta.json"

    metadata = build_faiss_dense_index(
        corpus_path=corpus_path,
        index_path=index_path,
        docstore_path=docstore_path,
        embedder=FakeEmbedder(),
        batch_size=2,
        overwrite=False,
        metadata_output_path=meta_path,
        embedding_model="fake",
        dimension=2,
        index_type="flat",
    )

    assert metadata["inserted_count"] == 3
    assert index_path.exists()
    assert docstore_path.exists()
    assert len(docstore_path.read_text(encoding="utf-8").splitlines()) == 3


def test_import_dense_embeddings_to_faiss_uses_exported_doc_order(tmp_path):
    pytest.importorskip("faiss")
    corpus_path = _write_test_corpus(tmp_path)
    output_dir = tmp_path / "embeddings"
    index_path = tmp_path / "faiss.index"
    docstore_path = tmp_path / "faiss_docs.jsonl"
    export_dense_embeddings(
        corpus_path=corpus_path,
        output_dir=output_dir,
        embedder=FakeEmbedder(),
        batch_size=2,
        shard_size=2,
        overwrite=False,
        embedding_model="fake",
        dimension=2,
    )

    metadata = import_dense_embeddings_to_faiss(
        corpus_path=corpus_path,
        embeddings_dir=output_dir,
        index_path=index_path,
        docstore_path=docstore_path,
        overwrite=False,
        index_type="flat",
    )

    assert metadata["inserted_count"] == 3
    records = [
        json.loads(line)
        for line in docstore_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["doc_id"] for record in records] == ["d1", "d2", "d3"]


def _write_test_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    records = [
        Document("d1", "Title 1", "Text 1", ["Text 1"]).to_dict(),
        Document("d2", "Title 2", "Text 2", ["Text 2"]).to_dict(),
        Document("d3", "Title 3", "Text 3", ["Text 3"]).to_dict(),
    ]
    corpus_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return corpus_path
