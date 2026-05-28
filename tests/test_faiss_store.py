import json

import pytest

from src.data.schema import Document
from src.retrieval.faiss_store import FAISSHotpotStore, make_faiss_index, write_faiss_docstore_record


def test_write_faiss_docstore_record_removes_source_locations(tmp_path):
    docstore_path = tmp_path / "docs.jsonl"
    document = Document(
        doc_id="d1",
        title="Title",
        text="Text.",
        sentences=["Text."],
        metadata={"source": "test", "source_locations": [{"question_id": "q1"}]},
    )

    with docstore_path.open("w", encoding="utf-8") as f:
        write_faiss_docstore_record(f, document)

    record = json.loads(docstore_path.read_text(encoding="utf-8"))
    assert record["doc_id"] == "d1"
    assert record["metadata"] == {"source": "test"}


def test_faiss_hotpot_store_searches_by_vector_similarity(tmp_path):
    faiss = pytest.importorskip("faiss")
    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "docs.jsonl"
    index = make_faiss_index(dimension=2, metric_type="COSINE", index_type="flat")

    import numpy as np

    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))

    docs = [
        Document("d1", "Title 1", "Text 1", ["Text 1"], {"source_locations": [1]}),
        Document("d2", "Title 2", "Text 2", ["Text 2"]),
    ]
    with docstore_path.open("w", encoding="utf-8") as f:
        for document in docs:
            write_faiss_docstore_record(f, document)

    store = FAISSHotpotStore(index_path=index_path, docstore_path=docstore_path)
    results = store.search([1.0, 0.0], top_k=2)

    assert [doc.doc_id for doc in results] == ["d1", "d2"]
    assert results[0].retrieval_source == "dense"
    assert results[0].metadata["dense_backend"] == "faiss"
    assert "source_locations" not in results[0].metadata
