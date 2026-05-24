from src.data.schema import Document
from src.retrieval.milvus_store import document_to_milvus_row, milvus_hit_to_retrieved_doc


def test_document_to_milvus_row_serializes_json_fields():
    document = Document(
        doc_id="d1",
        title="Title",
        text="Text.",
        sentences=["Sentence one.", "Sentence two."],
        metadata={"source": "test", "source_locations": [{"question_id": "q1"}]},
    )

    row = document_to_milvus_row(document, [0.1, 0.2])

    assert row["doc_id"] == "d1"
    assert row["embedding"] == [0.1, 0.2]
    assert "Sentence one." in row["sentences_json"]
    assert '"source": "test"' in row["metadata_json"]
    assert "source_locations" not in row["metadata_json"]


def test_milvus_hit_to_retrieved_doc_converts_entity_payload():
    hit = {
        "distance": 0.87,
        "entity": {
            "doc_id": "d1",
            "title": "Title",
            "text": "Text.",
            "sentences_json": '["Sentence."]',
            "metadata_json": '{"source": "test"}',
        },
    }

    doc = milvus_hit_to_retrieved_doc(hit, rank=1)

    assert doc.doc_id == "d1"
    assert doc.score == 0.87
    assert doc.rank == 1
    assert doc.retrieval_source == "dense"
    assert doc.sentences == ["Sentence."]
    assert doc.metadata == {"source": "test"}
