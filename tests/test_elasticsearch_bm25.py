import json

from src.data.schema import Document
from src.retrieval.elasticsearch_bm25 import (
    ElasticsearchBM25Retriever,
    document_to_elasticsearch_source,
    elasticsearch_bulk_actions,
    elasticsearch_hit_to_retrieved_doc,
)


def test_document_to_elasticsearch_source_serializes_nested_fields():
    document = Document(
        doc_id="d1",
        title="Title 1",
        text="Text 1",
        sentences=["Sentence 1"],
        metadata={"source": "test"},
    )

    source = document_to_elasticsearch_source(document)

    assert source["doc_id"] == "d1"
    assert source["title"] == "Title 1"
    assert json.loads(source["sentences_json"]) == ["Sentence 1"]
    assert json.loads(source["metadata_json"]) == {"source": "test"}


def test_elasticsearch_hit_to_retrieved_doc_parses_json_fields():
    hit = {
        "_id": "fallback-id",
        "_score": 12.5,
        "_source": {
            "doc_id": "d1",
            "title": "Title 1",
            "text": "Text 1",
            "sentences_json": json.dumps(["Sentence 1"]),
            "metadata_json": json.dumps({"source": "test"}),
        },
    }

    doc = elasticsearch_hit_to_retrieved_doc(hit, rank=3)

    assert doc.doc_id == "d1"
    assert doc.title == "Title 1"
    assert doc.sentences == ["Sentence 1"]
    assert doc.metadata == {"source": "test"}
    assert doc.score == 12.5
    assert doc.rank == 3
    assert doc.retrieval_source == "elasticsearch_bm25"


def test_elasticsearch_retriever_returns_empty_list_for_no_hits():
    retriever = ElasticsearchBM25Retriever(
        index_name="test-index",
        client=FakeElasticsearchClient([]),
    )

    assert retriever.retrieve("question", top_k=5) == []


def test_elasticsearch_retriever_maps_hits_to_retrieved_docs():
    retriever = ElasticsearchBM25Retriever(
        index_name="test-index",
        client=FakeElasticsearchClient(
            [
                {
                    "_score": 1.5,
                    "_source": {
                        "doc_id": "d1",
                        "title": "Title 1",
                        "text": "Text 1",
                        "sentences_json": "[]",
                        "metadata_json": "{}",
                    },
                }
            ]
        ),
    )

    docs = retriever.retrieve("question", top_k=1)

    assert len(docs) == 1
    assert docs[0].doc_id == "d1"
    assert docs[0].rank == 1
    assert docs[0].retrieval_source == "elasticsearch_bm25"


def test_elasticsearch_bulk_actions_use_doc_id_as_es_id(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    document = Document("d1", "Title 1", "Text 1", ["Text 1"])
    corpus_path.write_text(json.dumps(document.to_dict()) + "\n", encoding="utf-8")

    actions = list(elasticsearch_bulk_actions(corpus_path, index_name="idx"))

    assert actions[0]["_op_type"] == "index"
    assert actions[0]["_index"] == "idx"
    assert actions[0]["_id"] == "d1"
    assert actions[0]["_source"]["doc_id"] == "d1"


class FakeElasticsearchClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": self.hits}}
