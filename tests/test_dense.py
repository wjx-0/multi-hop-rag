from src.data.schema import RetrievedDoc
from src.retrieval.dense import DenseRetriever, document_embedding_text


class FakeEmbedder:
    def encode_texts(self, texts):
        return [[float(len(text))] for text in texts]


class FakeStore:
    def search(self, query_embedding, *, top_k):
        assert query_embedding == [8.0]
        return [
            RetrievedDoc(
                doc_id="d1",
                title="Title",
                text="Text.",
                sentences=["Text."],
                score=0.9,
                rank=99,
                retrieval_source="old",
            )
        ][:top_k]


def test_dense_retriever_embeds_query_and_returns_dense_docs():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore())

    results = retriever.retrieve("question", top_k=1)

    assert results[0].doc_id == "d1"
    assert results[0].rank == 1
    assert results[0].retrieval_source == "dense"


def test_document_embedding_text_joins_title_and_text():
    assert document_embedding_text("Title", "Body.") == "Title\nBody."
