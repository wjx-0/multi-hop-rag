from src.data.schema import Document
from src.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_relevant_doc_first():
    docs = [
        Document("d1", "A", "Arthur's Magazine was founded in 1844.", ["Arthur's Magazine was founded in 1844."]),
        Document("d2", "B", "First for Women was started in 1989.", ["First for Women was started in 1989."]),
        Document("d3", "C", "The city has a large railway station.", ["The city has a large railway station."]),
    ]
    results = BM25Retriever(docs).retrieve("When was Arthur's Magazine founded?", top_k=1)
    assert results[0].doc_id == "d1"
