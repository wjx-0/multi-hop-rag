from src.data.schema import Document
from src.retrieval.bm25 import BM25Retriever, load_bm25_cache, save_bm25_cache


def test_bm25_retrieves_relevant_doc_first():
    docs = [
        Document("d1", "A", "Arthur's Magazine was founded in 1844.", ["Arthur's Magazine was founded in 1844."]),
        Document("d2", "B", "First for Women was started in 1989.", ["First for Women was started in 1989."]),
        Document("d3", "C", "The city has a large railway station.", ["The city has a large railway station."]),
    ]
    results = BM25Retriever(docs).retrieve("When was Arthur's Magazine founded?", top_k=1)
    assert results[0].doc_id == "d1"


def test_bm25_top_k_partial_selection_matches_full_sort_order():
    docs = [
        Document("d1", "Alpha", "alpha alpha alpha alpha", ["alpha alpha alpha alpha"]),
        Document("d2", "Beta", "alpha alpha alpha", ["alpha alpha alpha"]),
        Document("d3", "Gamma", "alpha alpha", ["alpha alpha"]),
        Document("d4", "Delta", "alpha", ["alpha"]),
        Document("d5", "Epsilon", "unrelated", ["unrelated"]),
    ]
    retriever = BM25Retriever(docs)
    full_order = [doc.doc_id for doc in retriever.retrieve("alpha", top_k=50)]

    assert [doc.doc_id for doc in retriever.retrieve("alpha", top_k=1)] == full_order[:1]
    assert [doc.doc_id for doc in retriever.retrieve("alpha", top_k=2)] == full_order[:2]
    assert [doc.doc_id for doc in retriever.retrieve("alpha", top_k=50)] == full_order


def test_bm25_retrieve_handles_non_positive_and_oversized_top_k():
    docs = [
        Document("d1", "A", "alpha", ["alpha"]),
        Document("d2", "B", "beta", ["beta"]),
    ]
    retriever = BM25Retriever(docs)

    assert retriever.retrieve("alpha", top_k=0) == []
    assert retriever.retrieve("alpha", top_k=-1) == []
    assert len(retriever.retrieve("alpha", top_k=50)) == 2


def test_bm25_cache_round_trip_and_invalidates_when_corpus_changes(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text('{"doc_id": "d1"}\n', encoding="utf-8")
    cache_path = tmp_path / "bm25.pkl"
    retriever = BM25Retriever(
        [
            Document(
                "d1",
                "Arthur's Magazine",
                "Arthur's Magazine was founded in 1844.",
                ["Arthur's Magazine was founded in 1844."],
            )
        ]
    )

    save_bm25_cache(retriever, cache_path, corpus_path=corpus_path)
    loaded = load_bm25_cache(cache_path, corpus_path=corpus_path)

    assert loaded is not None
    assert loaded.retrieve("founded Arthur", top_k=1)[0].doc_id == "d1"

    corpus_path.write_text('{"doc_id": "d1"}\n{"doc_id": "d2"}\n', encoding="utf-8")

    assert load_bm25_cache(cache_path, corpus_path=corpus_path) is None
