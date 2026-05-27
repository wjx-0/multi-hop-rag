from pathlib import Path

import pytest

from scripts.run_global_bm25_rag import ensure_global_inputs_exist, load_or_build_global_bm25


def test_ensure_global_inputs_exist_reports_build_index_hint(tmp_path):
    missing_corpus = tmp_path / "missing_corpus.jsonl"
    missing_questions = tmp_path / "missing_questions.jsonl"

    with pytest.raises(FileNotFoundError, match="build_hotpotqa_indexes.py --mode global-corpus"):
        ensure_global_inputs_exist(Path(missing_corpus), Path(missing_questions))


def test_load_or_build_global_bm25_writes_and_reuses_cache(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    cache_path = tmp_path / "bm25.pkl"
    corpus_path.write_text(
        "\n".join(
            [
                '{"doc_id":"d1","title":"Arthur","text":"Arthur was founded in 1844.","sentences":["Arthur was founded in 1844."],"metadata":{}}',
                '{"doc_id":"d2","title":"Other","text":"A distractor.","sentences":["A distractor."],"metadata":{}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    retriever = load_or_build_global_bm25(corpus_path=corpus_path, cache_path=cache_path)
    cached_retriever = load_or_build_global_bm25(corpus_path=corpus_path, cache_path=cache_path)

    assert cache_path.exists()
    assert retriever.retrieve("Arthur founded", top_k=1)[0].doc_id == "d1"
    assert cached_retriever.retrieve("Arthur founded", top_k=1)[0].doc_id == "d1"
