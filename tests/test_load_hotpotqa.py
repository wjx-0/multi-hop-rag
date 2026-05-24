import json

from src.data.load_hotpotqa import (
    load_documents_jsonl,
    load_processed_hotpotqa_questions,
)


def test_load_documents_jsonl_reads_document_records(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": "d1",
                "title": "Title",
                "text": "Text.",
                "sentences": ["Text."],
                "metadata": {"corpus_type": "global_deduplicated"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    documents = load_documents_jsonl(path)

    assert len(documents) == 1
    assert documents[0].doc_id == "d1"
    assert documents[0].metadata["corpus_type"] == "global_deduplicated"


def test_load_processed_hotpotqa_questions_drops_context(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Question?",
                "answer": "Answer",
                "type": "comparison",
                "level": "hard",
                "supporting_facts": [["Title", 0]],
                "context": [["Title", ["Should not be loaded."]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = load_processed_hotpotqa_questions(path)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].context == []
