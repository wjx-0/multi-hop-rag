from src.data.build_corpus import build_global_deduplicated_corpus
from src.data.schema import HotpotQASample


def _sample(
    sample_id: str,
    context,
    *,
    answer: str = "answer",
    supporting_facts=None,
) -> HotpotQASample:
    return HotpotQASample(
        id=sample_id,
        question=f"Question {sample_id}?",
        answer=answer,
        type="bridge",
        level="medium",
        supporting_facts=supporting_facts or [["Shared Title", 0]],
        context=context,
    )


def test_global_corpus_deduplicates_same_title_and_text():
    train_sample = _sample(
        "train-1",
        [["Shared Title", ["Same sentence.", "Another fact."]]],
    )
    dev_sample = _sample(
        "dev-1",
        [[" shared title ", [" Same sentence. ", "Another fact."]]],
    )

    documents, dev_questions, title_to_doc_ids = build_global_deduplicated_corpus(
        train_samples=[train_sample],
        dev_samples=[dev_sample],
    )

    assert len(documents) == 1
    assert documents[0].metadata["source_question_ids"] == ["dev-1", "train-1"]
    assert documents[0].metadata["source_locations"] == [
        {"split": "train", "question_id": "train-1", "paragraph_index": 0},
        {"split": "dev", "question_id": "dev-1", "paragraph_index": 0},
    ]
    assert title_to_doc_ids == {"shared title": [documents[0].doc_id]}
    assert dev_questions[0]["id"] == "dev-1"


def test_global_corpus_keeps_same_title_with_different_text():
    train_sample = _sample(
        "train-1",
        [
            ["Shared Title", ["First version."]],
            ["Shared Title", ["Second version."]],
        ],
    )

    documents, _, title_to_doc_ids = build_global_deduplicated_corpus(
        train_samples=[train_sample],
        dev_samples=[],
    )

    assert len(documents) == 2
    assert {doc.text for doc in documents} == {"First version.", "Second version."}
    assert title_to_doc_ids["shared title"] == sorted(doc.doc_id for doc in documents)


def test_dev_question_records_do_not_include_context():
    dev_sample = _sample(
        "dev-1",
        [["Title", ["Sentence."]]],
        answer="gold",
        supporting_facts=[["Title", 0]],
    )

    _, dev_questions, _ = build_global_deduplicated_corpus(
        train_samples=[],
        dev_samples=[dev_sample],
    )

    assert dev_questions == [
        {
            "id": "dev-1",
            "question": "Question dev-1?",
            "answer": "gold",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Title", 0]],
        }
    ]
    assert "context" not in dev_questions[0]
