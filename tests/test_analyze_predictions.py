import json

from scripts.analyze_prediction_bad_cases import (
    analyze_record,
    build_summary_markdown,
    is_bad_case,
    load_question_metadata,
    main,
)


def test_analyze_record_tags_llm_and_rerank_errors():
    analysis = analyze_record(
        _record(
            metrics={"answer_em": 1.0, "answer_f1": 1.0, "llm_error": 1.0, "rerank_error": 1.0},
            agent_outputs={"llm_error": "timeout"},
        )
    )

    assert "llm_error" in analysis["error_tags"]
    assert "rerank_error" in analysis["error_tags"]
    assert analysis["llm_error"] == "timeout"
    assert analysis["rerank_error"] is True
    assert is_bad_case(analysis) is True


def test_analyze_record_tags_answer_wrong_but_evidence_present():
    analysis = analyze_record(
        _record(
            gold_answer="Chief of Protocol",
            pred_answer="Ambassador",
            supporting_facts=[["Shirley Temple", 0]],
            retrieved_titles=["Shirley Temple"],
            metrics={"answer_em": 0.0, "answer_f1": 0.0},
        )
    )

    assert "answer_wrong_but_evidence_present" in analysis["error_tags"]
    assert "evidence_missing" not in analysis["error_tags"]
    assert analysis["evidence_recall_at_answer_k"] == 1.0


def test_analyze_record_tags_answer_wrong_and_evidence_missing():
    analysis = analyze_record(
        _record(
            gold_answer="Chief of Protocol",
            pred_answer="Ambassador",
            supporting_facts=[["Shirley Temple", 0], ["Kiss and Tell", 0]],
            retrieved_titles=["Shirley Temple"],
            metrics={"answer_em": 0.0, "answer_f1": 0.0},
        )
    )

    assert "evidence_missing" in analysis["error_tags"]
    assert "answer_wrong_and_evidence_missing" in analysis["error_tags"]
    assert analysis["missing_gold_titles"] == ["kiss and tell"]
    assert analysis["evidence_recall_at_answer_k"] == 0.5


def test_analyze_record_tags_yes_no_wrong_partial_and_low_citation():
    analysis = analyze_record(
        _record(
            gold_answer="yes",
            pred_answer="yes, both are American",
            supporting_facts=[["Scott Derrickson", 0]],
            retrieved_titles=["Scott Derrickson"],
            metrics={"answer_em": 0.0, "answer_f1": 0.5, "supporting_fact_f1": 0.0},
        )
    )

    assert "yes_no_wrong" in analysis["error_tags"]
    assert "partial_answer" in analysis["error_tags"]
    assert "citation_low" in analysis["error_tags"]


def test_build_summary_markdown_includes_groups_and_bad_cases():
    records = [
        _record(
            type_="bridge",
            gold_answer="yes",
            pred_answer="no",
            supporting_facts=[["Title", 0]],
            retrieved_titles=["Title"],
            metrics={"answer_em": 0.0, "answer_f1": 0.0},
        )
    ]
    analyses = [analyze_record(record) for record in records]

    summary = build_summary_markdown(
        records=records,
        analyses=analyses,
        predictions_path="predictions.jsonl",
        bad_case_path="bad_cases.jsonl",
    )

    assert "Prediction Bad Case Analysis" in summary
    assert "## By Type" in summary
    assert "yes_no_wrong" in summary
    assert "bridge" in summary


def test_analyze_record_uses_question_metadata_when_type_missing(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": "q1", "type": "comparison", "level": "hard"}) + "\n",
        encoding="utf-8",
    )

    metadata = load_question_metadata(questions)
    record = _record(type_=None, level=None)
    analysis = analyze_record(record, question_metadata=metadata)

    assert analysis["type"] == "comparison"
    assert analysis["level"] == "hard"


def test_main_writes_summary_and_bad_cases(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "report"
    prediction_record = _record(
        gold_answer="Chief of Protocol",
        pred_answer="Ambassador",
        supporting_facts=[["Shirley Temple", 0]],
        retrieved_titles=["Shirley Temple"],
        metrics={"answer_em": 0.0, "answer_f1": 0.0},
    )
    predictions.write_text(json.dumps(prediction_record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_prediction_bad_cases.py",
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--questions",
            "",
        ],
    )

    main()

    summary = output_dir / "summary.md"
    bad_cases = output_dir / "bad_cases.jsonl"
    assert summary.exists()
    assert bad_cases.exists()
    assert "answer_wrong_but_evidence_present" in summary.read_text(encoding="utf-8")
    bad_case_rows = [json.loads(line) for line in bad_cases.read_text(encoding="utf-8").splitlines()]
    assert bad_case_rows[0]["id"] == "q1"


def _record(
    *,
    gold_answer="answer",
    pred_answer="answer",
    supporting_facts=None,
    retrieved_titles=None,
    metrics=None,
    agent_outputs=None,
    type_="bridge",
    level="hard",
):
    supporting_facts = supporting_facts if supporting_facts is not None else [["Title", 0]]
    retrieved_titles = retrieved_titles if retrieved_titles is not None else ["Title"]
    return {
        "id": "q1",
        "question": "Question?",
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        **({} if type_ is None else {"type": type_}),
        **({} if level is None else {"level": level}),
        "gold_supporting_facts": supporting_facts,
        "retrieved_docs": [
            {
                "title": title,
                "doc_id": f"d{i}",
                "text": "Evidence.",
                "sentences": ["Evidence."],
                "metadata": {},
                "score": 1.0,
                "rank": i,
                "retrieval_source": "reranker",
            }
            for i, title in enumerate(retrieved_titles, start=1)
        ],
        "metrics": metrics if metrics is not None else {"answer_em": 1.0, "answer_f1": 1.0},
        "agent_outputs": agent_outputs or {},
        "cost": {},
    }
