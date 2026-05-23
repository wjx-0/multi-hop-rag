from src.evaluation.answer_metrics import answer_metrics
from src.evaluation.evidence_metrics import (
    evidence_metrics,
    evidence_recall_at_k,
    supporting_fact_f1,
)


def test_answer_metrics_exact_match_normalized():
    metrics = answer_metrics("The Arthur's Magazine.", "Arthur's Magazine")
    assert metrics["answer_em"] == 1.0
    assert metrics["answer_f1"] == 1.0


def test_evidence_recall_at_k_uses_gold_supporting_titles():
    retrieved_docs = [
        {"title": "Arthur's Magazine", "rank": 1},
        {"title": "Some Distractor", "rank": 2},
        {"title": "First for Women", "rank": 3},
    ]
    gold_supporting_facts = [["Arthur's Magazine", 1], ["First for Women", 1]]

    assert evidence_recall_at_k(retrieved_docs, gold_supporting_facts, k=2) == 0.5
    assert evidence_recall_at_k(retrieved_docs, gold_supporting_facts, k=3) == 1.0


def test_supporting_fact_f1_requires_title_and_sentence_match():
    predicted = [
        {"source_title": "Arthur's Magazine", "source_sentence_id": 1},
        {"source_title": "First for Women", "source_sentence_id": 0},
    ]
    gold = [["Arthur's Magazine", 1], ["First for Women", 1]]

    metrics = supporting_fact_f1(predicted, gold)

    assert metrics["supporting_fact_precision"] == 0.5
    assert metrics["supporting_fact_recall"] == 0.5
    assert metrics["supporting_fact_f1"] == 0.5


def test_evidence_metrics_combines_recall_hit_and_supporting_fact_f1():
    retrieved_docs = [{"title": "Arthur's Magazine"}, {"title": "First for Women"}]
    gold = [["Arthur's Magazine", 1], ["First for Women", 1]]
    predicted = [{"source_title": "Arthur's Magazine", "source_sentence_id": 1}]

    metrics = evidence_metrics(
        retrieved_docs,
        gold,
        predicted_supporting_facts=predicted,
        ks=(1, 2),
    )

    assert metrics["evidence_recall@1"] == 0.5
    assert metrics["evidence_hit@1"] == 1.0
    assert metrics["evidence_recall@2"] == 1.0
    assert metrics["supporting_fact_f1"] == 2 / 3
