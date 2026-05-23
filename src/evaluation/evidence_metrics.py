"""Evidence retrieval and supporting-fact metrics."""

from __future__ import annotations

from typing import Any

from src.data.schema import RetrievedDoc


def _normalize_title(title: Any) -> str:
    return str(title).strip().lower()


def _doc_title(doc: dict[str, Any] | RetrievedDoc) -> str:
    if isinstance(doc, dict):
        return _normalize_title(doc.get("title", doc.get("source_title", "")))
    return _normalize_title(doc.title)


def _fact_tuple(fact: Any) -> tuple[str, int] | None:
    if isinstance(fact, dict):
        title = fact.get("source_title", fact.get("title"))
        sent_id = fact.get("source_sentence_id", fact.get("sent_id"))
    elif isinstance(fact, (list, tuple)) and len(fact) == 2:
        title, sent_id = fact
    else:
        return None

    if title is None or not isinstance(sent_id, int):
        return None
    return (_normalize_title(title), sent_id)


def gold_evidence_titles(gold_supporting_facts: list[list[Any]]) -> set[str]:
    return {
        title
        for fact in gold_supporting_facts
        if (parsed := _fact_tuple(fact)) is not None
        for title, _ in [parsed]
    }


def evidence_recall_at_k(
    retrieved_docs: list[dict[str, Any] | RetrievedDoc],
    gold_supporting_facts: list[list[Any]],
    *,
    k: int,
) -> float:
    """Paragraph/title-level evidence recall@k for HotpotQA supporting facts."""

    gold_titles = gold_evidence_titles(gold_supporting_facts)
    if not gold_titles:
        return 0.0

    retrieved_titles = {_doc_title(doc) for doc in retrieved_docs[:k]}
    return len(gold_titles & retrieved_titles) / len(gold_titles)


def evidence_hit_at_k(
    retrieved_docs: list[dict[str, Any] | RetrievedDoc],
    gold_supporting_facts: list[list[Any]],
    *,
    k: int,
) -> float:
    return float(evidence_recall_at_k(retrieved_docs, gold_supporting_facts, k=k) > 0.0)


def supporting_fact_f1(
    predicted_supporting_facts: list[Any],
    gold_supporting_facts: list[list[Any]],
) -> dict[str, float]:
    """Exact title + sentence-id supporting-fact precision/recall/F1."""

    pred = {parsed for fact in predicted_supporting_facts if (parsed := _fact_tuple(fact)) is not None}
    gold = {parsed for fact in gold_supporting_facts if (parsed := _fact_tuple(fact)) is not None}

    if not pred and not gold:
        return {
            "supporting_fact_precision": 1.0,
            "supporting_fact_recall": 1.0,
            "supporting_fact_f1": 1.0,
        }
    if not pred or not gold:
        return {
            "supporting_fact_precision": 0.0,
            "supporting_fact_recall": 0.0,
            "supporting_fact_f1": 0.0,
        }

    true_positive = len(pred & gold)
    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "supporting_fact_precision": precision,
        "supporting_fact_recall": recall,
        "supporting_fact_f1": f1,
    }


def evidence_metrics(
    retrieved_docs: list[dict[str, Any] | RetrievedDoc],
    gold_supporting_facts: list[list[Any]],
    *,
    predicted_supporting_facts: list[Any] | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"evidence_recall@{k}"] = evidence_recall_at_k(
            retrieved_docs,
            gold_supporting_facts,
            k=k,
        )
        metrics[f"evidence_hit@{k}"] = evidence_hit_at_k(
            retrieved_docs,
            gold_supporting_facts,
            k=k,
        )

    if predicted_supporting_facts is not None:
        metrics.update(supporting_fact_f1(predicted_supporting_facts, gold_supporting_facts))

    return metrics
