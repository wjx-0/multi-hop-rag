"""Answer EM/F1 metrics."""

from __future__ import annotations

from collections import Counter

from src.utils.text import normalize_answer


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, ground_truth: str) -> dict[str, float]:
    return {
        "answer_em": exact_match_score(prediction, ground_truth),
        "answer_f1": f1_score(prediction, ground_truth),
    }
