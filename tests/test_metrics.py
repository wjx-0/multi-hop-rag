from src.evaluation.answer_metrics import answer_metrics


def test_answer_metrics_exact_match_normalized():
    metrics = answer_metrics("The Arthur's Magazine.", "Arthur's Magazine")
    assert metrics["answer_em"] == 1.0
    assert metrics["answer_f1"] == 1.0
