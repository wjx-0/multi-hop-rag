from scripts.evaluate_prediction_metrics import average_cost, has_cost_key


def test_average_cost_reads_cost_key():
    records = [
        {"cost": {"rerank_calls": 1}},
        {"cost": {"rerank_calls": 3}},
    ]

    assert average_cost(records, "rerank_calls") == 2.0


def test_has_cost_key_detects_optional_cost_metric():
    records = [
        {"cost": {"llm_calls": 0}},
        {"cost": {"rerank_calls": 1}},
    ]

    assert has_cost_key(records, "rerank_calls") is True
    assert has_cost_key(records, "missing") is False
