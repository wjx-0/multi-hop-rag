from scripts.diagnose_dense_retrieval import parse_args


def test_run_dense_diagnostic_parse_args():
    args = parse_args(
        [
            "--limit",
            "100",
            "--top-k",
            "50",
            "--query-batch-size",
            "16",
            "--output",
            "outputs/predictions/dense.jsonl",
        ]
    )

    assert args.limit == 100
    assert args.top_k == 50
    assert args.query_batch_size == 16
    assert args.output == "outputs/predictions/dense.jsonl"
