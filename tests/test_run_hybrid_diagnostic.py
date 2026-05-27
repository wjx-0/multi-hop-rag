from scripts.diagnose_hybrid_retrieval import parse_args


def test_run_hybrid_diagnostic_parse_args():
    args = parse_args(
        [
            "--limit",
            "100",
            "--bm25-top-k",
            "40",
            "--dense-top-k",
            "60",
            "--final-top-k",
            "50",
            "--rrf-k",
            "70",
            "--query-batch-size",
            "16",
            "--output",
            "outputs/predictions/hybrid.jsonl",
        ]
    )

    assert args.limit == 100
    assert args.bm25_top_k == 40
    assert args.dense_top_k == 60
    assert args.final_top_k == 50
    assert args.rrf_k == 70
    assert args.query_batch_size == 16
    assert args.output == "outputs/predictions/hybrid.jsonl"
