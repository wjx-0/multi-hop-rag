from scripts.diagnose_hybrid_retrieval import parse_args, uniform_sample_indexes


def test_run_hybrid_diagnostic_parse_args():
    args = parse_args(
        [
            "--limit",
            "100",
            "--sample-size",
            "500",
            "--sample-strategy",
            "uniform",
            "--bm25-top-k",
            "40",
            "--dense-top-k",
            "60",
            "--final-top-k",
            "50",
            "--rrf-k",
            "70",
            "--title-boost-weight",
            "0.0007",
            "--query-batch-size",
            "16",
            "--output",
            "outputs/predictions/hybrid.jsonl",
        ]
    )

    assert args.limit == 100
    assert args.sample_size == 500
    assert args.sample_strategy == "uniform"
    assert args.bm25_top_k == 40
    assert args.dense_top_k == 60
    assert args.final_top_k == 50
    assert args.rrf_k == 70
    assert args.title_boost_weight == 0.0007
    assert args.query_batch_size == 16
    assert args.output == "outputs/predictions/hybrid.jsonl"


def test_uniform_sample_indexes_cover_the_full_range_evenly():
    indexes = uniform_sample_indexes(total_count=10, sample_size=4)

    assert indexes == [0, 3, 6, 9]


def test_uniform_sample_indexes_returns_all_when_sample_is_larger():
    indexes = uniform_sample_indexes(total_count=3, sample_size=10)

    assert indexes == [0, 1, 2]
