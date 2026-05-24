from scripts.build_index import parse_args


def test_build_index_parse_args_keeps_global_corpus_mode():
    args = parse_args(["--mode", "global-corpus", "--limit-train", "10"])

    assert args.mode == "global-corpus"
    assert args.limit_train == 10


def test_build_index_parse_args_supports_milvus_dense_mode():
    args = parse_args(
        [
            "--mode",
            "milvus-dense",
            "--limit",
            "100",
            "--drop-existing",
            "--embedding-model",
            "BAAI/bge-m3",
        ]
    )

    assert args.mode == "milvus-dense"
    assert args.limit == 100
    assert args.drop_existing is True
    assert args.embedding_model == "BAAI/bge-m3"


def test_build_index_parse_args_supports_dense_embedding_export_mode():
    args = parse_args(
        [
            "--mode",
            "dense-embeddings",
            "--embedding-output-dir",
            "data/indexes/hotpotqa_global/bge_m3_embeddings",
            "--embedding-shard-size",
            "4096",
        ]
    )

    assert args.mode == "dense-embeddings"
    assert args.embedding_output_dir == "data/indexes/hotpotqa_global/bge_m3_embeddings"
    assert args.embedding_shard_size == 4096


def test_build_index_parse_args_supports_milvus_embedding_import_mode():
    args = parse_args(
        [
            "--mode",
            "milvus-import-embeddings",
            "--embedding-input-dir",
            "data/indexes/hotpotqa_global/bge_m3_embeddings",
            "--milvus-insert-batch-size",
            "512",
        ]
    )

    assert args.mode == "milvus-import-embeddings"
    assert args.embedding_input_dir == "data/indexes/hotpotqa_global/bge_m3_embeddings"
    assert args.milvus_insert_batch_size == 512
