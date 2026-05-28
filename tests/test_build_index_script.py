from scripts.build_hotpotqa_indexes import parse_args


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


def test_build_index_parse_args_supports_faiss_dense_mode():
    args = parse_args(
        [
            "--mode",
            "faiss-dense",
            "--faiss-index",
            "data/indexes/hotpotqa_global/faiss.index",
            "--faiss-docstore",
            "data/indexes/hotpotqa_global/faiss_docs.jsonl",
            "--faiss-index-type",
            "flat",
            "--faiss-ef-search",
            "96",
        ]
    )

    assert args.mode == "faiss-dense"
    assert args.faiss_index == "data/indexes/hotpotqa_global/faiss.index"
    assert args.faiss_docstore == "data/indexes/hotpotqa_global/faiss_docs.jsonl"
    assert args.faiss_index_type == "flat"
    assert args.faiss_ef_search == 96


def test_build_index_parse_args_supports_faiss_embedding_import_mode():
    args = parse_args(
        [
            "--mode",
            "faiss-import-embeddings",
            "--embedding-input-dir",
            "data/indexes/hotpotqa_global/bge_m3_embeddings",
        ]
    )

    assert args.mode == "faiss-import-embeddings"
    assert args.embedding_input_dir == "data/indexes/hotpotqa_global/bge_m3_embeddings"


def test_build_index_parse_args_supports_elasticsearch_bm25_mode():
    args = parse_args(
        [
            "--mode",
            "elasticsearch-bm25",
            "--limit",
            "1000",
            "--drop-existing",
            "--elasticsearch-url",
            "http://localhost:9200",
            "--elasticsearch-index",
            "hotpotqa_test",
        ]
    )

    assert args.mode == "elasticsearch-bm25"
    assert args.limit == 1000
    assert args.drop_existing is True
    assert args.elasticsearch_url == "http://localhost:9200"
    assert args.elasticsearch_index == "hotpotqa_test"
