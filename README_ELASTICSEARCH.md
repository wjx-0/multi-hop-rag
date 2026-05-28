# Elasticsearch BM25 后端

这个后端用于把 BM25 从 Python `rank_bm25` 迁移到 Elasticsearch。它主要优化查询速度和 Python 内存占用，不保证检索指标一定提升。

## 启动服务

```bash
docker compose -f infra/elasticsearch/docker-compose.yml up -d
```

健康检查：

```bash
curl http://localhost:9200
```

停止服务：

```bash
docker compose -f infra/elasticsearch/docker-compose.yml down
```

默认配置是 single-node，关闭安全认证，JVM heap 为 `2g`。

## 构建 BM25 Index

先确认已经有 global corpus：

```text
data/processed/hotpotqa/global/corpus.jsonl
```

Smoke 构建 1000 条：

```bash
conda run -n qream-rag python scripts/build_hotpotqa_indexes.py \
  --mode elasticsearch-bm25 \
  --limit 1000 \
  --drop-existing
```

全量构建：

```bash
conda run -n qream-rag python scripts/build_hotpotqa_indexes.py \
  --mode elasticsearch-bm25 \
  --drop-existing
```

默认 index 名：

```text
hotpotqa_global_bm25
```

## Hybrid Diagnostic 使用 Elasticsearch

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_retrieval.py \
  --sample-size 1000 \
  --sample-strategy uniform \
  --bm25-backend elasticsearch \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --final-top-k 50 \
  --output outputs/predictions/hybrid_retrieval_esbm25_uniform1000_top50.jsonl
```

评估：

```bash
conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/hybrid_retrieval_esbm25_uniform1000_top50.jsonl
```

对照原来的 Python BM25：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_retrieval.py \
  --sample-size 1000 \
  --sample-strategy uniform \
  --bm25-backend rank_bm25 \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --final-top-k 50 \
  --output outputs/predictions/hybrid_retrieval_rankbm25_uniform1000_top50.jsonl
```

## 注意

- `rank_bm25` 仍然是默认后端。
- 使用 `--bm25-backend elasticsearch` 时，脚本不会加载 `bm25.pkl`。
- Elasticsearch 会新增约 2-4GB 服务内存，但能减少 Python 进程里的 BM25 index 内存。
