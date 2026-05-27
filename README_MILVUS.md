# Milvus Setup

本项目使用 Milvus 作为 Phase 2 之后的 dense vector store。

推荐路线：

```text
Phase 1:
HotpotQA per-sample corpus + BM25
不需要 Milvus

Phase 2:
HotpotQA deduplicated global corpus + Milvus
用于 dense retrieval 和 hybrid retrieval
```

---

## 1. 为什么用 Milvus

Milvus 更接近真实 RAG 系统中的向量数据库，而不是只把向量索引保存在本地文件里。

适合本项目的原因：

- 支持大规模向量检索。
- 支持 collection、schema、metadata scalar fields。
- 能保存 document id、title、sentence 信息等元数据。
- 方便后续把检索服务独立出来。
- 比本地 FAISS 更适合展示工程化 RAG 项目。

BM25 仍然保留，用于 hybrid retrieval：

```text
BM25 sparse retrieval + Milvus dense retrieval -> score fusion -> reranker
```

---

## 2. 本地启动 Milvus

官方推荐本地开发使用 Docker Compose 启动 Milvus Standalone。

建议把 Milvus compose 文件放在：

```text
infra/milvus/docker-compose.yml
```

下载官方 Docker Compose 文件：

```bash
mkdir -p infra/milvus
cd infra/milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

启动：

```bash
docker compose up -d
```

检查：

```bash
docker compose ps
```

默认服务：

```text
Milvus gRPC / HTTP endpoint: localhost:19530
Milvus Web UI: http://127.0.0.1:9091/webui/
```

停止：

```bash
docker compose down
```

---

## 3. Python 依赖

安装 PyMilvus 和 embedding 依赖：

```bash
pip install pymilvus==2.6.11 sentence-transformers==5.1.2 numpy==2.2.6
```

验证：

```bash
python -c "from pymilvus import MilvusClient; print('pymilvus ok')"
```

---

## 4. 本项目中的 Collection 设计

第一版 collection：

```text
hotpotqa_global_chunks
```

建议字段：

| Field | Type | Purpose |
|---|---|---|
| doc_id | VARCHAR | Milvus primary key / 项目内部 document id |
| title | VARCHAR | HotpotQA paragraph title |
| text | VARCHAR | paragraph text |
| sentences_json | VARCHAR | paragraph sentences JSON |
| metadata_json | VARCHAR | 精简 metadata JSON |
| embedding | FLOAT_VECTOR | dense embedding |

完整 `source_locations` 仍以 `corpus.jsonl` 为准，不直接写入 Milvus，避免 metadata 过大。

---

## 5. 检索流程

Phase 2 dense retrieval：

```text
question
  -> embedding model
  -> Milvus vector search
  -> top-k RetrievedDoc
```

构建 dense index smoke：

```bash
conda run -n qream-rag python scripts/build_hotpotqa_indexes.py \
  --mode milvus-dense \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --limit 100 \
  --drop-existing
```

本地直接构建全量 Milvus dense index：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/build_hotpotqa_indexes.py \
  --mode milvus-dense \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --drop-existing \
  --embedding-batch-size 64
```

如果用 GPU 服务器加速，推荐先在服务器只导出 embedding 分片：

```bash
python -u scripts/build_hotpotqa_indexes.py \
  --mode dense-embeddings \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --embedding-output-dir data/indexes/hotpotqa_global/bge_m3_embeddings \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --embedding-shard-size 8192 \
  --drop-existing
```

把 `data/indexes/hotpotqa_global/bge_m3_embeddings/` 传回本地后，导入本地 Milvus：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/build_hotpotqa_indexes.py \
  --mode milvus-import-embeddings \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --embedding-input-dir data/indexes/hotpotqa_global/bge_m3_embeddings \
  --drop-existing \
  --milvus-insert-batch-size 1024
```

单问题 dense retrieval smoke：

```bash
conda run -n qream-rag python scripts/query_milvus_dense_retrieval.py \
  --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
  --top-k 5
```

批量 dense retrieval diagnostic，不调用 LLM：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_dense_retrieval.py \
  --limit 100 \
  --top-k 50 \
  --query-batch-size 32 \
  --output outputs/predictions/dense_retrieval_global_100_top50.jsonl
```

汇总 evidence metrics：

```bash
conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/dense_retrieval_global_100_top50.jsonl
```

BM25 + Dense hybrid retrieval diagnostic：

Profile A，BM25 top20 + Dense top50 -> Hybrid final top50：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_retrieval.py \
  --limit 100 \
  --bm25-top-k 20 \
  --dense-top-k 50 \
  --final-top-k 50 \
  --output outputs/predictions/hybrid_retrieval_global_100_bm25top20_dense50_final50.jsonl
```

Profile B，BM25 top50 + Dense top50 -> Hybrid final top50：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_retrieval.py \
  --limit 100 \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --final-top-k 50 \
  --output outputs/predictions/hybrid_retrieval_global_100_bm25top50_dense50_final50.jsonl
```

如果不想只测 dev 前 N 条，可以用均匀位置抽样。例如从 7,405 条 dev questions 中均匀抽 500 条：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_retrieval.py \
  --sample-size 500 \
  --sample-strategy uniform \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --final-top-k 50 \
  --output outputs/predictions/hybrid_retrieval_global_uniform500_top50.jsonl
```

汇总 hybrid evidence metrics：

```bash
conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/hybrid_retrieval_global_100_bm25top20_dense50_final50.jsonl

conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/hybrid_retrieval_global_100_bm25top50_dense50_final50.jsonl
```

Hybrid 融合默认启用轻量 title boost：问题和文档标题的有效 token 命中时，每个 token 加 `0.0005`，单文档最多加 `0.0015`。这个分数只用于小幅调整 RRF 排序；如果要回到纯 RRF，可以加：

```bash
--title-boost-weight 0
```

Hybrid + DashScope reranker diagnostic，不调用回答 LLM：

需要 `.env` 中配置 `DASHSCOPE_API_KEY`。可选覆盖：

```text
DASHSCOPE_RERANK_MODEL=qwen3-rerank
DASHSCOPE_RERANK_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
```

```bash
conda run --no-capture-output -n qream-rag python -u scripts/diagnose_hybrid_rerank.py \
  --limit 20 \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --hybrid-top-k 50 \
  --rerank-top-n 50 \
  --output outputs/predictions/rerank_hybrid_global_20_top50.jsonl
```

汇总 rerank evidence metrics：

```bash
conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/rerank_hybrid_global_20_top50.jsonl
```

重点看 `evidence_recall@10` / `evidence_recall@20` 是否比 Hybrid top50 更高，同时确认 `rerank_error` 接近 0、`avg_rerank_calls` 符合预期。

Hybrid + Reranker + DashScope answer generation baseline：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/run_hybrid_rerank_rag.py \
  --limit 20 \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --hybrid-top-k 50 \
  --rerank-top-n 50 \
  --answer-top-k 10 \
  --output outputs/predictions/rerank_rag_global_20_top10.jsonl
```

对比更长上下文：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/run_hybrid_rerank_rag.py \
  --limit 20 \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --hybrid-top-k 50 \
  --rerank-top-n 50 \
  --answer-top-k 20 \
  --output outputs/predictions/rerank_rag_global_20_top20.jsonl
```

汇总 answer + evidence metrics：

```bash
conda run -n qream-rag python scripts/evaluate_prediction_metrics.py \
  --predictions outputs/predictions/rerank_rag_global_20_top10.jsonl
```

生成 bad case 分析报告：

```bash
conda run -n qream-rag python scripts/analyze_prediction_bad_cases.py \
  --predictions outputs/predictions/rerank_rag_global_20_top10.jsonl \
  --output-dir outputs/reports/rerank_rag_global_20_top10
```

Hybrid + reranker RAG：

```text
question
  -> BM25 top-k
  -> Milvus dense top-k
  -> RRF fusion
  -> DashScope qwen3-rerank
  -> top-k evidence
  -> DashScope answer generation
```

当前第一版 hybrid 使用 RRF，避免直接混合 BM25 分数和 dense cosine 分数。

---

## 6. 实现阶段

Milvus 不进入 Phase 1。

推荐安排：

```text
Phase 1:
BM25 + per-sample corpus

Phase 2:
build global corpus
-> generate embeddings
-> create Milvus collection
-> insert documents
-> dense search
-> hybrid retrieval
```

需要新增的实现文件：

```text
src/retrieval/milvus_store.py
scripts/build_hotpotqa_indexes.py
```

---

## 7. 参考来源

- Milvus Docker Compose: https://milvus.io/docs/install_standalone-docker-compose.md
- PyMilvus install: https://milvus.io/docs/install-pymilvus.md
