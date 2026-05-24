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
conda run -n qream-rag python scripts/build_index.py \
  --mode milvus-dense \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --limit 100 \
  --drop-existing
```

本地直接构建全量 Milvus dense index：

```bash
conda run --no-capture-output -n qream-rag python -u scripts/build_index.py \
  --mode milvus-dense \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --drop-existing \
  --embedding-batch-size 64
```

如果用 GPU 服务器加速，推荐先在服务器只导出 embedding 分片：

```bash
python -u scripts/build_index.py \
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
conda run --no-capture-output -n qream-rag python -u scripts/build_index.py \
  --mode milvus-import-embeddings \
  --corpus-input data/processed/hotpotqa/global/corpus.jsonl \
  --embedding-input-dir data/indexes/hotpotqa_global/bge_m3_embeddings \
  --drop-existing \
  --milvus-insert-batch-size 1024
```

单问题 dense retrieval smoke：

```bash
conda run -n qream-rag python scripts/run_dense_retrieval.py \
  --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
  --top-k 5
```

Hybrid retrieval：

```text
question
  -> BM25 top-k
  -> Milvus dense top-k
  -> normalize scores
  -> weighted fusion
  -> reranker
```

默认融合：

```python
final_score = 0.5 * dense_score + 0.5 * bm25_score
```

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
scripts/build_index.py
```

---

## 7. 参考来源

- Milvus Docker Compose: https://milvus.io/docs/install_standalone-docker-compose.md
- PyMilvus install: https://milvus.io/docs/install-pymilvus.md
