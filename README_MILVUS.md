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

安装 PyMilvus：

```bash
pip install pymilvus==2.6.11
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
| id | VARCHAR | Milvus primary key |
| doc_id | VARCHAR | 项目内部 document id |
| title | VARCHAR | HotpotQA paragraph title |
| text | VARCHAR | paragraph text |
| source_question_ids | ARRAY/VARCHAR | 来源样本 id |
| embedding | FLOAT_VECTOR | dense embedding |

第一版可以把 `source_question_ids` 简化为 JSON string，减少 schema 复杂度。

---

## 5. 检索流程

Phase 2 dense retrieval：

```text
question
  -> embedding model
  -> Milvus vector search
  -> top-k RetrievedDoc
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
