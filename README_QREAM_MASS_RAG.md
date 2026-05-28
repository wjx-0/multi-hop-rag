# Router-QREAM-MASS-RAG

面向多跳问答的路由式可信 RAG 系统。

本项目的核心想法是：**不是所有问题都应该走重型多 Agent 流程**。简单事实型问题直接使用普通 RAG，复杂多跳问题才触发 QREAM-style 文档重写、多证据抽取、多 Agent 推理和答案校验。

详细实施计划已从 README 中拆出，见：

- [PROJECT_PLAN.md](PROJECT_PLAN.md)
- [README_DATASETS.md](README_DATASETS.md)
- [README_MILVUS.md](README_MILVUS.md)
- [README_ELASTICSEARCH.md](README_ELASTICSEARCH.md)

---

## Core Idea

```text
User Question
  -> Question Router
      -> Simple Question:
           Standard RAG
      -> Complex Question:
           Hybrid Retrieval
           + Reranker
           + QREAM-style Document Rewriting
           + Extractor / Reasoner / Synthesizer / Verifier Agents
```

如果简单路径回答低置信、无引用或校验失败，系统会自动升级到复杂路径。

```text
Simple Route:
question -> retrieval -> reranker -> answer generation -> lightweight verifier

Complex Route:
question -> hybrid retrieval -> reranker -> QREAM rewriter
         -> evidence extractor -> reasoner -> synthesizer -> verifier
```

---

## Project Goals

- 用 HotpotQA distractor 构建可评测的 Multi-hop RAG 实验系统。
- 用 Question Router 区分简单问题和复杂问题。
- 简单问题走低成本 Standard RAG。
- 复杂问题走 QREAM + Multi-Agent RAG。
- 输出答案、引用、证据、路由信息、成本和评测指标。
- 通过消融实验说明路由机制在保持效果的同时降低 token cost 和 latency。

---

## Current Status

当前仓库已经完成 Phase 1 的最小闭环：

```text
HotpotQA raw JSON
-> HotpotQASample
-> paragraph-level Document chunks
-> per-sample BM25 retrieval
-> MockLLM answer placeholder
-> Answer EM / F1
-> prediction JSONL
```

并已开始 Phase 2 的全局语料准备：

```text
HotpotQA train + dev context
-> global deduplicated paragraph corpus
-> dev questions JSONL
-> normalized title -> doc_ids index
-> global BM25 Standard RAG baseline
-> BM25 cache at data/indexes/hotpotqa_global/bm25.pkl
```

已经跑通：

```text
python3 scripts/prepare_hotpotqa_per_sample.py --limit 20 --output data/processed/hotpotqa/per_sample/dev_samples_smoke.jsonl
python3 scripts/build_hotpotqa_indexes.py --mode global-corpus --limit-train 10 --limit-dev 5
python3 scripts/run_global_bm25_rag.py --limit 20 --top-k 5 --output outputs/predictions/standard_rag_global_bm25_api_smoke.jsonl
python3 scripts/evaluate_prediction_metrics.py --predictions outputs/predictions/standard_rag_global_bm25_api_smoke.jsonl
python3 -m pytest
```

如果修改或重建了 global corpus，可以强制刷新 BM25 cache：

```bash
python3 scripts/run_global_bm25_rag.py --rebuild-bm25-cache --limit 1 --llm mock
```

DashScope API 默认会限速并重试临时网络错误。网络不稳时可以显式放慢请求：

```bash
python3 scripts/run_global_bm25_rag.py \
  --limit 20 \
  --api-min-request-interval-seconds 2 \
  --api-max-retries 5 \
  --api-retry-backoff-seconds 2
```

Hybrid rerank 脚本默认使用本地 `Qwen/Qwen3-Reranker-0.6B`，从 Hugging Face cache 离线加载，不再要求 rerank 阶段调用 DashScope API：

```bash
conda run -n qream-rag python scripts/diagnose_hybrid_rerank.py \
  --limit 20 \
  --reranker-backend local \
  --local-reranker-batch-size 4
```

如需切回 DashScope `qwen3-rerank` API，可以显式指定：

```bash
conda run -n qream-rag python scripts/diagnose_hybrid_rerank.py \
  --limit 20 \
  --reranker-backend dashscope
```

当前还没有实现 Router、QREAM、多 Agent；Milvus vector index 已在 Phase 2 baseline 中使用。

---

## Project Structure

```text
.
├── README_QREAM_MASS_RAG.md          # 项目入口说明
├── PROJECT_PLAN.md                   # 详细实施计划
├── README_DATASETS.md                # 数据集选择与 corpus 构建说明
├── README_MILVUS.md                  # Milvus 本地启动与 collection 设计
├── README_ELASTICSEARCH.md           # Elasticsearch BM25 后端启动与索引说明
├── requirements.txt                  # Python 依赖
│
├── configs/                          # 配置文件
│   ├── data.yaml                     # 数据路径、split、样本数量配置占位
│   ├── retrieval.yaml                # BM25、embedding、Milvus、reranker 参数
│   ├── model.yaml                    # LLM provider、模型名、生成参数占位
│   ├── router.yaml                   # simple / complex 路由与升级策略占位
│   ├── agent.yaml                    # QREAM 和多 Agent 开关占位
│   └── experiment.yaml               # 实验方法、输出路径、评测设置占位
│
├── data/                             # 数据目录
│   ├── raw/
│   │   └── hotpotqa/
│   │       ├── hotpot_dev_distractor_v1.json    # dev distractor，Phase 1 调试和 dev 评测
│   │       └── hotpot_train_v1.1.json            # train，后续构建 global corpus
│   ├── processed/
│   │   └── hotpotqa/
│   │       ├── per_sample/            # Phase 1 processed JSONL 输出
│   │       └── global/                # Phase 2 corpus.jsonl / questions_dev.jsonl / title_to_doc_ids.json
│   └── indexes/                       # BM25 cache、Milvus 元数据、GPU embedding 分片
│       └── hotpotqa_global/
│           ├── bm25.pkl               # global BM25 cache，本地生成
│           ├── dense_build_meta.json  # dense / Milvus 构建记录
│           └── bge_m3_embeddings/     # GPU 服务器导出的 embedding .npz 分片
│
├── infra/
│   ├── milvus/                        # Milvus Docker Compose 文件
│   └── elasticsearch/                 # Elasticsearch BM25 Docker Compose 文件
│
├── outputs/                           # 运行输出
│   ├── predictions/                   # prediction JSONL
│   │   └── standard_rag_dev_smoke.jsonl
│   │                                   # Phase 1 smoke prediction 输出
│   ├── retrieved_docs/                # 后续检索结果详情
│   ├── rewritten_docs/                # 后续 QREAM rewritten docs
│   ├── route_logs/                    # 后续 router 决策日志
│   ├── logs/                          # 后续运行日志
│   └── reports/                       # 后续实验报告
│
├── scripts/                           # 命令行入口
│   ├── prepare_hotpotqa_per_sample.py          # raw HotpotQA -> per-sample processed JSONL
│   ├── run_global_bm25_rag.py                  # 默认运行 global BM25 + DashScope API；支持 per-sample/mock debug
│   ├── ask_bm25_rag_demo.py                    # 单问题 demo：输入 question，在小型 HotpotQA corpus 中检索回答
│   ├── evaluate_prediction_metrics.py          # 汇总 prediction JSONL 指标
│   ├── build_hotpotqa_indexes.py               # 构建 global corpus / dense embeddings / Milvus dense index
│   ├── query_milvus_dense_retrieval.py         # 单问题 dense retrieval smoke
│   ├── diagnose_dense_retrieval.py             # 批量 dense evidence recall 诊断，不调用 LLM
│   ├── diagnose_hybrid_retrieval.py            # 批量 BM25 + Dense hybrid evidence recall 诊断
│   ├── diagnose_decomposed_hybrid_retrieval.py # 批量 LLM 查询分解 + Hybrid evidence recall 诊断
│   ├── diagnose_hybrid_rerank.py               # 批量 Hybrid top50 + 本地/DashScope rerank 诊断
│   ├── run_hybrid_rerank_rag.py                # Hybrid + Rerank + DashScope answer baseline
│   ├── analyze_prediction_bad_cases.py         # prediction JSONL bad case 分析报告
│   ├── run_router_rag_placeholder.py           # 后续运行 Router-RAG
│   └── run_qream_mass_rag_placeholder.py       # 后续运行 QREAM + MASS-RAG
│
├── src/                               # 项目源码
│   ├── __init__.py                    # Python package 标记
│   │
│   ├── data/                          # 数据解析、文档化、预处理
│   │   ├── __init__.py
│   │   ├── schema.py                  # Document、RetrievedDoc、HotpotQASample、PipelineResult
│   │   ├── load_hotpotqa.py           # 读取 HotpotQA raw JSON
│   │   ├── build_corpus.py            # context paragraph -> Document chunk；supporting_facts 校验
│   │   └── preprocess.py              # raw samples -> processed per-sample JSONL
│   │
│   ├── retrieval/                     # BM25、Dense、Hybrid、Reranker、Milvus
│   │   ├── __init__.py
│   │   ├── bm25.py                    # Phase 1 BM25 检索器，基于 rank_bm25 包
│   │   ├── decomposed_hybrid.py       # 多 query BM25/Dense RRF 融合
│   │   ├── dense.py                   # BGE-M3 embedding 和 Dense Retriever
│   │   ├── hybrid.py                  # BM25 + Dense RRF 融合检索
│   │   ├── query_decomposition.py     # LLM 查询分解和 JSONL cache
│   │   ├── reranker.py                # 本地 Qwen3 reranker + DashScope qwen3-rerank API wrapper
│   │   ├── index_builder.py           # global corpus / dense index 构建逻辑
│   │   └── milvus_store.py            # Milvus collection / insert / search 封装
│   │
│   ├── pipeline/                      # Standard RAG、Router-RAG、QREAM-MASS-RAG
│   │   ├── __init__.py
│   │   ├── standard_rag.py            # Phase 1 Standard RAG：BM25 -> Mock answer -> metrics
│   │   ├── router_rag.py              # 后续 Router-RAG 主入口占位
│   │   └── qream_mass_rag.py          # 后续复杂路径 QREAM + Multi-Agent 占位
│   │
│   ├── evaluation/                    # 答案、证据、引用、路由、成本评测
│   │   ├── __init__.py
│   │   ├── answer_metrics.py          # Answer EM / F1
│   │   ├── evidence_metrics.py        # 后续 supporting fact / evidence recall 指标占位
│   │   ├── citation_metrics.py        # 后续 citation precision / recall 指标占位
│   │   ├── faithfulness.py            # 后续 unsupported claim / groundedness 指标占位
│   │   ├── route_metrics.py           # 后续 route distribution / upgrade rate 指标占位
│   │   └── cost_metrics.py            # 后续 token、latency、LLM calls 聚合占位
│   │
│   ├── utils/                         # IO、文本处理、LLMClient、日志
│   │   ├── __init__.py
│   │   ├── io.py                      # JSON / JSONL 读写工具
│   │   ├── text.py                    # 文本 normalize、tokenize、answer normalize
│   │   ├── llm_client.py              # LLMClient 协议和 Phase 1 MockLLMClient
│   │   └── logger.py                  # 后续日志工具占位
│   │
│   ├── router/                        # 后续 simple / complex 问题路由
│   │   ├── __init__.py
│   │   ├── schema.py                  # 后续 QuestionRoute schema 占位
│   │   ├── rules.py                   # 后续规则路由信号占位
│   │   ├── llm_classifier.py          # 后续 LLM fallback classifier 占位
│   │   └── question_router.py         # 后续问题路由器占位
│   │
│   ├── qream/                         # 后续 QREAM-style 文档重写
│   │   ├── __init__.py
│   │   ├── schema.py                  # 后续 RewrittenDoc schema 占位
│   │   ├── prompts.py                 # 后续 QREAM prompt 模板占位
│   │   └── rewriter.py                # 后续 QREAM-style document rewriter 占位
│   │
│   └── agents/                        # 后续 Extractor / Reasoner / Synthesizer / Verifier
│       ├── __init__.py
│       ├── base.py                    # 后续 BaseAgent 抽象占位
│       ├── extractor.py               # 后续 Evidence Extractor Agent 占位
│       ├── reasoner.py                # 后续 Reasoner Agent 占位
│       ├── synthesizer.py             # 后续 Synthesis Agent 占位
│       ├── verifier.py                # 后续 Verifier Agent 占位
│       └── summarizer.py              # 后续可选 Summarizer Agent 占位
│
└── tests/                             # 单元测试和 smoke tests
    ├── __init__.py                    # Python package 标记
    ├── test_metrics.py                # Answer EM / F1 测试
    ├── test_retrieval.py              # BM25 排序测试
    ├── test_pipeline.py               # Standard RAG smoke test
    ├── test_router.py                 # 后续 router 测试占位
    ├── test_qream.py                  # 后续 QREAM 测试占位
    └── test_agents.py                 # 后续 Agent 测试占位
```

### `src/` Module Details

| Module | Key Files | Responsibility |
|---|---|---|
| `src/data/` | `schema.py`, `load_hotpotqa.py`, `build_corpus.py`, `preprocess.py` | 定义核心数据结构，读取 HotpotQA，将 paragraph 转成 Document chunks |
| `src/retrieval/` | `bm25.py`, `dense.py`, `hybrid.py`, `decomposed_hybrid.py`, `query_decomposition.py`, `reranker.py`, `milvus_store.py` | Phase 1 已用 `rank_bm25` 实现 BM25；Phase 2 已接 Milvus dense retrieval、hybrid fusion、查询分解诊断和 reranker |
| `src/pipeline/` | `standard_rag.py`, `router_rag.py`, `qream_mass_rag.py` | 编排端到端流程；当前已实现 Standard RAG smoke baseline |
| `src/evaluation/` | `answer_metrics.py`, `evidence_metrics.py`, `citation_metrics.py`, `route_metrics.py`, `cost_metrics.py` | 当前已实现 Answer EM/F1；后续补证据、引用、路由和成本指标 |
| `src/utils/` | `io.py`, `text.py`, `llm_client.py`, `logger.py` | 通用 IO、文本归一化、LLMClient 抽象和日志工具 |
| `src/router/` | `question_router.py`, `rules.py`, `llm_classifier.py`, `schema.py` | 后续实现 simple / complex 路由和 LLM fallback |
| `src/qream/` | `rewriter.py`, `prompts.py`, `schema.py` | 后续实现 QREAM-style document rewriting |
| `src/agents/` | `base.py`, `extractor.py`, `reasoner.py`, `synthesizer.py`, `verifier.py` | 后续实现多 Agent 证据抽取、推理、综合和校验 |

---

## Main Experiments

| Method | Purpose |
|---|---|
| No RAG | 检查 LLM 自身能力 |
| Standard RAG All | 所有问题都使用普通 RAG |
| QREAM + MASS All | 所有问题都使用复杂流程 |
| Router-RAG without Upgrade | 验证 router 本身效果 |
| Router-RAG | 验证路由 + 自动升级的综合效果 |

核心指标：

- Answer EM / F1
- Supporting Fact F1
- Evidence Recall@5 / Recall@10
- Citation Precision / Recall
- Unsupported Claim Rate
- Route Accuracy / Upgrade Rate
- Average LLM Calls / Tokens / Latency

---

## Project Plan

完整路线、阶段验收、接口设计、实验矩阵和测试计划见：

- [PROJECT_PLAN.md](PROJECT_PLAN.md)

数据集选择、corpus 构建方式和各阶段数据用途见：

- [README_DATASETS.md](README_DATASETS.md)

Milvus 本地启动、collection 设计和向量检索计划见：

- [README_MILVUS.md](README_MILVUS.md)

Elasticsearch BM25 后端启动、构建和对比实验见：

- [README_ELASTICSEARCH.md](README_ELASTICSEARCH.md)
