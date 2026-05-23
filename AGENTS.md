# AGENTS.md

这个文件给 AI coding agent 提供本仓库的项目上下文和协作规则。

## 代码风格

在保障代码易读的同时，尽量保障简洁

## 项目简介

本仓库是 `Router-QREAM-MASS-RAG`，一个面向多跳问答的 RAG 项目。

核心思路：

```text
简单问题  -> Standard RAG
复杂问题  -> QREAM-style 文档重写 + 多证据抽取 + 多 Agent 推理
```

项目第一阶段使用 HotpotQA，后续会扩展到 Milvus global corpus 检索。

## 当前阶段

当前已完成：

```text
Phase 1 MVP / smoke baseline
```

已经实现：

- HotpotQA raw JSON loader。
- `context` paragraph 到 `Document` chunk 的转换。
- 基于 `rank_bm25` 的 per-sample BM25 检索。
- Standard RAG pipeline。
- Mock LLM 和阿里 DashScope client。
- Prediction JSONL 输出。
- Answer EM / F1。


## 运行环境

默认使用 conda 环境：

```bash
conda activate qream-rag
```

如果用非交互方式运行命令，优先使用：

```bash
conda run -n qream-rag python ...
conda run -n qream-rag pip ...
```

用户期望所有 Python 相关命令都在 `qream-rag` 环境中执行。

## 常用命令

数据预处理 smoke test：

```bash
conda run -n qream-rag python scripts/prepare_data.py \
  --limit 20 \
  --output data/processed/hotpotqa/per_sample/dev_samples_smoke.jsonl
```

批量 mock RAG：

```bash
conda run -n qream-rag python scripts/run_standard_rag.py \
  --limit 20 \
  --top-k 5 \
  --llm mock \
  --output outputs/predictions/standard_rag_mock.jsonl
```

单问题 mock demo：

```bash
conda run -n qream-rag python scripts/ask_standard_rag.py \
  --question "Which magazine was started first, Arthur's Magazine or First for Women?" \
  --corpus-limit 7405 \
  --top-k 5 \
  --llm mock
```

阿里 API 模式需要 `.env`：

```text
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

## 代码规则

- 优先写清楚易读的代码，不追求过度压缩。
- 如果改变架构或阶段计划，需要同步更新 `README_QREAM_MASS_RAG.md` 和 `PROJECT_PLAN.md`。
- 如果新增依赖，需要更新 `requirements.txt`。
- 如果新增脚本，需要更新 README 中的项目结构。

## 文档

关键文档：

- `README_QREAM_MASS_RAG.md`：项目总览和目录结构。
- `PROJECT_PLAN.md`：详细实施计划。
- `README_DATASETS.md`：数据集和 corpus 策略。
- `README_MILVUS.md`：Milvus 计划和启动方式。

代码行为变化时，要同步更新相关文档。
