# Router-QREAM-MASS-RAG Project Plan

## 1. 项目目标

本项目要实现一个面向 **Multi-hop QA** 的路由式可信 RAG 系统。

系统不把所有问题都交给同一个复杂 pipeline，而是先判断问题复杂度：

```text
简单事实型问题 -> Standard RAG
复杂多跳问题 -> QREAM + Multi-Agent RAG
```

最终目标：

- 简单问题低成本、低延迟回答。
- 复杂问题进行多证据检索、文档重写、多跳推理和答案校验。
- 统一输出答案、引用、证据、路由结果、成本和评测指标。
- 通过实验说明 Router-RAG 相比全量复杂流程能节省成本，同时保持接近的答案质量和更好的证据可信度。

---

## 2. 设计原则

### 2.1 先跑通，再增强

第一阶段只做最小可运行 Standard RAG，不提前实现 QREAM、多 Agent 或复杂消融。

开发顺序必须保持：

```text
Standard RAG
-> Retrieval Upgrade
-> Question Router
-> Router-RAG
-> QREAM + Multi-Agent
-> Full Experiments
```

### 2.2 简单问题不走重流程

简单问题默认只走：

```text
retrieval -> reranker -> answer generation -> lightweight verifier
```

不调用 QREAM，不调用多 Agent。

### 2.3 复杂问题才走深度推理

复杂问题默认走：

```text
hybrid retrieval
-> reranker
-> QREAM-style rewriting
-> Extractor Agent
-> Reasoner Agent
-> Synthesizer Agent
-> Verifier Agent
```

### 2.4 所有结果必须可评测

每条输出都必须保留：

- question id
- route_initial
- route_final
- was_upgraded
- retrieved_docs
- pred_answer
- citations
- agent_outputs
- metrics
- cost

### 2.5 Gold 信息只用于评测

HotpotQA 的 `answer` 和 `supporting_facts` 只能用于 evaluation，不能用于：

- router
- retrieval
- QREAM rewriting
- agent reasoning
- answer generation

---

## 3. 系统架构

### 3.1 总流程

```text
User Question
  -> Question Router
      -> simple:
           Standard RAG
           -> lightweight verifier
           -> if failed: upgrade to complex route

      -> complex:
           QREAM + Multi-Agent RAG
```

### 3.2 Simple Route

```text
question
  -> retrieve top-k docs
  -> rerank
  -> answer generation
  -> lightweight verifier
  -> final answer with citations
```

Simple route 目标：

- 控制 LLM 调用次数。
- 降低平均 token cost。
- 对单跳事实型问题保持足够高的 Answer F1。
- 保留 citations，方便统一评测。

### 3.3 Complex Route

```text
question
  -> hybrid retrieval
  -> reranker
  -> QREAM-style document rewriting
  -> Extractor Agent
  -> Reasoner Agent
  -> Synthesizer Agent
  -> Verifier Agent
  -> final answer with citations
```

Complex route 目标：

- 抽取多个支持证据。
- 连接 bridge / comparison 问题所需的中间实体和关系。
- 显式生成推理链。
- 检查答案是否被引用证据支持。

### 3.4 自动升级

Simple route 结果满足以下任一条件时，升级到 complex route：

- `confidence < 0.55`
- `citations` 为空
- `verifier.is_supported = false`
- `pred_answer` 为空
- 模型明确拒答
- 检索最高分低于配置阈值

复杂流程结果覆盖简单流程结果，但最终输出保留：

```text
route_initial = simple
route_final = complex
was_upgraded = true
```

---

## 4. 数据计划

详细数据集说明已单独整理到：

- [README_DATASETS.md](README_DATASETS.md)

本节只保留项目计划中的数据使用摘要。

### 4.1 数据路线

| 阶段 | 数据 | Corpus 形式 | 用途 |
|---|---|---|---|
| Phase 1 | HotpotQA distractor | Per-sample corpus | 跑通 pipeline 和评测闭环 |
| Phase 2 | HotpotQA 去重语料 | Global deduplicated corpus | 正式 RAG 检索实验 |
| Phase 3 | Natural Questions 小样本 | Wikipedia page / paragraph corpus | 补充 simple route |
| Phase 4 | 2WikiMultiHopQA | Dataset context corpus | 验证复杂路径泛化 |
| Phase 5 | MuSiQue | Dataset context corpus | 复杂多跳压力测试 |

第一版只实现：

```text
HotpotQA distractor per-sample corpus
HotpotQA deduplicated global corpus
```

### 4.2 使用原则

- Phase 1 是过渡和 sanity check，不作为最终主结果。
- Phase 2 是第一版正式主实验设置。
- Gold answer 和 supporting_facts 只能用于 evaluation。
- Natural Questions、2WikiMultiHopQA、MuSiQue 暂时写入计划，后续扩展。

### 4.3 数据规模

开发阶段建议：

```text
smoke test: 20-50 条
debug experiment: 100 条
main dev experiment: 300-1000 条
extended experiment: 1000-5000 条
```

---

## 5. 核心接口

### 5.1 Document

```python
{
    "doc_id": "Arthur's Magazine__0",
    "title": "Arthur's Magazine",
    "text": "...",
    "sentences": ["...", "..."],
    "metadata": {
        "dataset": "hotpotqa",
        "source_question_id": "...",
        "paragraph_index": 0
    }
}
```

### 5.2 RetrievedDoc

```python
{
    "doc_id": "...",
    "title": "...",
    "text": "...",
    "sentences": ["..."],
    "score": 0.93,
    "rank": 1,
    "retrieval_source": "bm25 | dense | hybrid | reranker"
}
```

### 5.3 QuestionRoute

```python
{
    "route": "simple",
    "confidence": 0.82,
    "reason": "single-entity factual question",
    "signals": ["single_entity", "no_comparison_keyword"]
}
```

### 5.4 RewrittenDoc

```python
{
    "source_doc_id": "...",
    "source_title": "...",
    "source_sentence_ids": [0, 2],
    "rewritten_text": "...",
    "preserved_evidence": ["..."]
}
```

### 5.5 Evidence

```python
{
    "claim": "...",
    "source_doc_id": "...",
    "source_title": "...",
    "source_sentence_id": 0,
    "source_text": "..."
}
```

### 5.6 PipelineResult

```python
{
    "id": "sample_id",
    "question": "...",
    "gold_answer": "...",
    "route_initial": {},
    "route_final": "simple | complex",
    "was_upgraded": false,
    "pred_answer": "...",
    "pred_citations": [],
    "retrieved_docs": [],
    "rewritten_docs": [],
    "agent_outputs": {},
    "metrics": {},
    "cost": {}
}
```

### 5.7 LLMClient

所有模型调用统一走：

```python
class LLMClient:
    def generate(self, messages, **kwargs) -> str:
        ...

    def generate_json(self, messages, schema=None, **kwargs) -> dict:
        ...
```

`LLMClient` 必须负责：

- 本地模型和 API 模型的统一入口。
- token 统计。
- latency 统计。
- JSON 解析失败的重试或错误返回。

---

## 6. 模块计划

### 6.1 Data

路径：

```text
src/data/
```

需要实现：

- `load_hotpotqa.py`：加载 HotpotQA JSON。
- `schema.py`：定义样本、文档、证据等数据结构。
- `build_corpus.py`：从 context 构建 Document corpus。
- `preprocess.py`：生成 processed data。

验收标准：

- 能读取 HotpotQA distractor 样本。
- 能将每个 paragraph 转为 Document。
- 能保留 title、sentence_id、question_id、paragraph_index。
- 能输出 JSONL 或缓存文件。

### 6.2 Retrieval

路径：

```text
src/retrieval/
```

需要实现：

- BM25 Retriever
- Dense Retriever
- Hybrid Retriever
- Reranker wrapper
- Index builder

第一阶段只要求 BM25，默认使用 `rank_bm25` 包实现，不维护手写 BM25 公式版本。

验收标准：

- 输入 question 和 corpus，输出排序后的 RetrievedDoc。
- 每个结果包含 score、rank、retrieval_source。
- 支持 top_k 配置。
- 能计算 Evidence Recall@k。

### 6.3 Router

路径：

```text
src/router/
```

需要实现：

- 规则 router。
- LLM fallback classifier。
- QuestionRoute schema。

规则信号：

- comparison keyword：earlier、later、first、more、less、larger、smaller、older、younger。
- temporal relation：founded before、started first、born after、died before。
- multiple entities：多个专有名词、多个引号实体、多个逗号连接实体。
- bridge pattern：`who/what/which + relation + entity`。
- explicit complex signal：both、between、compared with、same as、different from。

验收标准：

- 单实体事实型问题路由到 simple。
- 比较问题路由到 complex。
- bridge 问题路由到 complex。
- 不确定样本触发 LLM fallback。
- Router 不读取 gold answer 或 supporting_facts。

### 6.4 QREAM

路径：

```text
src/qream/
```

需要实现：

- QREAM-style rewriter。
- Prompt 模板。
- RewrittenDoc schema。

重写约束：

- 只能使用原文信息。
- 不直接回答问题。
- 必须保留 `source_doc_id`。
- 尽量保留 `source_sentence_ids`。
- 输出 JSON。

验收标准：

- 对 top-k docs 输出 rewritten_docs。
- 每条 rewritten_doc 可追溯到原文。
- 不生成没有来源的新事实。

### 6.5 Agents

路径：

```text
src/agents/
```

第一版主路径实现四个 Agent：

- Extractor Agent
- Reasoner Agent
- Synthesizer Agent
- Verifier Agent

Summarizer 暂时只保留占位，后续作为消融实验。

验收标准：

- 每个 Agent 继承 BaseAgent。
- 每个 Agent 输出结构化 JSON。
- JSON 解析失败时返回可记录的错误对象。
- Verifier 能判断答案是否被 citations 支持。

### 6.6 Pipelines

路径：

```text
src/pipeline/
```

需要实现：

- `standard_rag.py`
- `qream_mass_rag.py`
- `router_rag.py`

验收标准：

- 三个 pipeline 输出统一 PipelineResult。
- Router-RAG 能根据 route 调用不同 pipeline。
- Simple route 失败时能自动升级。
- 所有结果可被 evaluation script 读取。

### 6.7 Evaluation

路径：

```text
src/evaluation/
```

需要实现：

- Answer EM / F1
- Supporting Fact EM / F1
- Evidence Recall@k
- Citation Precision / Recall
- Unsupported Claim Rate
- Route Distribution / Upgrade Rate
- Cost aggregation

验收标准：

- 能读取 prediction JSONL。
- 能输出总体指标表。
- 能按 route 分组统计。
- 能输出 bad case 文件。

---

## 7. 开发阶段

### Phase 1: Standard RAG Baseline

目标：

```text
跑通最小普通 RAG baseline。
```

任务：

- 加载 HotpotQA distractor。
- 构造 Document corpus。
- 实现 BM25 Retriever。
- 实现 Standard RAG pipeline。
- 实现 LLMClient 抽象。
- 输出 prediction JSONL。
- 实现 Answer EM / F1。

产出：

- `scripts/prepare_data.py`
- `scripts/run_standard_rag.py`
- `scripts/evaluate.py`
- Standard RAG 结果表。

验收：

- 20 条样本 smoke test 可跑通。
- 每条输出包含 `pred_answer`、`retrieved_docs`、`cost`。
- Answer EM / F1 能正常计算。

### Phase 2: Retrieval Upgrade

目标：

```text
提升检索质量，并开始评估证据召回。
Phase 2 开始引入 Milvus 作为 dense vector store。
```

任务：

- 构建 HotpotQA deduplicated global corpus。（已实现 JSONL corpus / dev questions / title index）
- 实现 global BM25 Standard RAG baseline。（已实现默认 global corpus + DashScope API 路径 + BM25 cache）
- 实现 Milvus vector store。（已实现 hotpotqa_global_chunks collection 封装）
- 实现 Dense Retriever。（已实现 BGE-M3 -> Milvus dense top-k）
- 支持 GPU 服务器离线导出 BGE-M3 embedding 分片，并在本地导入 Milvus。（已实现 `dense-embeddings` / `milvus-import-embeddings`）
- 实现 Hybrid Retriever。
- 接入 Reranker。
- 实现 Evidence Recall@k。
- 实现 Supporting Fact F1。
- 保存 retrieved_docs。

产出：

- BM25 vs Milvus Dense vs Hybrid vs Hybrid + Reranker 对比表。

验收：

- Milvus collection 可创建、写入和查询。
- `Evidence Recall@5` 和 `Evidence Recall@10` 可计算。
- retrieved_docs 包含 title、score、rank、source。

### Phase 3: Question Router

目标：

```text
实现 simple / complex 路由。
```

任务：

- 实现 rule router。
- 实现 LLM fallback classifier。
- 输出 QuestionRoute。
- 实现 route_metrics。
- 人工检查 100 条样本路由是否合理。

产出：

- route distribution。
- route accuracy 近似分析。
- router bad cases。

验收：

- Router 单测通过。
- 不确定问题会触发 fallback。
- Router 输出可写入 prediction JSONL。

### Phase 4: Router-RAG Simple Path

目标：

```text
把 router 和 Standard RAG 串起来。
```

任务：

- 实现 `router_rag.py`。
- simple route 调用 Standard RAG。
- complex route 暂时可返回占位错误或调用未实现 stub。
- 实现 lightweight verifier。
- 实现 `should_upgrade`。

产出：

- Router-RAG without complex route smoke test。

验收：

- simple 问题能完整输出。
- 低置信或无引用样本能标记 `was_upgraded = true`。
- 输出包含 `route_initial`、`route_final`、`was_upgraded`。

### Phase 5: QREAM + Multi-Agent Complex Path

目标：

```text
实现复杂问题路径。
```

任务：

- 实现 QREAM Rewriter。
- 实现 Extractor Agent。
- 实现 Reasoner Agent。
- 实现 Synthesizer Agent。
- 实现 Verifier Agent。
- 接入 `qream_mass_rag.py`。

产出：

- QREAM + MASS-RAG complex pipeline result。

验收：

- complex route 能完整输出 rewritten_docs 和 agent_outputs。
- citations 可追溯到原始句子。
- verifier 能输出 supported / unsupported。

### Phase 6: Full Router-RAG

目标：

```text
完成完整路由式系统。
```

任务：

- complex route 接入 Router-RAG。
- simple route 失败时自动升级。
- 输出完整 PipelineResult。
- 跑主实验和路由消融。

产出：

- Router-RAG main result。
- Router-RAG without Upgrade result。
- Standard RAG All result。
- QREAM + MASS All result。

验收：

- 至少 300 条 dev 样本可跑完整实验。
- 输出主结果表、路由分析表、成本对比表。
- 至少整理 3-5 个 bad cases。

---

## 8. 实验计划

### 8.1 主实验

| Method | Purpose |
|---|---|
| No RAG | 检查 LLM 自身能力 |
| Standard RAG All | 所有问题都使用普通 RAG |
| QREAM + MASS All | 所有问题都使用复杂流程 |
| Router-RAG without Upgrade | 验证 router 本身效果 |
| Router-RAG | 验证路由 + 自动升级的综合效果 |

预期结论：

- Router-RAG 的 Answer F1 接近 QREAM + MASS All。
- Router-RAG 的成本明显低于 QREAM + MASS All。
- Router-RAG 的 citation / faithfulness 优于 Standard RAG All。
- Upgrade 机制能修复一部分 simple route 失败样本。

### 8.2 Router 消融

| Method | Purpose |
|---|---|
| Rule Router only | 检查纯规则路由效果 |
| Rule + LLM Router | 检查 LLM fallback 是否有效 |
| Router without Upgrade | 检查没有升级机制时的错误 |
| Router with Upgrade | 检查自动升级是否恢复失败样例 |

### 8.3 Complex Route 消融

| Method | Purpose |
|---|---|
| Complex without QREAM | 验证文档重写是否有效 |
| Complex without Verifier | 验证答案校验是否有效 |
| Extractor only | 验证只抽证据是否足够 |
| Extractor + Reasoner | 验证显式推理是否有效 |

第一版不做所有 Agent 全排列，避免实验爆炸。

---

## 9. 指标计划

### 9.1 Answer Metrics

- Answer EM
- Answer F1

### 9.2 Evidence Metrics

- Supporting Fact EM
- Supporting Fact F1
- Evidence Recall@5
- Evidence Recall@10

### 9.3 Citation Metrics

- Citation Precision
- Citation Recall
- Citation Accuracy

### 9.4 Faithfulness Metrics

- Unsupported Claim Rate
- Faithfulness
- Groundedness

### 9.5 Route Metrics

- Route Distribution
- Route Accuracy
- Upgrade Rate
- Complex Recovery Rate
- Wrong Route Error Rate

### 9.6 Cost Metrics

- Average LLM Calls
- Average Input Tokens
- Average Output Tokens
- Average Latency
- Cost Saving vs Complex

---

## 10. 输出文件

### 10.1 Prediction JSONL

每条样本一行：

```json
{
  "id": "sample_id",
  "question": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "route_initial": {
    "route": "simple",
    "confidence": 0.82,
    "reason": "...",
    "signals": ["single_entity"]
  },
  "route_final": "complex",
  "was_upgraded": true,
  "gold_supporting_facts": [["title", 0]],
  "pred_citations": [],
  "retrieved_docs": [],
  "rewritten_docs": [],
  "agent_outputs": {},
  "metrics": {},
  "cost": {}
}
```

### 10.2 Reports

最终至少输出：

- 主结果表。
- 路由分析表。
- 成本对比表。
- Bad case 分析。
- 简历项目描述。

---

## 11. 推荐表格

### 11.1 主结果表

| Method | Ans F1 | Sup F1 | Evidence R@5 | Citation P | Citation R | Unsupported Rate | Calls | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No RAG | - | - | - | - | - | - | - | - |
| Standard RAG All | - | - | - | - | - | - | - | - |
| QREAM + MASS All | - | - | - | - | - | - | - | - |
| Router-RAG without Upgrade | - | - | - | - | - | - | - | - |
| Router-RAG | - | - | - | - | - | - | - | - |

### 11.2 路由分析表

| Method | Simple % | Complex % | Upgrade % | Route Acc | Recovery Rate | Wrong Route Error |
|---|---:|---:|---:|---:|---:|---:|
| Rule Router only | - | - | - | - | - | - |
| Rule + LLM Router | - | - | - | - | - | - |
| Router-RAG | - | - | - | - | - | - |

### 11.3 成本对比表

| Method | Avg Calls | Avg Input Tokens | Avg Output Tokens | Avg Latency | Cost Saving vs Complex |
|---|---:|---:|---:|---:|---:|
| Standard RAG All | - | - | - | - | - |
| QREAM + MASS All | - | - | - | - | - |
| Router-RAG | - | - | - | - | - |

---

## 12. 测试计划

### 12.1 Unit Tests

必须覆盖：

- HotpotQA 数据解析。
- Document schema 构造。
- BM25 retrieval 输出格式。
- Answer EM / F1。
- Supporting Fact F1。
- Citation Precision / Recall。
- Router simple / complex 分类。
- `should_upgrade` 逻辑。

### 12.2 Smoke Tests

必须覆盖：

- 20 条样本 Standard RAG 跑通。
- 20 条样本 Router-RAG 跑通。
- 复杂路径至少 5 条样本跑通。
- prediction JSONL 能被 evaluate script 读取。

### 12.3 Experiment Checks

必须检查：

- 每种方法输出样本数量一致。
- 每条样本都有 route 信息。
- 每条样本都有 cost 信息。
- 空 citation 不导致 evaluation 崩溃。
- JSON 解析失败能记录错误而不是中断全局实验。

---

## 13. 当前仓库状态

当前已经完成 Phase 1 smoke baseline，并开始 Phase 2 global corpus：

```text
HotpotQA loader
per-sample BM25 Standard RAG
Answer / Evidence metrics
DashScope client
HotpotQA global deduplicated corpus builder
Global BM25 Standard RAG baseline
```

下一步从 **Milvus Dense / Hybrid Retrieval Upgrade** 继续。

---

## 14. 第一阶段实施清单

下一次真正开始写代码时，优先完成：

```text
1. 在 src/data/schema.py 定义 Document、HotpotQASample、PipelineResult。
2. 在 src/data/load_hotpotqa.py 实现 HotpotQA JSON loader。
3. 在 src/data/build_corpus.py 实现 context -> Document list。
4. 在 src/retrieval/bm25.py 实现 BM25 Retriever。
5. 在 src/utils/llm_client.py 定义 LLMClient 抽象和 MockLLMClient。
6. 在 src/pipeline/standard_rag.py 实现最小 Standard RAG。
7. 在 src/evaluation/answer_metrics.py 实现 EM / F1。
8. 在 scripts/run_standard_rag.py 串起第一阶段流程。
9. 在 tests/ 中补最小单测。
```

第一阶段不要做：

- Dense retrieval
- Milvus
- Reranker
- Router
- QREAM
- Multi-Agent
- 完整消融实验

---

## 15. 简历表述方向

```text
构建 Router-QREAM-MASS-RAG：一个面向多跳问答的路由式可信 RAG 系统。项目基于 HotpotQA，设计 Question Router 将简单事实型问题分发到低成本 Standard RAG，将复杂 bridge / comparison 问题分发到 QREAM-style 文档重写与多 Agent 证据综合流程。系统实现 BM25 + Dense Hybrid Retrieval、Reranker、证据抽取、多跳推理、答案综合与答案校验，并通过 Answer F1、Supporting Fact F1、Citation Precision / Recall、Unsupported Claim Rate、LLM Calls 和 Latency 等指标评估答案质量、引用可信度和成本收益。
```
