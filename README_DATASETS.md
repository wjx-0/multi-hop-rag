# Datasets README

本文件单独整理 Router-QREAM-MASS-RAG 项目使用的数据集和 corpus 构建方式。

核心结论：

```text
Phase 1 用 HotpotQA distractor per-sample corpus 过渡和调试。
Phase 2 用 HotpotQA deduplicated global corpus 做正式 RAG 检索实验。
后续用 Natural Questions 补 simple route，用 2WikiMultiHopQA / MuSiQue 验证复杂多跳泛化。
```

---

## 1. 数据集选择总览

| 阶段 | 数据 | Corpus 形式 | 主要用途 | 是否主实验 |
|---|---|---|---|---|
| Phase 1 | HotpotQA distractor | Per-sample corpus | 跑通 pipeline 和评测闭环 | 否，作为 sanity check |
| Phase 2 | HotpotQA distractor/train 去重 | Global deduplicated corpus | 正式 RAG 检索实验 | 是 |
| Phase 3 | Natural Questions 小样本 | Wikipedia page / paragraph corpus | 补充 simple route 数据 | 辅助 |
| Phase 4 | 2WikiMultiHopQA | Dataset context corpus | 验证复杂路径泛化 | 扩展 |
| Phase 5 | MuSiQue | Dataset context corpus | 复杂多跳压力测试 | 扩展 |

第一版只需要实现：

```text
HotpotQA distractor per-sample corpus
HotpotQA deduplicated global corpus
```

Natural Questions、2WikiMultiHopQA、MuSiQue 先写入计划，不放入第一版实现范围。

---

## 2. Phase 1: HotpotQA Distractor Per-Sample Corpus

### 2.1 用途

Phase 1 用来快速跑通：

- HotpotQA loader
- Document schema
- BM25 Retriever
- Standard RAG baseline
- Answer EM / F1
- Supporting facts 和 citation 对齐
- JSONL prediction 输出

它的角色是 **工程过渡层和 sanity check**，不是最终最有说服力的 RAG 设置。

### 2.2 原始数据长什么样

HotpotQA distractor 的一条样本大致是：

```json
{
  "_id": "sample_id",
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "type": "comparison",
  "level": "medium",
  "supporting_facts": [
    ["Arthur's Magazine", 1],
    ["First for Women", 1]
  ],
  "context": [
    [
      "Arthur's Magazine",
      [
        "Arthur's Magazine was an American literary periodical published in Philadelphia.",
        "It was founded in 1844."
      ]
    ],
    [
      "First for Women",
      [
        "First for Women is a woman's magazine.",
        "The magazine was started in 1989."
      ]
    ],
    [
      "Distractor Title",
      [
        "This paragraph is related by keywords but does not support the answer."
      ]
    ]
  ]
}
```

关键字段：

- `_id`：样本 id
- `question`：问题
- `answer`：标准答案
- `type`：`bridge` 或 `comparison`
- `level`：`easy` / `medium` / `hard`
- `supporting_facts`：标准证据句，格式是 `[title, sent_id]`
- `context`：候选段落，每个段落是 `[title, sentences]`

### 2.3 Phase 1 的 Corpus 长什么样

Phase 1 不构建全局文档库，而是每条样本临时构建一个小 corpus：

```json
[
  {
    "doc_id": "sample_id::0",
    "title": "Arthur's Magazine",
    "text": "Arthur's Magazine was an American literary periodical published in Philadelphia. It was founded in 1844.",
    "sentences": [
      "Arthur's Magazine was an American literary periodical published in Philadelphia.",
      "It was founded in 1844."
    ],
    "metadata": {
      "dataset": "hotpotqa",
      "corpus_type": "per_sample",
      "source_question_id": "sample_id",
      "paragraph_index": 0
    }
  }
]
```

检索范围：

```text
当前 question 只在当前样本自己的 context 中检索。
```

### 2.4 为什么需要 Phase 1

如果直接上全局 corpus，失败原因会很难定位：

- loader 是否解析错？
- sentence_id 是否对齐错？
- BM25 是否实现错？
- corpus 去重是否错？
- citation 是否匹配错？
- evaluation 是否写错？

Phase 1 先把这些基础问题排干净。

### 2.5 Phase 1 的局限

Phase 1 不是真实开放 RAG：

```text
候选文档已经被数据集放在每个问题旁边了。
检索空间很小。
gold paragraphs 很可能已经在 context 里。
```

所以 Phase 1 只能作为 debug / sanity check，不能作为最终主结果。

---

## 3. Phase 2: HotpotQA Deduplicated Global Corpus

### 3.1 用途

Phase 2 是本项目第一版的正式 RAG 设置。

目标：

- 验证 BM25 / Dense / Hybrid / Reranker 的真实检索能力。
- 让 Router-RAG 面对更接近真实 RAG 的搜索空间。
- 继续保留 HotpotQA 的 supporting facts，方便自动评测证据。

### 3.2 原始数据来源

仍然使用 HotpotQA 的 `context` 段落，但构建方式不同。

Phase 2 会把多个 split 或多个样本里的 paragraph 全部抽出来：

```text
hotpot_train_v1.1.json
hotpot_dev_distractor_v1.json
```

然后按规则去重。

第一版推荐去重 key：

```text
normalized_title + normalized_text
```

不要只按 title 去重，因为同一 title 可能在不同样本中出现不同 sentence 版本。

### 3.3 Phase 2 的 Corpus 长什么样

全局 corpus 是 JSONL，每一行一个 document：

```json
{
  "doc_id": "hotpotqa::global::00000001",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine was an American literary periodical published in Philadelphia. It was founded in 1844.",
  "sentences": [
    "Arthur's Magazine was an American literary periodical published in Philadelphia.",
    "It was founded in 1844."
  ],
  "metadata": {
    "dataset": "hotpotqa",
    "corpus_type": "global_deduplicated",
    "source_question_ids": ["sample_id_1", "sample_id_2"],
    "paragraph_indices": [0, 3],
    "dedup_key": "arthur's magazine::<hash>"
  }
}
```

问题集单独保存：

```json
{
  "id": "sample_id",
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "type": "comparison",
  "level": "medium",
  "supporting_facts": [
    ["Arthur's Magazine", 1],
    ["First for Women", 1]
  ]
}
```

检索范围：

```text
每个 question 都从整个 global corpus 中检索。
```

### 3.4 为什么 Phase 2 更重要

Phase 2 更像真实 RAG：

- 检索空间更大。
- 干扰文档更多。
- gold evidence 不一定排在前面。
- Hybrid retrieval 和 reranker 的价值更明显。
- QREAM 和多 Agent 的收益更有说服力。

最终报告和简历中，主结果应该以 Phase 2 为主。

### 3.5 Phase 2 的注意点

需要特别处理：

- title 标准化
- text 标准化
- 去重后 doc_id 稳定生成
- supporting facts 的 `[title, sent_id]` 与 global doc 的 sentence 对齐
- 同 title 多版本段落的匹配问题
- 检索不到 gold paragraph 时的 evidence recall 统计

### 3.6 Global Corpus 评测注意事项

HotpotQA 的 `supporting_facts` 标注是：

```text
[title, sentence_id]
```

但 global corpus 的去重 key 是：

```text
normalized_title + normalized_text
```

因此评测时要注意：同一个 title 可能对应多个不同 paragraph 版本，甚至 sentence 切分略有差异。模型可能检索到语义上正确、title 也正确的证据，但如果严格按原始 `sentence_id` 对齐，`Supporting Fact F1` 仍可能被判错。

第一版建议把证据指标分层解释：

- `Evidence Recall@k`：优先作为 title-level / paragraph-level 检索覆盖指标，判断 gold title 是否进入 top-k。
- `Supporting Fact F1`：作为严格 citation 指标，要求 `title + sentence_id` 同时命中。
- `Answer F1`：单独评价最终答案，不要用它替代证据质量。

后续如果要更严谨，应实现一个 gold evidence resolver：

```text
gold [title, sentence_id]
-> 在 global corpus 中查找 normalized title 对应 docs
-> 根据 gold sentence 文本或 sentence overlap 解析到具体 doc_id + sentence_id
```

在 resolver 完成前，报告中需要说明 title-level evidence recall 偏宽，strict supporting-fact / citation 指标可能因为同 title 多版本和 sentence 对齐问题偏严。

---

## 4. Phase 3: Natural Questions 小样本

### 4.1 用途

Natural Questions 用来补充 simple route。

原因：

HotpotQA 主要是多跳问题，如果只用 HotpotQA，Router 可能大部分问题都会判成 complex，不利于展示：

```text
简单问题 -> Standard RAG
复杂问题 -> QREAM + Multi-Agent
```

Natural Questions 更接近真实用户搜索问题，适合构造 simple factual QA。

### 4.2 使用方式

第一版不需要引入。

后续可以抽取一个小样本：

```text
100-300 条 answerable examples
```

转换为统一格式：

```json
{
  "id": "nq_sample_id",
  "question": "...",
  "answer": "...",
  "context_docs": [
    {
      "title": "...",
      "sentences": ["..."]
    }
  ],
  "expected_route": "simple"
}
```

### 4.3 为什么不作为第一主数据

- 多跳 supporting facts 不如 HotpotQA 直接。
- citation 评测需要额外适配。
- 对 QREAM + Multi-Agent 复杂路径的支撑不如 HotpotQA。

所以 Natural Questions 只作为 simple route 补充数据。

---

## 5. Phase 4: 2WikiMultiHopQA

### 5.1 用途

2WikiMultiHopQA 用于验证复杂路径泛化能力。

它更强调 reasoning steps 和 evidence 信息，适合测试：

- Extractor Agent
- Reasoner Agent
- Verifier Agent
- complex route 的 evidence grounding

### 5.2 为什么放在后面

它需要额外 schema 适配和评测脚本适配。

第一版如果同时支持 HotpotQA 和 2WikiMultiHopQA，工程面会变大，容易拖慢主线。

推荐等 HotpotQA global corpus 跑通后再接入。

---

## 6. Phase 5: MuSiQue

### 6.1 用途

MuSiQue 用于最终复杂多跳压力测试。

它的设计目标是减少可以靠 shortcut 解题的伪多跳问题，更强调 connected reasoning，并包含 2-4 hop 问题。

适合验证：

- Router 是否能识别真正复杂问题。
- Reasoner 是否能处理多步依赖。
- QREAM 是否能帮助压缩和对齐复杂证据。

### 6.2 为什么不第一版使用

- 难度更高。
- 2-4 hop 会增加 prompt 和 agent 设计复杂度。
- 输出与评测适配成本高于 HotpotQA。

所以 MuSiQue 放在最终扩展，不进入 MVP。

---

## 7. 不建议第一版使用的数据

### 7.1 完整 Wikipedia dump

不建议第一版直接使用。

原因：

- 下载和清洗成本高。
- chunking 和索引成本高。
- 评测 evidence 对齐更麻烦。
- 会让项目初期陷入基础设施工作。

HotpotQA global deduplicated corpus 已经足够作为第一版真实 RAG 检索设置。

### 7.2 ASQA

不建议第一版使用。

原因：

- 更偏长答案和 ambiguous QA。
- 与当前 multi-hop citation 主线不完全一致。
- 自动评测更复杂。

### 7.3 随机网页 / PDF / 自己爬取资料

不建议第一版使用。

原因：

- 没有标准答案。
- 没有 supporting facts。
- citation 和 faithfulness 很难自动评测。

---

## 8. 数据文件建议存放方式

```text
data/
├── raw/
│   ├── hotpotqa/
│   │   ├── hotpot_train_v1.1.json
│   │   └── hotpot_dev_distractor_v1.json
│   ├── natural_questions/
│   ├── 2wikimultihopqa/
│   └── musique/
├── processed/
│   ├── hotpotqa/
│   │   ├── per_sample/
│   │   │   └── dev_samples.jsonl
│   │   ├── global/
│   │   │   ├── corpus.jsonl
│   │   │   ├── questions_dev.jsonl
│   │   │   └── title_to_doc_ids.json
│   │   └── debug/
│   └── mixed_router/
│       ├── simple_questions.jsonl
│       └── complex_questions.jsonl
└── indexes/
    ├── hotpotqa_per_sample/
    └── hotpotqa_global/
```

---

## 9. 最终推荐执行顺序

```text
1. HotpotQA distractor per-sample corpus
   目标：跑通 pipeline。

2. HotpotQA deduplicated global corpus
   目标：做正式 RAG 检索实验。

3. Natural Questions 小样本
   目标：补充 simple route。

4. 2WikiMultiHopQA
   目标：验证复杂路径泛化。

5. MuSiQue
   目标：复杂多跳压力测试。
```

一句话：

```text
Phase 1 是过渡和 debug。
Phase 2 是第一版主实验。
Natural Questions 补简单问题。
2WikiMultiHopQA 和 MuSiQue 做复杂问题扩展。
```

---

## 10. 参考来源

- HotpotQA official site: https://hotpotqa.github.io/
- HotpotQA GitHub: https://github.com/hotpotqa/hotpot
- Natural Questions paper page: https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/
- 2WikiMultiHopQA GitHub: https://github.com/Alab-NII/2wikimultihop
- MuSiQue paper: https://arxiv.org/abs/2108.00573
