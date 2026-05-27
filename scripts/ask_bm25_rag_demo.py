# 中文说明：运行一个单问题 BM25 RAG demo，用于快速检查检索和回答链路。
"""Ask one question against a small HotpotQA BM25 demo corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_corpus import context_to_documents
from src.data.load_hotpotqa import iter_hotpotqa
from src.data.schema import Document
from src.retrieval.bm25 import BM25Retriever
from src.utils.llm_client import AliyunDashScopeClient, MockLLMClient
from src.utils.text import normalize_whitespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--input", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--corpus-limit", type=int, default=200, help="Number of HotpotQA samples used to build the demo corpus.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", choices=["mock", "aliyun"], default="mock")
    return parser.parse_args()


def build_demo_corpus(input_path: str, *, limit: int) -> list[Document]:
    documents: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for sample in iter_hotpotqa(input_path, limit=limit):
        for doc in context_to_documents(sample, corpus_type="demo_global"):
            key = (doc.title.lower().strip(), normalize_whitespace(doc.text).lower())
            if key in seen:
                continue
            seen.add(key)
            documents.append(doc)
    return documents


def main() -> None:
    args = parse_args()
    documents = build_demo_corpus(args.input, limit=args.corpus_limit)
    retriever = BM25Retriever(documents)
    retrieved_docs = retriever.retrieve(args.question, top_k=args.top_k)
    if args.llm == "aliyun":
        generation = AliyunDashScopeClient().answer_from_docs(args.question, retrieved_docs)
    else:
        generation = MockLLMClient().answer_from_docs(args.question, retrieved_docs)

    print("Question:")
    print(args.question)
    print()
    print("Answer:" if args.llm == "aliyun" else "Answer placeholder:")
    print(generation.answer)
    print()
    print("Top retrieved docs:")
    for doc in retrieved_docs:
        print(f'{doc.rank}. {doc.title} | score={doc.score:.4f} | doc_id={doc.doc_id}')
        print(f"   {doc.text[:260]}...")
    print()
    print("Cost:")
    print(generation.cost)


if __name__ == "__main__":
    main()
