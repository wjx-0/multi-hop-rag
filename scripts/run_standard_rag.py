"""Run the Phase 1 Standard RAG smoke baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_hotpotqa import iter_hotpotqa
from src.pipeline.standard_rag import StandardRAGPipeline
from src.utils.io import write_jsonl
from src.utils.llm_client import AliyunDashScopeClient, MockLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", default="outputs/predictions/standard_rag_dev_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", choices=["mock", "aliyun"], default="mock")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_client = AliyunDashScopeClient() if args.llm == "aliyun" else MockLLMClient()
    pipeline = StandardRAGPipeline(top_k=args.top_k, llm_client=llm_client)
    results = (
        pipeline.run(sample).to_dict()
        for sample in iter_hotpotqa(args.input, limit=args.limit)
    )
    count = write_jsonl(results, args.output)
    print(f"wrote {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
