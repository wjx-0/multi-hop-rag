# 中文说明：汇总 prediction JSONL 的答案、证据、调用成本等评估指标。
"""Evaluate prediction JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="outputs/predictions/standard_rag_dev_smoke.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.predictions))
    if not records:
        print("no records")
        return

    metric_keys = sorted({key for record in records for key in record.get("metrics", {})})
    print(f"records: {len(records)}")
    for key in metric_keys:
        values = [float(record.get("metrics", {}).get(key, 0.0)) for record in records]
        print(f"{key}: {sum(values) / len(values):.4f}")

    calls = [float(record.get("cost", {}).get("llm_calls", 0.0)) for record in records]
    print(f"avg_llm_calls: {sum(calls) / len(calls):.4f}")
    if has_cost_key(records, "rerank_calls"):
        rerank_calls = average_cost(records, "rerank_calls")
        print(f"avg_rerank_calls: {rerank_calls:.4f}")


def has_cost_key(records: list[dict], key: str) -> bool:
    return any(key in record.get("cost", {}) for record in records)


def average_cost(records: list[dict], key: str) -> float:
    values = [float(record.get("cost", {}).get(key, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
