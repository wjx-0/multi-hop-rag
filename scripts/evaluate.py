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


if __name__ == "__main__":
    main()
