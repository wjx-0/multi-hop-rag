"""Prepare HotpotQA per-sample JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import write_per_sample_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", default="data/processed/hotpotqa/per_sample/dev_samples.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_per_sample_jsonl(args.input, args.output, limit=args.limit)
    print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
