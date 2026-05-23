"""Preprocess HotpotQA raw files into project JSONL records."""

from __future__ import annotations

from pathlib import Path

from src.data.build_corpus import processed_sample_record
from src.data.load_hotpotqa import iter_hotpotqa
from src.utils.io import write_jsonl


def write_per_sample_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> int:
    records = (processed_sample_record(sample) for sample in iter_hotpotqa(input_path, limit=limit))
    return write_jsonl(records, output_path)
