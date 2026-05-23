"""Load HotpotQA raw JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.data.schema import HotpotQASample
from src.utils.io import read_json


def _convert_raw_sample(raw: dict) -> HotpotQASample:
    return HotpotQASample(
        id=raw["_id"],
        question=raw["question"],
        answer=raw["answer"],
        type=raw.get("type", ""),
        level=raw.get("level", ""),
        supporting_facts=list(raw.get("supporting_facts", [])),
        context=list(raw.get("context", [])),
    )


def iter_hotpotqa(path: str | Path, limit: int | None = None) -> Iterator[HotpotQASample]:
    data = read_json(path)
    for index, raw in enumerate(data):
        if limit is not None and index >= limit:
            break
        yield _convert_raw_sample(raw)


def load_hotpotqa(path: str | Path, limit: int | None = None) -> list[HotpotQASample]:
    return list(iter_hotpotqa(path, limit=limit))
