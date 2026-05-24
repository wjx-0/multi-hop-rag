"""Load HotpotQA raw JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.data.schema import Document, HotpotQASample
from src.utils.io import read_json, read_jsonl


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


def iter_documents_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Document]:
    for index, record in enumerate(read_jsonl(path)):
        if limit is not None and index >= limit:
            break
        yield Document.from_dict(record)


def load_documents_jsonl(path: str | Path, limit: int | None = None) -> list[Document]:
    return list(iter_documents_jsonl(path, limit=limit))


def iter_processed_hotpotqa_questions(
    path: str | Path,
    limit: int | None = None,
) -> Iterator[HotpotQASample]:
    for index, record in enumerate(read_jsonl(path)):
        if limit is not None and index >= limit:
            break
        yield HotpotQASample(
            id=record["id"],
            question=record["question"],
            answer=record["answer"],
            type=record.get("type", ""),
            level=record.get("level", ""),
            supporting_facts=list(record.get("supporting_facts", [])),
            context=[],
        )


def load_processed_hotpotqa_questions(
    path: str | Path,
    limit: int | None = None,
) -> list[HotpotQASample]:
    return list(iter_processed_hotpotqa_questions(path, limit=limit))
