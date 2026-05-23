"""Text helpers shared by retrieval and evaluation."""

from __future__ import annotations

import re
import string


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def simple_tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def normalize_answer(text: str) -> str:
    """HotpotQA-style answer normalization."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return value.translate(table)

    text = text.lower()
    text = remove_punc(text)
    text = remove_articles(text)
    return normalize_whitespace(text)
