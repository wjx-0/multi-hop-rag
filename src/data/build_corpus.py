"""Build paragraph-level corpora from HotpotQA samples."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from src.data.schema import Document, HotpotQASample
from src.utils.text import normalize_whitespace

''' Example input sample.context:
    [
    ["Arthur's Magazine", ["sentence 0", "sentence 1"]],
    ["First for Women", ["sentence 0", "sentence 1"]]
]'''
def context_to_documents(
    sample: HotpotQASample,
    *,
    corpus_type: str = "per_sample",
) -> list[Document]:
    documents: list[Document] = []
    for paragraph_index, item in enumerate(sample.context):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        clean_sentences = [str(sentence) for sentence in sentences]
        text = normalize_whitespace(" ".join(clean_sentences))
        documents.append(
            Document(
                doc_id=f"{sample.id}::{paragraph_index}",
                title=title,
                text=text,
                sentences=clean_sentences,
                metadata={
                    "dataset": "hotpotqa",
                    "corpus_type": corpus_type,
                    "source_question_id": sample.id,
                    "paragraph_index": paragraph_index,
                },
            )
        )
    return documents


def validate_supporting_facts(
    sample: HotpotQASample,
    documents: list[Document],
) -> dict[str, Any]:
    title_to_docs: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        title_to_docs[doc.title].append(doc)

    invalid: list[dict[str, Any]] = []
    valid_count = 0
    for fact in sample.supporting_facts:
        if not isinstance(fact, list) or len(fact) != 2:
            invalid.append({"fact": fact, "reason": "bad_fact_shape"})
            continue
        title, sent_id = fact
        if title not in title_to_docs:
            invalid.append({"title": title, "sent_id": sent_id, "reason": "title_missing"})
            continue
        if not isinstance(sent_id, int):
            invalid.append({"title": title, "sent_id": sent_id, "reason": "sent_id_not_int"})
            continue
        if not any(0 <= sent_id < len(doc.sentences) for doc in title_to_docs[title]):
            invalid.append({"title": title, "sent_id": sent_id, "reason": "sent_id_out_of_range"})
            continue
        valid_count += 1

    return {
        "valid_supporting_facts": valid_count,
        "invalid_supporting_facts": len(invalid),
        "invalid_details": invalid,
    }


def processed_sample_record(sample: HotpotQASample) -> dict[str, Any]:
    documents = context_to_documents(sample)
    validation = validate_supporting_facts(sample, documents)
    return {
        "id": sample.id,
        "question": sample.question,
        "answer": sample.answer,
        "type": sample.type,
        "level": sample.level,
        "supporting_facts": sample.supporting_facts,
        "documents": [doc.to_dict() for doc in documents],
        "validation": validation,
    }


def make_global_doc_id(title: str, text: str) -> str:
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"hotpotqa::global::{digest}"
