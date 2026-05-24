"""Build paragraph-level corpora from HotpotQA samples."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
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


def normalize_global_title(title: str) -> str:
    return normalize_whitespace(title).lower()


def normalize_global_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def make_global_dedup_key(title: str, text: str) -> tuple[str, str]:
    return (normalize_global_title(title), normalize_global_text(text))


def make_global_doc_id(title: str, text: str) -> str:
    normalized_title, normalized_text = make_global_dedup_key(title, text)
    digest = hashlib.sha1(f"{normalized_title}\n{normalized_text}".encode("utf-8")).hexdigest()[:16]
    return f"hotpotqa::global::{digest}"


def make_global_metadata_dedup_key(title: str, text: str) -> str:
    normalized_title, normalized_text = make_global_dedup_key(title, text)
    digest = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()[:16]
    return f"{normalized_title}::{digest}"


def dev_question_record(sample: HotpotQASample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "question": sample.question,
        "answer": sample.answer,
        "type": sample.type,
        "level": sample.level,
        "supporting_facts": sample.supporting_facts,
    }


def build_global_deduplicated_corpus(
    *,
    train_samples: Iterable[HotpotQASample],
    dev_samples: Iterable[HotpotQASample],
) -> tuple[list[Document], list[dict[str, Any]], dict[str, list[str]]]:
    """Build HotpotQA global paragraph corpus and dev question records.

    Gold answers/supporting facts are only copied into dev question records for
    evaluation; they are not used for document deduplication.
    """

    dev_sample_list = list(dev_samples)
    builders: dict[tuple[str, str], dict[str, Any]] = {}
    for split, samples in (("train", train_samples), ("dev", dev_sample_list)):
        for sample in samples:
            for paragraph_index, item in enumerate(sample.context):
                document_parts = _global_document_parts(sample, item, paragraph_index, split)
                if document_parts is None:
                    continue

                title, text, sentences, location = document_parts
                dedup_key = make_global_dedup_key(title, text)
                if dedup_key not in builders:
                    builders[dedup_key] = {
                        "document": Document(
                            doc_id=make_global_doc_id(title, text),
                            title=title,
                            text=text,
                            sentences=sentences,
                            metadata={
                                "dataset": "hotpotqa",
                                "corpus_type": "global_deduplicated",
                                "dedup_key": make_global_metadata_dedup_key(title, text),
                            },
                        ),
                        "source_question_ids": set(),
                        "source_locations": [],
                    }

                builders[dedup_key]["source_question_ids"].add(sample.id)
                builders[dedup_key]["source_locations"].append(location)

    documents: list[Document] = []
    for builder in builders.values():
        document = builder["document"]
        document.metadata["source_question_ids"] = sorted(builder["source_question_ids"])
        document.metadata["source_locations"] = builder["source_locations"]
        documents.append(document)

    documents.sort(key=lambda doc: doc.doc_id)
    title_to_doc_ids = build_title_to_doc_ids(documents)
    dev_questions = [dev_question_record(sample) for sample in dev_sample_list]
    return documents, dev_questions, title_to_doc_ids


def build_title_to_doc_ids(documents: Iterable[Document]) -> dict[str, list[str]]:
    title_to_doc_ids: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        title_to_doc_ids[normalize_global_title(document.title)].append(document.doc_id)
    return {
        title: sorted(doc_ids)
        for title, doc_ids in sorted(title_to_doc_ids.items())
    }


def _global_document_parts(
    sample: HotpotQASample,
    item: Any,
    paragraph_index: int,
    split: str,
) -> tuple[str, str, list[str], dict[str, Any]] | None:
    if not isinstance(item, list) or len(item) != 2:
        return None

    title, sentences = item
    if not isinstance(title, str) or not isinstance(sentences, list):
        return None

    clean_title = normalize_whitespace(title)
    clean_sentences = [normalize_whitespace(str(sentence)) for sentence in sentences]
    text = normalize_whitespace(" ".join(clean_sentences))
    if not clean_title or not text:
        return None

    location = {
        "split": split,
        "question_id": sample.id,
        "paragraph_index": paragraph_index,
    }
    return clean_title, text, clean_sentences, location
