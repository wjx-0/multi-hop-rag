"""Elasticsearch BM25 retriever and corpus indexing helpers."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.data.load_hotpotqa import iter_documents_jsonl
from src.data.schema import Document, RetrievedDoc

DEFAULT_ELASTICSEARCH_URL = "http://localhost:9200"
DEFAULT_ELASTICSEARCH_INDEX = "hotpotqa_global_bm25"
ELASTICSEARCH_RETRIEVAL_SOURCE = "elasticsearch_bm25"


class ElasticsearchBM25Retriever:
    """BM25 retrieval backed by an Elasticsearch text index."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_ELASTICSEARCH_URL,
        index_name: str = DEFAULT_ELASTICSEARCH_INDEX,
        client: Any | None = None,
        retrieval_source: str = ELASTICSEARCH_RETRIEVAL_SOURCE,
    ) -> None:
        self.url = url
        self.index_name = index_name
        self.client = client or make_elasticsearch_client(url)
        self.retrieval_source = retrieval_source

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDoc]:
        if top_k <= 0:
            return []

        response = self.client.search(
            index=self.index_name,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["title", "text"],
                }
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            elasticsearch_hit_to_retrieved_doc(
                hit,
                rank=rank,
                retrieval_source=self.retrieval_source,
            )
            for rank, hit in enumerate(hits, start=1)
        ]


def make_elasticsearch_client(url: str):
    module = _import_elasticsearch_module()
    return module.Elasticsearch(url)


def build_elasticsearch_bm25_index(
    *,
    corpus_path: str | Path,
    url: str = DEFAULT_ELASTICSEARCH_URL,
    index_name: str = DEFAULT_ELASTICSEARCH_INDEX,
    limit: int | None = None,
    drop_existing: bool = False,
    batch_size: int = 500,
    client: Any | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    es_client = client or make_elasticsearch_client(url)
    helpers = _import_elasticsearch_helpers()

    if drop_existing and es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=elasticsearch_index_body())

    success_count = 0
    for ok, _ in helpers.streaming_bulk(
        es_client,
        elasticsearch_bulk_actions(corpus_path, index_name=index_name, limit=limit),
        chunk_size=batch_size,
        raise_on_error=False,
    ):
        if ok:
            success_count += 1

    es_client.indices.refresh(index=index_name)
    return {
        "index_name": index_name,
        "inserted_count": success_count,
        "corpus_path": str(corpus_path),
    }


def elasticsearch_index_body() -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "sentences_json": {"type": "text", "index": False},
                "metadata_json": {"type": "text", "index": False},
            }
        },
    }


def elasticsearch_bulk_actions(
    corpus_path: str | Path,
    *,
    index_name: str,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    for document in iter_documents_jsonl(corpus_path, limit=limit):
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": document.doc_id,
            "_source": document_to_elasticsearch_source(document),
        }


def document_to_elasticsearch_source(document: Document) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "text": document.text,
        "sentences_json": json.dumps(document.sentences, ensure_ascii=False),
        "metadata_json": json.dumps(document.metadata, ensure_ascii=False),
    }


def elasticsearch_hit_to_retrieved_doc(
    hit: dict[str, Any],
    *,
    rank: int,
    retrieval_source: str = ELASTICSEARCH_RETRIEVAL_SOURCE,
) -> RetrievedDoc:
    source = hit.get("_source", {})
    return RetrievedDoc(
        doc_id=str(source.get("doc_id", hit.get("_id", ""))),
        title=str(source.get("title", "")),
        text=str(source.get("text", "")),
        sentences=_loads_json_list(source.get("sentences_json")),
        metadata=_loads_json_dict(source.get("metadata_json")),
        score=float(hit.get("_score") or 0.0),
        rank=rank,
        retrieval_source=retrieval_source,
    )


def _loads_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _loads_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _import_elasticsearch_module():
    try:
        return importlib.import_module("elasticsearch")
    except ImportError as error:
        raise ImportError(
            "The elasticsearch package is required for the Elasticsearch BM25 backend. "
            "Install it with `conda run -n qream-rag pip install -r requirements.txt`."
        ) from error


def _import_elasticsearch_helpers():
    try:
        return importlib.import_module("elasticsearch.helpers")
    except ImportError as error:
        _import_elasticsearch_module()
        raise ImportError("Unable to import elasticsearch.helpers.") from error
