"""Milvus vector store helpers for HotpotQA paragraph chunks."""

from __future__ import annotations

import json
from typing import Any

from src.data.schema import Document, RetrievedDoc

DEFAULT_OUTPUT_FIELDS = ["doc_id", "title", "text", "sentences_json", "metadata_json"]


def document_to_milvus_row(document: Document, embedding: list[float]) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "text": document.text,
        "sentences_json": json.dumps(document.sentences, ensure_ascii=False),
        "metadata_json": json.dumps(_compact_metadata(document.metadata), ensure_ascii=False),
        "embedding": embedding,
    }


def milvus_hit_to_retrieved_doc(
    hit: dict[str, Any],
    *,
    rank: int,
    retrieval_source: str = "dense",
) -> RetrievedDoc:
    entity = hit.get("entity", hit)
    score = hit.get("distance", hit.get("score", 0.0))
    sentences = _loads_json_list(entity.get("sentences_json", "[]"))
    metadata = _loads_json_dict(entity.get("metadata_json", "{}"))
    return RetrievedDoc(
        doc_id=entity.get("doc_id", hit.get("id", "")),
        title=entity.get("title", ""),
        text=entity.get("text", ""),
        sentences=sentences,
        metadata=metadata,
        score=float(score),
        rank=rank,
        retrieval_source=retrieval_source,
    )


class MilvusHotpotStore:
    def __init__(
        self,
        *,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = "hotpotqa_global_chunks",
        dimension: int = 1024,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        index_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_params = index_params or {"M": 16, "efConstruction": 200}
        self.search_params = search_params or {"ef": 64}
        self.client = self._make_client()

    def _make_client(self):
        try:
            from pymilvus import MilvusClient
        except ImportError as error:
            raise ImportError("Install pymilvus before using MilvusHotpotStore.") from error

        kwargs = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        return MilvusClient(**kwargs)

    def has_collection(self) -> bool:
        return bool(self.client.has_collection(self.collection_name))

    def drop_collection(self) -> None:
        if self.has_collection():
            self.client.drop_collection(self.collection_name)

    def create_collection(self, *, drop_existing: bool = False) -> None:
        if drop_existing:
            self.drop_collection()
        if self.has_collection():
            return

        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("sentences_json", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=self.index_params,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_documents(self, documents: list[Document], embeddings: list[list[float]]) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length.")
        rows = [
            document_to_milvus_row(document, embedding)
            for document, embedding in zip(documents, embeddings)
        ]
        if not rows:
            return 0
        self.client.insert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def load_collection(self) -> None:
        self.client.load_collection(self.collection_name)

    def flush(self) -> None:
        self.client.flush(self.collection_name)

    def search(self, query_embedding: list[float], *, top_k: int) -> list[RetrievedDoc]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            output_fields=DEFAULT_OUTPUT_FIELDS,
            search_params={"metric_type": self.metric_type, "params": self.search_params},
        )
        hits = results[0] if results else []
        return [
            milvus_hit_to_retrieved_doc(hit, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))


def _loads_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _loads_json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "source_locations"
    }
