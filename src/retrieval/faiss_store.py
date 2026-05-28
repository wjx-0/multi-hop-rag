"""FAISS dense vector store for HotpotQA paragraph chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.schema import Document, RetrievedDoc

DEFAULT_FAISS_INDEX_PATH = "data/indexes/hotpotqa_global/faiss_bge_m3.index"
DEFAULT_FAISS_DOCSTORE_PATH = "data/indexes/hotpotqa_global/faiss_bge_m3_docs.jsonl"
DEFAULT_FAISS_INDEX_TYPE = "hnsw"
DEFAULT_FAISS_HNSW_M = 32
DEFAULT_FAISS_EF_CONSTRUCTION = 200
DEFAULT_FAISS_EF_SEARCH = 128


def make_faiss_index(
    *,
    dimension: int,
    metric_type: str = "COSINE",
    index_type: str = DEFAULT_FAISS_INDEX_TYPE,
    hnsw_m: int = DEFAULT_FAISS_HNSW_M,
    ef_construction: int = DEFAULT_FAISS_EF_CONSTRUCTION,
):
    try:
        import faiss
    except ImportError as error:
        raise ImportError("Install faiss-cpu before using the FAISS dense backend.") from error

    metric = _faiss_metric(metric_type)
    normalized_index_type = index_type.lower()
    if normalized_index_type == "flat":
        if metric == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(dimension)
        if metric == faiss.METRIC_L2:
            return faiss.IndexFlatL2(dimension)
        raise ValueError(f"Unsupported FAISS metric type: {metric_type}")
    if normalized_index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, hnsw_m, metric)
        index.hnsw.efConstruction = ef_construction
        return index
    raise ValueError(f"Unsupported FAISS index type: {index_type}")


class FAISSHotpotStore:
    """Read-only FAISS store used by dense and hybrid retrieval diagnostics."""

    def __init__(
        self,
        *,
        index_path: str | Path = DEFAULT_FAISS_INDEX_PATH,
        docstore_path: str | Path = DEFAULT_FAISS_DOCSTORE_PATH,
        metric_type: str = "COSINE",
        ef_search: int = DEFAULT_FAISS_EF_SEARCH,
    ) -> None:
        self.index_path = Path(index_path)
        self.docstore_path = Path(docstore_path)
        self.metric_type = metric_type
        self.ef_search = ef_search
        self.index = self._load_index()
        self.documents = self._load_documents()
        self._validate_alignment()
        self._configure_search()

    def load_collection(self) -> None:
        """Keep the same small interface as MilvusHotpotStore."""

    def search(self, query_embedding: list[float], *, top_k: int) -> list[RetrievedDoc]:
        import numpy as np

        query_array = np.asarray([query_embedding], dtype="float32")
        scores, indexes = self.index.search(query_array, top_k)
        results: list[RetrievedDoc] = []
        for rank, (score, row_index) in enumerate(zip(scores[0], indexes[0]), start=1):
            if row_index < 0:
                continue
            document = self.documents[int(row_index)]
            metadata = dict(document.metadata)
            metadata["dense_backend"] = "faiss"
            metadata["faiss_row_id"] = int(row_index)
            results.append(
                RetrievedDoc(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    sentences=list(document.sentences),
                    metadata=metadata,
                    score=float(score),
                    rank=rank,
                    retrieval_source="dense",
                )
            )
        return results

    def count(self) -> int:
        return len(self.documents)

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        try:
            import faiss
        except ImportError as error:
            raise ImportError("Install faiss-cpu before using the FAISS dense backend.") from error
        return faiss.read_index(str(self.index_path))

    def _load_documents(self) -> list[Document]:
        if not self.docstore_path.exists():
            raise FileNotFoundError(f"Missing FAISS docstore: {self.docstore_path}")
        documents: list[Document] = []
        with self.docstore_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(Document.from_dict(json.loads(line)))
        return documents

    def _validate_alignment(self) -> None:
        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and docstore size mismatch: "
                f"index has {self.index.ntotal}, docstore has {len(self.documents)}."
            )

    def _configure_search(self) -> None:
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.ef_search


def write_faiss_docstore_record(f, document: Document) -> None:
    record = document.to_dict()
    record["metadata"] = _compact_metadata(document.metadata)
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _faiss_metric(metric_type: str) -> int:
    import faiss

    normalized = metric_type.upper()
    if normalized in {"COSINE", "IP", "INNER_PRODUCT"}:
        return faiss.METRIC_INNER_PRODUCT
    if normalized in {"L2", "EUCLIDEAN"}:
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported FAISS metric type: {metric_type}")


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "source_locations"
    }
