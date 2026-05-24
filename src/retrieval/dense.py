"""Dense retrieval with sentence-transformers embeddings and Milvus."""

from __future__ import annotations

from typing import Protocol

from src.data.schema import RetrievedDoc


class Embedder(Protocol):
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class DenseStore(Protocol):
    def search(self, query_embedding: list[float], *, top_k: int) -> list[RetrievedDoc]:
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("Install sentence-transformers before using BGE embeddings.") from error

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        return SentenceTransformer(self.model_name, **kwargs)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class DenseRetriever:
    def __init__(self, *, embedder: Embedder, store: DenseStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDoc]:
        query_embedding = self.embedder.encode_texts([query])[0]
        results = self.store.search(query_embedding, top_k=top_k)
        for rank, result in enumerate(results, start=1):
            result.rank = rank
            result.retrieval_source = "dense"
        return results


def document_embedding_text(title: str, text: str) -> str:
    return f"{title}\n{text}".strip()
