"""BM25 retriever backed by the rank_bm25 package."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.data.schema import Document, RetrievedDoc
from src.utils.text import simple_tokenize

BM25_CACHE_VERSION = 1


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5, # 一个词在文档中重复出现多次会加分，但加分效果会递减，k1控制这个递减的程度，k1越大，重复出现的词对得分的贡献越大。
        b: float = 0.75, # 长文档可能包含更多的相关信息，但也可能包含更多的噪音，b控制文档长度对得分的影响，b=0表示不考虑文档长度，b=1表示完全考虑文档长度。
        retrieval_source: str = "bm25",
        include_title: bool = True, # 是否把标题也作为检索的一部分，通常标题包含了文档的核心内容，加入标题可以提高检索的相关性。
    ) -> None:
        self.documents = documents
        self.k1 = k1 
        self.b = b
        self.retrieval_source = retrieval_source
        self.include_title = include_title
        self.doc_tokens = [
            simple_tokenize(f"{doc.title} {doc.text}" if include_title else doc.text)
            for doc in documents
        ]
        self.index = BM25Okapi(self.doc_tokens, k1=k1, b=b) if self.doc_tokens else None # 创建BN25索引

    def score(self, query: str) -> list[float]:
        query_terms = simple_tokenize(query)
        if self.index is None:
            return [] 
        return [float(score) for score in self.index.get_scores(query_terms)]

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDoc]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[RetrievedDoc] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            retrieved_doc = RetrievedDoc.from_document(
                self.documents[index],
                score=score,
                rank=rank,
                retrieval_source=self.retrieval_source,
            )
            results.append(retrieved_doc)

        return results
    ''' 得分返回的结果
    rank = 1, (index, score) = (3, 12.5)
    rank = 2, (index, score) = (0, 8.2)
    rank = 3, (index, score) = (5, 3.7)
    '''
    '''
    [
    RetrievedDoc(
        doc_id="sample_id::5",
        title="Arthur's Magazine",
        text="Arthur's Magazine was an American literary periodical. It was founded in 1844.",
        sentences=[
            "Arthur's Magazine was an American literary periodical.",
            "It was founded in 1844."
        ],
        metadata={
            "dataset": "hotpotqa",
            "corpus_type": "per_sample",
            "source_question_id": "sample_id",
            "paragraph_index": 5
        },
        score=12.53,
        rank=1,
        retrieval_source="bm25"
    ),
    RetrievedDoc(
        doc_id="sample_id::2",
        title="First for Women",
        text="First for Women is a woman's magazine. The magazine was started in 1989.",
        sentences=[
            "First for Women is a woman's magazine.",
            "The magazine was started in 1989."
        ],
        metadata={
            "dataset": "hotpotqa",
            "corpus_type": "per_sample",
            "source_question_id": "sample_id",
            "paragraph_index": 2
        },
        score=10.14,
        rank=2,
        retrieval_source="bm25"
    ),
    RetrievedDoc(
        doc_id="sample_id::7",
        title="Some Distractor",
        text="...",
        sentences=["..."],
        metadata={...},
        score=4.22,
        rank=3,
        retrieval_source="bm25"
    )
    ]
'''


def corpus_fingerprint(path: str | Path) -> dict[str, Any]:
    corpus_path = Path(path)
    stat = corpus_path.stat()
    return {
        "path": str(corpus_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def save_bm25_cache(
    retriever: BM25Retriever,
    cache_path: str | Path,
    *,
    corpus_path: str | Path,
) -> None:
    output_path = Path(cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BM25_CACHE_VERSION,
        "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "retriever": retriever,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25_cache(
    cache_path: str | Path,
    *,
    corpus_path: str | Path,
) -> BM25Retriever | None:
    input_path = Path(cache_path)
    if not input_path.exists():
        return None

    try:
        with input_path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != BM25_CACHE_VERSION:
        return None
    if payload.get("corpus_fingerprint") != corpus_fingerprint(corpus_path):
        return None

    retriever = payload.get("retriever")
    if not isinstance(retriever, BM25Retriever):
        return None
    return retriever
