"""M2: BM25 tiếng Việt, dense retrieval và Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Chuẩn hóa NFC và tách từ; có fallback không phụ thuộc underthesea."""
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        return ""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")
    except (ImportError, RuntimeError):
        return " ".join(re.findall(r"\w+", text.lower(), re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Tạo chỉ mục BM25 từ retrieval text."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split() for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if not self.bm25 or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).lower().split())
        indices = sorted(
            range(len(scores)), key=lambda index: float(scores[index]), reverse=True
        )[:top_k]
        return [
            SearchResult(
                self.documents[index]["text"],
                float(scores[index]),
                dict(self.documents[index].get("metadata", {})),
                "bm25",
            )
            for index in indices
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(
        self, chunks: list[dict], collection: str = COLLECTION_NAME
    ) -> None:
        """Encode theo batch và ghi payload vào Qdrant."""
        if not chunks:
            raise ValueError("Không có chunk để lập chỉ mục dense")
        from qdrant_client.models import Distance, PointStruct, VectorParams

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        if len(vectors[0]) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding có {len(vectors[0])} chiều, cấu hình là {EMBEDDING_DIM}"
            )
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        vector = self._get_encoder().encode(
            query, normalize_embeddings=True
        ).tolist()
        hits = self.client.search(
            collection_name=collection, query_vector=vector, limit=top_k
        )
        results: list[SearchResult] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text, float(hit.score), payload, "dense"))
        return results


def _result_key(result: SearchResult) -> str:
    chunk_id = result.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return hashlib.sha1(result.text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Hợp nhất các bảng hạng mà không trộn score khác thang đo."""
    fused: dict[str, tuple[float, SearchResult]] = {}
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            key = _result_key(result)
            score, representative = fused.get(key, (0.0, result))
            fused[key] = (score + 1.0 / (k + rank), representative)
    ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)[:top_k]
    return [
        SearchResult(result.text, score, dict(result.metadata), "hybrid")
        for score, result in ranked
    ]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()
        self.parent_store: dict[str, str] = {}
        self.build_latency_ms: dict[str, float] = {}

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(
        self, query: str, top_k: int = HYBRID_TOP_K
    ) -> list[SearchResult]:
        return reciprocal_rank_fusion(
            [
                self.bm25.search(query, BM25_TOP_K),
                self.dense.search(query, DENSE_TOP_K),
            ],
            top_k=top_k,
        )


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original: {sample}\nSegmented: {segment_vietnamese(sample)}")
