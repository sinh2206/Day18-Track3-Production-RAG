"""M3: cross-encoder reranking và đo độ trễ."""

from __future__ import annotations

import numbers
import re
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean

from config import RERANK_TOP_K, RERANKER_MODEL


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _lexical_score(query: str, text: str) -> float:
    tokens = lambda value: set(re.findall(r"\w+", value.lower(), re.UNICODE))
    query_tokens, text_tokens = tokens(query), tokens(text)
    return 2 * len(query_tokens & text_tokens) / max(
        len(query_tokens) + len(text_tokens), 1
    )


@lru_cache(maxsize=2)
def _load_cached(model_name: str):
    try:
        from FlagEmbedding import FlagReranker

        return "flag", FlagReranker(model_name, use_fp16=False)
    except Exception as flag_error:
        try:
            from sentence_transformers import CrossEncoder

            return "cross_encoder", CrossEncoder(model_name)
        except Exception as cross_error:
            warnings.warn(
                "Không tải được reranker; dùng lexical fallback. "
                f"FlagEmbedding: {flag_error}; CrossEncoder: {cross_error}",
                stacklevel=2,
            )
            return "lexical", None


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        self._backend = ""

    def _load_model(self):
        if not self._backend:
            self._backend, self._model = _load_cached(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        pairs = [(query, document["text"]) for document in documents]
        if self._backend == "flag":
            raw_scores = model.compute_score(pairs, normalize=True)
        elif self._backend == "cross_encoder":
            raw_scores = model.predict(pairs)
        else:
            raw_scores = [_lexical_score(*pair) for pair in pairs]
        if isinstance(raw_scores, numbers.Real):
            raw_scores = [raw_scores]
        ranked = sorted(
            zip(raw_scores, documents, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )[:top_k]
        return [
            RerankResult(
                document["text"],
                float(document.get("score", 0.0)),
                float(score),
                dict(document.get("metadata", {})),
                rank,
            )
            for rank, (score, document) in enumerate(ranked, start=1)
        ]


class FlashrankReranker:
    """Reranker nhẹ; tự lùi về cross-encoder khi flashrank không khả dụng."""

    def __init__(self):
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            self._model = self._model or Ranker()
            passages = [
                {"id": index, "text": document["text"]}
                for index, document in enumerate(documents)
            ]
            output = self._model.rerank(
                RerankRequest(query=query, passages=passages)
            )[:top_k]
            return [
                RerankResult(
                    documents[int(item["id"])]["text"],
                    float(documents[int(item["id"])].get("score", 0.0)),
                    float(item["score"]),
                    dict(documents[int(item["id"])].get("metadata", {})),
                    rank,
                )
                for rank, item in enumerate(output, start=1)
            ]
        except (ImportError, RuntimeError, ValueError, KeyError):
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    if n_runs <= 0:
        raise ValueError("n_runs phải lớn hơn 0")
    reranker.rerank(query, documents)
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    p95_index = min(round(0.95 * (len(ordered) - 1)), len(ordered) - 1)
    return {
        "avg_ms": round(mean(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


if __name__ == "__main__":
    sample_query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    sample_documents = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7},
    ]
    print(CrossEncoderReranker().rerank(sample_query, sample_documents))
