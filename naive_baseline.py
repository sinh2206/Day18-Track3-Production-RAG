"""Baseline công bằng: paragraph chunking + dense-only + cùng generator/M4."""

from __future__ import annotations

import time

from config import NAIVE_COLLECTION
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.pipeline import generate_answer


def main() -> dict:
    started = time.perf_counter()
    chunks: list[dict] = []
    for document in load_documents(strict=True):
        for chunk in chunk_basic(document["text"], metadata=document["metadata"]):
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = (
                f"{metadata.get('doc_id', 'doc')}_"
                f"{metadata.get('page', 0)}_{metadata['chunk_index']}"
            )
            chunks.append({"text": chunk.text, "metadata": metadata})
    if not chunks:
        raise RuntimeError("Baseline không có chunk để lập chỉ mục")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    test_set = load_test_set()
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    for item in test_set:
        hits = search.search(item["question"], 3, collection=NAIVE_COLLECTION)
        retrieved = [hit.text for hit in hits]
        questions.append(item["question"])
        answers.append(generate_answer(item["question"], retrieved))
        contexts.append(retrieved)
        ground_truths.append(item["ground_truth"])

    results = evaluate_ragas(questions, answers, contexts, ground_truths)
    results["latency_ms"] = {
        "total": round((time.perf_counter() - started) * 1000, 3)
    }
    save_report(
        results,
        failure_analysis(results["per_question"], bottom_n=5),
        "naive_baseline_report.json",
    )
    return results


if __name__ == "__main__":
    main()
