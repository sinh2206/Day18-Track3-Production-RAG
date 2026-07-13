"""M4: RAGAS, đánh giá ngoại tuyến và phân tích lỗi."""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from config import OPENAI_API_KEY, RAGAS_STRICT, TEST_SET_PATH

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or not data:
        raise ValueError("Test set phải là một danh sách không rỗng")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("question", "ground_truth")
        ):
            raise ValueError(f"Mẫu test #{index + 1} thiếu question/ground_truth")
    return data


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), re.UNICODE))


def _coverage(target: str, evidence: str) -> float:
    target_tokens = _tokens(target)
    return len(target_tokens & _tokens(evidence)) / max(len(target_tokens), 1)


def _heuristic_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question: list[EvalResult] = []
    for question, answer, context, ground_truth in zip(
        questions, answers, contexts, ground_truths, strict=True
    ):
        joined = " ".join(context)
        relevant = [_coverage(ground_truth, item) for item in context]
        per_question.append(
            EvalResult(
                question,
                answer,
                context,
                ground_truth,
                _coverage(answer, joined),
                max(_coverage(question, answer), _coverage(ground_truth, answer)),
                sum(score > 0.25 for score in relevant) / max(len(relevant), 1),
                _coverage(ground_truth, joined),
            )
        )
    return {
        **{
            metric: sum(getattr(item, metric) for item in per_question)
            / len(per_question)
            for metric in METRICS
        },
        "per_question": per_question,
        "evaluation_backend": "heuristic",
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Chạy RAGAS khi có API key; fallback có nhãn cho môi trường ngoại tuyến."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if lengths != {len(questions)} or not questions:
        raise ValueError("Bốn đầu vào đánh giá phải cùng độ dài và không rỗng")
    if not all(isinstance(items, list) for items in contexts):
        raise TypeError("Mỗi contexts phải là list[str]")
    if not OPENAI_API_KEY:
        return _heuristic_evaluation(
            questions, answers, contexts, ground_truths
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        frame = result.to_pandas()
        per_question = [
            EvalResult(
                questions[index],
                answers[index],
                contexts[index],
                ground_truths[index],
                *[
                    0.0
                    if math.isnan(float(row[metric]))
                    else float(row[metric])
                    for metric in METRICS
                ],
            )
            for index, (_, row) in enumerate(frame.iterrows())
        ]
        return {
            **{
                metric: sum(getattr(item, metric) for item in per_question)
                / len(per_question)
                for metric in METRICS
            },
            "per_question": per_question,
            "evaluation_backend": "ragas",
        }
    except Exception as exc:
        if RAGAS_STRICT:
            raise
        warnings.warn(f"RAGAS lỗi, chuyển sang heuristic: {exc}", stacklevel=2)
        return _heuristic_evaluation(
            questions, answers, contexts, ground_truths
        )


def _value(item: EvalResult | dict, key: str):
    return item.get(key) if isinstance(item, dict) else getattr(item, key)


def failure_analysis(
    eval_results: list[EvalResult] | list[dict], bottom_n: int = 10
) -> list[dict]:
    """Chọn câu tệ nhất và gắn chẩn đoán theo metric thấp nhất."""
    diagnosis = {
        "faithfulness": (
            "Câu trả lời có nội dung không được context hỗ trợ",
            "Siết prompt, giảm temperature và loại context mâu thuẫn",
        ),
        "answer_relevancy": (
            "Câu trả lời lệch trọng tâm câu hỏi",
            "Cải thiện prompt hoặc bước hiểu/viết lại truy vấn",
        ),
        "context_precision": (
            "Nhiều chunk truy xuất không liên quan",
            "Tăng chất lượng RRF, reranking hoặc lọc metadata",
        ),
        "context_recall": (
            "Context bỏ sót bằng chứng cần thiết",
            "Kiểm tra OCR, chunking, BM25 và dense retrieval",
        ),
    }
    ordered = sorted(
        eval_results,
        key=lambda item: sum(float(_value(item, metric)) for metric in METRICS),
    )[: max(bottom_n, 0)]
    failures: list[dict] = []
    for item in ordered:
        scores = {metric: float(_value(item, metric)) for metric in METRICS}
        worst_metric = min(scores, key=scores.get)
        reason, fix = diagnosis[worst_metric]
        failures.append(
            {
                "question": _value(item, "question"),
                "answer": _value(item, "answer"),
                "ground_truth": _value(item, "ground_truth"),
                "contexts": _value(item, "contexts"),
                "average_score": round(sum(scores.values()) / len(scores), 4),
                "worst_metric": worst_metric,
                "score": round(scores[worst_metric], 4),
                "diagnosis": reason,
                "suggested_fix": fix,
            }
        )
    return failures


def save_report(
    results: dict, failures: list[dict], path: str = "ragas_report.json"
) -> None:
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {metric: float(results.get(metric, 0.0)) for metric in METRICS},
        "num_questions": len(per_question),
        "evaluation_backend": results.get("evaluation_backend", "unknown"),
        "latency_ms": results.get("latency_ms", {}),
        "per_question": [
            asdict(item) if is_dataclass(item) else item for item in per_question
        ],
        "failures": failures,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Report saved to {target}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
