"""Entry point: baseline -> production -> so sánh -> báo cáo Markdown."""

from __future__ import annotations

import json
import time
from pathlib import Path

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _metric_rows(naive: dict, production: dict) -> list[str]:
    rows: list[str] = []
    for metric in METRICS:
        baseline = float(naive["aggregate"].get(metric, 0.0))
        current = float(production["aggregate"].get(metric, 0.0))
        rows.append(
            f"| {metric} | {baseline:.4f} | {current:.4f} | "
            f"{current - baseline:+.4f} |"
        )
    return rows


def _one_line(value: object, limit: int = 700) -> str:
    text = " ".join(str(value).replace("|", "\\|").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def write_analysis(naive: dict, production: dict) -> None:
    """Sinh báo cáo có số liệu; người dùng chỉ cần điền tên và nhận xét nhóm."""
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    rows = _metric_rows(naive, production)
    deltas = {
        metric: float(production["aggregate"].get(metric, 0.0))
        - float(naive["aggregate"].get(metric, 0.0))
        for metric in METRICS
    }
    biggest = max(deltas, key=deltas.get)
    latency = production.get("latency_ms", {})
    latency_rows = [
        f"| {name} | {float(value):.3f} |" for name, value in latency.items()
    ] or ["| Chưa có | 0.000 |"]

    group_report = "\n".join(
        [
            "# Group Report — Lab 18: Production RAG",
            "",
            "**Nhóm:** [Điền tên nhóm]  ",
            "**Ngày:** [Điền ngày chạy cuối]",
            "",
            "## Thành viên & Phân công",
            "",
            "| Tên | Module | Hoàn thành | Tests pass |",
            "|---|---|---:|---:|",
            "| [Điền tên] | M1: Chunking/OCR | ☐ | /13 |",
            "| [Điền tên] | M2: Hybrid Search | ☐ | /5 |",
            "| [Điền tên] | M3: Reranking | ☐ | /5 |",
            "| [Điền tên] | M4: Evaluation | ☐ | /4 |",
            "| [Điền tên] | M5: Enrichment | ☐ | /10 |",
            "",
            "## Kiến trúc đã tích hợp",
            "",
            "PDF/OCR → hierarchical child/parent → enrichment tùy chọn → "
            "BM25 + BGE-M3 + RRF → cross-encoder → parent context → LLM → RAGAS.",
            "",
            "## Kết quả RAGAS",
            "",
            "| Metric | Naive | Production | Δ |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"**Evaluation backend:** {production.get('evaluation_backend', 'unknown')}",
            "",
            "## Latency breakdown",
            "",
            "| Công đoạn | ms |",
            "|---|---:|",
            *latency_rows,
            "",
            "## Key Findings",
            "",
            f"1. Biggest improvement theo số liệu: **{biggest}** "
            f"({deltas[biggest]:+.4f}).",
            "2. Biggest challenge: [Đối chiếu bottom-5 và điền nhận xét].",
            "3. Surprise finding: [Điền sau khi xem ablation/log].",
            "",
            "## Presentation Notes",
            "",
            "1. Trình bày bảng naive/production ở trên.",
            f"2. Biggest win: {biggest}; giải thích bằng thay đổi retrieval/generation.",
            "3. Case study: dùng failure #1 trong failure_analysis.md.",
            "4. Next step: sửa tầng gây worst metric phổ biến nhất rồi chạy lại.",
        ]
    )
    (analysis_dir / "group_report.md").write_text(group_report, encoding="utf-8")

    failures = production.get("failures", [])
    failure_lines = [
        "# Failure Analysis — Lab 18: Production RAG",
        "",
        "**Nhóm:** [Điền tên nhóm]  ",
        "**Thành viên:** [Điền tên và module]",
        "",
        "## RAGAS Scores",
        "",
        "| Metric | Naive | Production | Δ |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "## Bottom-5 Failures",
        "",
    ]
    for index, failure in enumerate(failures[:5], start=1):
        metric = failure.get("worst_metric", "unknown")
        route = (
            "Output sai → Context thiếu/nhiễu → sửa retrieval"
            if metric in {"context_precision", "context_recall"}
            else "Output sai → Context cần kiểm chứng → sửa prompt/generation"
        )
        failure_lines.extend(
            [
                f"### #{index}",
                "",
                f"- **Question:** {_one_line(failure.get('question', ''))}",
                f"- **Expected:** {_one_line(failure.get('ground_truth', ''))}",
                f"- **Got:** {_one_line(failure.get('answer', ''))}",
                f"- **Worst metric:** {metric} = {failure.get('score', 0):.4f}",
                f"- **Context mẫu:** {_one_line(failure.get('contexts', []))}",
                f"- **Error Tree:** {route}",
                f"- **Root cause:** {failure.get('diagnosis', '')}",
                f"- **Suggested fix:** {failure.get('suggested_fix', '')}",
                "",
            ]
        )
    if failures:
        first = failures[0]
        failure_lines.extend(
            [
                "## Case Study (cho presentation)",
                "",
                f"**Question:** {_one_line(first.get('question', ''))}",
                "",
                "1. Output đúng? → Không/điểm thấp.",
                f"2. Worst metric → {first.get('worst_metric', 'unknown')}.",
                f"3. Root cause → {first.get('diagnosis', '')}.",
                f"4. Fix → {first.get('suggested_fix', '')}.",
            ]
        )
    (analysis_dir / "failure_analysis.md").write_text(
        "\n".join(failure_lines), encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    print("[1/3] Running fair baseline...")
    from naive_baseline import main as run_baseline

    run_baseline()
    print("[2/3] Running production pipeline...")
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    for name in ("ragas_report.json", "naive_baseline_report.json"):
        source = Path(name)
        if source.exists():
            source.replace(report_dir / name)

    naive = json.loads(
        (report_dir / "naive_baseline_report.json").read_text(encoding="utf-8")
    )
    production = json.loads(
        (report_dir / "ragas_report.json").read_text(encoding="utf-8")
    )
    write_analysis(naive, production)
    print("[3/3] Comparison")
    for row in _metric_rows(naive, production):
        print(row)
    print(f"Total: {time.perf_counter() - started:.1f}s")
    print("Điền tên/nhận xét còn thiếu trong analysis/ rồi chạy check_lab.py.")


if __name__ == "__main__":
    main()
