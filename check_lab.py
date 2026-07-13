"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import sys
import subprocess


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        if path.endswith("ragas_report.json") and (
            data.get("evaluation_backend") != "ragas"
            or int(data.get("num_questions", 0)) < 20
            or len(data.get("failures", [])) < 5
        ):
            print(f"  ❌ {path} cần RAGAS thật, ít nhất 20 câu và bottom-5")
            return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "TODO" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest and return (passed, total)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, timeout=120,
        )
        lines = result.stdout.strip().split("\n")
        summary = lines[-1] if lines else ""
        # Parse "X passed, Y failed" or "X passed"
        passed = total = 0
        for part in summary.split(","):
            part = part.strip()
            if "passed" in part:
                passed = int(part.split()[0])
                total += passed
            if "failed" in part:
                total += int(part.split()[0])
        return passed, total
    except Exception as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py",
              "requirements.txt", "docker-compose.yml"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json(
            "reports/ragas_report.json",
            ["aggregate", "num_questions", "evaluation_backend", "failures"],
        ):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    if not check_file("analysis/failure_analysis.md"):
        errors += 1
    if not check_file("analysis/group_report.md"):
        errors += 1

    print("\n📚 Test set:")
    try:
        with open("test_set.json", encoding="utf-8") as f:
            test_set = json.load(f)
        valid = (
            isinstance(test_set, list)
            and len(test_set) >= 20
            and all("question" in item and "ground_truth" in item for item in test_set)
        )
        print(f"  {'✅' if valid else '❌'} {len(test_set)} câu hỏi")
        if not valid:
            errors += 1
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"  ❌ test_set.json — {exc}")
        errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections = [
            f for f in os.listdir(ref_dir)
            if f.startswith("reflection_")
            and f.endswith(".md")
            and f != "reflection_TEMPLATE.md"
        ]
    if reflections:
        for r in reflections:
            print(f"  ✅ {ref_dir}/{r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân trong {ref_dir}/")

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct >= 80 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
    else:
        print("  ⚠️  Không chạy được tests")

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)


if __name__ == "__main__":
    validate()
