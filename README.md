# Lab 18: Production RAG Pipeline

Pipeline RAG tiếng Việt hoàn chỉnh:

```text
PDF/Markdown/TXT
  -> trích xuất văn bản hoặc OCR
  -> hierarchical chunking (child retrieve, parent context)
  -> enrichment tùy chọn
  -> BM25 + BGE-M3/Qdrant + RRF
  -> BGE reranker
  -> LLM generation
  -> RAGAS + failure analysis + latency report
```

## Thành phần

- `src/m1_chunking.py`: đọc PDF theo trang, OCR fallback, basic/semantic/
  hierarchical/structure-aware chunking.
- `src/m2_search.py`: tách từ tiếng Việt, BM25, dense retrieval và RRF.
- `src/m3_rerank.py`: FlagEmbedding/CrossEncoder, Flashrank và benchmark.
- `src/m4_eval.py`: RAGAS, heuristic offline có gắn nhãn, bottom-N diagnosis.
- `src/m5_enrichment.py`: summary, HyQA, contextual prepend, auto metadata.
- `src/pipeline.py`: tích hợp M1-M5, hydrate parent và sinh câu trả lời.
- `naive_baseline.py`: paragraph + dense-only, dùng cùng generator/evaluator.
- `main.py`: chạy hai pipeline, so sánh và sinh báo cáo trong `reports/`,
  `analysis/`.
- `test_set.json`: 20 câu hỏi có ground truth từ Nghị định 13/2023/NĐ-CP.

Nguồn đối chiếu test set: [toàn văn do Cổng Thông tin điện tử Chính phủ công
bố](https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-13-2023-nd-cp-bao-ve-du-lieu-ca-nhan-119230516104357809.htm).

## Chuẩn bị

Yêu cầu Python 3.10+, Docker và Tesseract OCR có gói ngôn ngữ Việt (`vie`).
PDF trong `data/` là PDF ảnh nên Tesseract là bắt buộc nếu chưa có bản Markdown
hoặc TXT cùng tên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d
```

Điền `OPENAI_API_KEY` trong `.env` để dùng LLM và RAGAS thật. Đặt
`RAGAS_STRICT=true` để không chấp nhận evaluator fallback trong lần chạy chấm
điểm. Bật `ENABLE_ENRICHMENT=true` khi muốn chạy phần bonus; mặc định tắt để
tránh phát sinh nhiều API call khi kiểm tra luồng chính.

Nếu Tesseract không nằm trong `PATH`, điền đường dẫn executable vào
`TESSERACT_CMD`. `OCR_LANG=vie+eng` yêu cầu cả hai language pack tương ứng.

## Thứ tự người học tự chạy

```powershell
python -m pytest tests/test_m1.py -v
python -m pytest tests/test_m2.py -v
python -m pytest tests/test_m3.py -v
python -m pytest tests/test_m4.py -v
python -m pytest tests/test_m5.py -v
ruff check src/
rg -n "TODO" src
python main.py
python check_lab.py
```

`main.py` chạy baseline rồi production, chuyển hai JSON vào `reports/`, tự điền
số liệu và bottom-5 vào `analysis/group_report.md` và
`analysis/failure_analysis.md`. Sau đó người học cần điền tên nhóm, nhận xét và
tạo `analysis/reflections/reflection_[Tên].md` từ template.

## Đầu ra

```text
reports/
  naive_baseline_report.json
  ragas_report.json
analysis/
  group_report.md
  failure_analysis.md
  reflections/reflection_[Tên].md
```

Mục tiêu rubric: pipeline exit code 0, có RAGAS thật, ít nhất một metric đạt
0,75, bottom-5 có diagnosis/fix/Error Tree; bonus cho faithfulness từ 0,85,
enrichment và latency breakdown.

## Lưu ý đánh giá

- `evaluation_backend` trong JSON phải là `ragas` cho lần nộp chính thức.
- Baseline và production dùng cùng test set, generator và evaluator để delta có
  ý nghĩa.
- Không commit `.env`; hai báo cáo JSON không còn bị `.gitignore` loại bỏ.
- Chưa có lệnh nào được chạy trong lần hoàn thiện mã này; người học chịu trách
  nhiệm chạy, kiểm tra số liệu và điền phần thông tin cá nhân.
