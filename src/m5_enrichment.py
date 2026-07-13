"""M5: làm giàu chunk trước khi embedding, luôn bảo toàn nguyên văn."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass

from config import GENERATION_MODEL, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


_client = None


def _chat(system: str, user: str, max_tokens: int) -> str:
    global _client
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI

        _client = _client or OpenAI(api_key=OPENAI_API_KEY)
        response = _client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        warnings.warn(f"Enrichment LLM lỗi, dùng fallback: {exc}", stacklevel=2)
        return ""


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if item.strip()
    ]
    return " ".join(sentences[:count])


def summarize_chunk(text: str) -> str:
    """Tóm tắt 2-3 câu; fallback extractive khi không có API."""
    fallback = _first_sentences(text)
    summary = _chat(
        "Tóm tắt đoạn văn trong 2-3 câu tiếng Việt. Giữ nguyên số liệu và điều kiện.",
        text,
        180,
    )
    return summary if summary and len(summary) <= max(len(text) * 2, 1) else fallback


def generate_hypothesis_questions(
    text: str, n_questions: int = 3
) -> list[str]:
    """Sinh các câu hỏi mà chunk có thể trả lời."""
    if n_questions <= 0:
        return []
    output = _chat(
        f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
        "Mỗi dòng một câu hỏi, không giải thích.",
        text,
        220,
    )
    if output:
        questions = [
            re.sub(r"^\s*\d+[.)-]?\s*", "", line).strip()
            for line in output.splitlines()
            if line.strip()
        ]
        return [
            question if question.endswith("?") else f"{question}?"
            for question in questions[:n_questions]
        ]
    subject = _first_sentences(text, 1).rstrip(".!?")
    fallbacks = [
        "Đoạn văn quy định nội dung chính nào?",
        f"Thông tin nào được nêu về {subject[:80]}?",
        "Điều kiện hoặc số liệu quan trọng trong đoạn văn là gì?",
    ]
    return fallbacks[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Thêm một câu định vị và luôn chứa nguyên văn."""
    context = _chat(
        "Viết đúng một câu ngắn nêu tài liệu/vị trí và chủ đề của đoạn văn.",
        f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}",
        100,
    )
    context = context or (
        f"Đoạn trích từ {document_title}." if document_title else "Đoạn trích tài liệu."
    )
    return f"{context}\n\n{text}"


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category = "general"
    for candidate, keywords in {
        "legal": ("nghị định", "điều ", "pháp luật"),
        "finance": ("tài chính", "doanh thu", "lợi nhuận"),
        "hr": ("nhân viên", "nghỉ phép", "thử việc"),
        "it": ("mật khẩu", "hệ thống", "vpn"),
    }.items():
        if any(keyword in lowered for keyword in keywords):
            category = candidate
            break
    entities = list(
        dict.fromkeys(
            re.findall(
                r"\b(?:[A-ZĐ][\wÀ-ỹ-]+(?:\s+[A-ZĐ][\wÀ-ỹ-]+){0,4})\b", text
            )
        )
    )[:10]
    topic = " ".join(_first_sentences(text, 1).split()[:12])
    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": "vi" if re.search(r"[À-ỹ]", text) else "unknown",
    }


def extract_metadata(text: str) -> dict:
    """Trích metadata JSON; fallback xác định bằng quy tắc."""
    output = _chat(
        "Trả về duy nhất JSON hợp lệ với các khóa topic, entities, category, "
        "language. entities là mảng chuỗi.",
        text,
        180,
    )
    if output:
        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", output.strip()).strip()
            metadata = json.loads(cleaned)
            if isinstance(metadata, dict):
                return metadata
        except json.JSONDecodeError:
            pass
    return _fallback_metadata(text)


def enrich_chunks(
    chunks: list[dict], methods: list[str] | None = None
) -> list[EnrichedChunk]:
    """Áp dụng các kỹ thuật được chọn cho mọi chunk."""
    methods = methods or ["contextual", "hyqa", "metadata"]
    allowed = {"summary", "hyqa", "contextual", "metadata", "full"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Enrichment method không hợp lệ: {sorted(unknown)}")
    full = "full" in methods
    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        text = chunk["text"]
        original_metadata = dict(chunk.get("metadata", {}))
        summary = summarize_chunk(text) if full or "summary" in methods else ""
        questions = (
            generate_hypothesis_questions(text)
            if full or "hyqa" in methods
            else []
        )
        contextual = (
            contextual_prepend(text, str(original_metadata.get("source", "")))
            if full or "contextual" in methods
            else text
        )
        automatic = extract_metadata(text) if full or "metadata" in methods else {}
        retrieval_parts = [contextual]
        if summary:
            retrieval_parts.append(f"Tóm tắt: {summary}")
        if questions:
            retrieval_parts.append("Câu hỏi liên quan: " + " | ".join(questions))
        enriched.append(
            EnrichedChunk(
                text,
                "\n\n".join(retrieval_parts),
                summary,
                questions,
                {**automatic, **original_metadata},
                "+".join(methods),
            )
        )
    return enriched


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print(enrich_chunks([{"text": sample, "metadata": {"source": "demo"}}]))
