"""M1: nạp tài liệu và các chiến lược chunking nâng cao."""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_MODEL,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _read_pdf(path: Path, allow_ocr: bool = True) -> list[dict]:
    """Đọc từng trang; OCR các trang ảnh nếu pytesseract khả dụng."""
    from pypdf import PdfReader

    cache_path = path.parent / ".ocr_cache" / f"{path.stem}.json"
    signature = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    texts: list[str] = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                texts = [str(item) for item in cached.get("pages", [])]
        except (json.JSONDecodeError, OSError):
            pass
    if not texts:
        reader = PdfReader(path)
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
    missing = [i for i, text in enumerate(texts) if not text]
    if missing and allow_ocr:
        try:
            import fitz
            import pytesseract
            from PIL import Image

            if command := os.getenv("TESSERACT_CMD"):
                pytesseract.pytesseract.tesseract_cmd = command
            document = fitz.open(path)
            language = os.getenv("OCR_LANG", "vie+eng")
            for index in missing:
                pixmap = document[index].get_pixmap(dpi=200, alpha=False)
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
                texts[index] = pytesseract.image_to_string(
                    image, lang=language
                ).strip()
            document.close()
        except Exception as exc:  # OCR là dependency hệ thống tùy chọn
            warnings.warn(f"Không OCR được {path.name}: {exc}", stacklevel=2)

    if any(texts):
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"signature": signature, "pages": texts},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    doc_id = path.stem
    return [
        {
            "text": text,
            "metadata": {
                "doc_id": doc_id,
                "source": path.name,
                "page": index + 1,
            },
        }
        for index, text in enumerate(texts)
        if text
    ]


def load_documents(
    data_dir: str = DATA_DIR,
    strict: bool = False,
    allow_ocr: bool | None = None,
) -> list[dict]:
    """Nạp Markdown, text và PDF; giữ metadata nguồn/trang."""
    directory = Path(data_dir)
    allow_ocr = strict if allow_ocr is None else allow_ocr
    documents: list[dict] = []
    text_files = sorted([*directory.glob("*.md"), *directory.glob("*.txt")])
    converted_stems = {path.stem for path in text_files}

    for path in text_files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "doc_id": path.stem,
                        "source": path.name,
                    },
                }
            )
    for path in sorted(directory.glob("*.pdf")):
        if path.stem not in converted_stems:
            documents.extend(_read_pdf(path, allow_ocr))

    if strict and not documents:
        raise RuntimeError(
            "Không nạp được tài liệu. PDF ảnh cần Tesseract với gói ngôn ngữ vie."
        )
    return documents


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    """Baseline: gom đoạn văn đến giới hạn ký tự."""
    metadata = metadata or {}
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(
                Chunk(current, {**metadata, "chunk_index": len(chunks)})
            )
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(Chunk(current, {**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if part.strip()
    ]


@lru_cache(maxsize=1)
def _get_semantic_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SEMANTIC_MODEL)


def _lexical_similarity(left: str, right: str) -> float:
    tokenize = lambda value: set(re.findall(r"\w+", value.lower(), re.UNICODE))
    a, b = tokenize(left), tokenize(right)
    return len(a & b) / max(len(a | b), 1)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Nhóm các câu liền kề theo cosine similarity."""
    metadata = metadata or {}
    sentences = _sentences(text)
    if not sentences:
        return []
    try:
        vectors = _get_semantic_model().encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )
        similarities = [
            float(vectors[i - 1] @ vectors[i]) for i in range(1, len(vectors))
        ]
    except Exception as exc:
        warnings.warn(f"Dùng lexical fallback cho semantic chunking: {exc}")
        similarities = [
            _lexical_similarity(sentences[i - 1], sentences[i])
            for i in range(1, len(sentences))
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities, strict=True):
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentence)
    return [
        Chunk(
            " ".join(group),
            {**metadata, "chunk_index": index, "strategy": "semantic"},
        )
        for index, group in enumerate(groups)
    ]


def _pack_blocks(blocks: list[str], limit: int) -> list[str]:
    expanded = [
        piece.strip()
        for block in blocks
        for piece in (
            [block[i : i + limit] for i in range(0, len(block), limit)]
            if len(block) > limit
            else [block]
        )
        if piece.strip()
    ]
    packed: list[str] = []
    current = ""
    for block in expanded:
        if current and len(current) + len(block) + 2 > limit:
            packed.append(current)
            current = ""
        current = f"{current}\n\n{block}".strip()
    if current:
        packed.append(current)
    return packed


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Tạo parent lớn và child nhỏ có liên kết ổn định."""
    if child_size <= 0 or parent_size <= child_size:
        raise ValueError("parent_size phải lớn hơn child_size > 0")
    metadata = metadata or {}
    blocks = [part.strip() for part in text.split("\n\n") if part.strip()]
    parent_texts = _pack_blocks(blocks, parent_size)
    identity = str(metadata.get("doc_id") or metadata.get("source") or "doc")
    if metadata.get("page") is not None:
        identity += f"_page_{metadata['page']}"
    doc_id = re.sub(r"\W+", "_", identity).strip("_")
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{doc_id}_parent_{parent_index}"
        parent_metadata = {
            **metadata,
            "chunk_type": "parent",
            "parent_id": parent_id,
        }
        parents.append(Chunk(parent_text, parent_metadata))
        child_texts = _pack_blocks(_sentences(parent_text), child_size)
        for child_index, child_text in enumerate(child_texts):
            child_id = f"{parent_id}_child_{child_index}"
            children.append(
                Chunk(
                    child_text,
                    {
                        **metadata,
                        "chunk_type": "child",
                        "chunk_id": child_id,
                    },
                    parent_id,
                )
            )
    return parents, children


def chunk_structure_aware(
    text: str, metadata: dict | None = None
) -> list[Chunk]:
    """Tách Markdown theo heading và giữ heading trong nội dung."""
    metadata = metadata or {}
    matches = list(re.finditer(r"^#{1,3}\s+.+$", text, flags=re.MULTILINE))
    if not matches:
        clean = text.strip()
        return [
            Chunk(clean, {**metadata, "section": "", "strategy": "structure"})
        ] if clean else []

    chunks: list[Chunk] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(
            Chunk(
                preamble,
                {**metadata, "section": "Mở đầu", "strategy": "structure"},
            )
        )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start() : end].strip()
        section = re.sub(r"^#{1,3}\s+", "", match.group()).strip()
        chunks.append(
            Chunk(
                section_text,
                {**metadata, "section": section, "strategy": "structure"},
            )
        )
    return chunks


def _stats(chunks: list[Chunk]) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "num_chunks": len(lengths),
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "min_length": min(lengths, default=0),
        "max_length": max(lengths, default=0),
    }


def compare_strategies(documents: list[dict]) -> dict:
    """Chạy bốn chiến lược và trả thống kê so sánh."""
    basic: list[Chunk] = []
    semantic: list[Chunk] = []
    parents: list[Chunk] = []
    children: list[Chunk] = []
    structure: list[Chunk] = []
    for document in documents:
        text, metadata = document["text"], document.get("metadata", {})
        basic.extend(chunk_basic(text, metadata=metadata))
        semantic.extend(chunk_semantic(text, metadata=metadata))
        new_parents, new_children = chunk_hierarchical(text, metadata=metadata)
        parents.extend(new_parents)
        children.extend(new_children)
        structure.extend(chunk_structure_aware(text, metadata=metadata))

    hierarchical = _stats(children)
    hierarchical["num_parents"] = len(parents)
    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": hierarchical,
        "structure": _stats(structure),
    }
    for name, values in results.items():
        print(f"{name:<13} | {values}")
    return results


if __name__ == "__main__":
    compare_strategies(load_documents(strict=True))
