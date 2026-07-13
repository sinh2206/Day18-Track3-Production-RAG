"""Cấu hình dùng chung, có thể ghi đè bằng biến môi trường."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
ENABLE_ENRICHMENT = env_bool("ENABLE_ENRICHMENT")
RAGAS_STRICT = env_bool("RAGAS_STRICT")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab18_production")
NAIVE_COLLECTION = os.getenv("NAIVE_COLLECTION", "lab18_naive")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SEMANTIC_MODEL = os.getenv(
    "SEMANTIC_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

DATA_DIR = str(ROOT_DIR / "data")
TEST_SET_PATH = str(ROOT_DIR / "test_set.json")
REPORT_DIR = str(ROOT_DIR / "reports")
