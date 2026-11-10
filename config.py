"""كل إعدادات النظام في مكان واحد: النماذج، الروابط، أعداد البحث، الأوزان، الحدود، المسارات.

لا يكتب أي ملف آخر هذه القيم بنفسه، بل يستوردها من هنا. بجانب كل قيمة سبب
اختيارها، ونتيجة قياسها إن قيست.

الاستخدام:
    import config
    config.TOP_K
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WEB_SOURCES_DIR = DATA_DIR / "web_sources"
STRUCTURED_DIR = DATA_DIR / "structured"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
PROMPTS_DIR = ROOT / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system.txt"
WEB_DIR = ROOT / "web"
API_KEY_FILE = ROOT / "api_key.txt"
REQUEST_LOG_FILE = ROOT / "logs" / "requests.jsonl"
FEEDBACK_LOG_FILE = ROOT / "logs" / "feedback.jsonl"

UNIVERSITY_NAMES = {"KSU": "جامعة الملك سعود", "KAU": "جامعة الملك عبدالعزيز"}
UNDERGRADUATE_STAGE = "المرحلة الجامعية"

MAX_CHARS = 1800
KSU_TRANSFER_MAX_CHARS = 2200
COVER_MAX = 450
STUB_MAX = 260
LEAD_MAX = 160

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"
EMBEDDING_BATCH_SIZE = 16
QUERY_BATCH_SIZE = 8

QDRANT_URL = os.environ.get("RAG_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTIONS = {"KSU": "ksu_rules", "KAU": "kau_rules"}
QDRANT_TIMEOUT = 5

DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
SPARSE_HEAD_FILE = "sparse_linear.pt"
INDEX_FILES = {
    "chunks": "chunks.jsonl",
    "vectors": "embeddings.npy",
    "weights": "sparse_weights.json",
}
SPARSE_BATCH_SIZE = 8
QUERY_SPARSE_CACHE_SIZE = 256

TOP_K = 4
RERANK_CANDIDATES = 30
RETRIEVE_CACHE_SIZE = 64
HISTORY_TURNS = 3

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

ANSWER_MODEL = "deepseek/deepseek-v4.1-flash-1017"
ANSWER_TEMPERATURE = 0.1
ANSWER_MAX_TOKENS = 4000
FALLBACK_ANSWER_MODEL = "qwen/qwen3-32b"
FALLBACK_MAX_TOKENS = 4000

RERANK_MODEL = "qwen/qwen3-reranker-8b"
RERANK_TEXT_MAX = 4000

CHAT_TIMEOUT = 120
RERANK_TIMEOUT = 30
CREDITS_TIMEOUT = 20
CHAT_RETRIES = 2
RERANK_RETRIES = 1
RETRY_WAIT_SECONDS = 2
