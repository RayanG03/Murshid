"""Application settings loaded from environment variables and ``backend/.env``.

All model IDs and tunable settings for the RAG backend live here. Secret values
(API keys) are read from environment variables or ``backend/.env`` (gitignored);
never hardcode them in source files.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

# E5 embedding prefixes (required by the multilingual-e5 family).
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Supported universities (API contract: "ksu" | "kau").
UNIVERSITIES: tuple[str, ...] = ("ksu", "kau")

# Display names used in prompts per supported university.
UNIVERSITY_DISPLAY_NAMES: dict[str, str] = {
    "kau": "جامعة الملك عبدالعزيز",
    "ksu": "جامعة الملك سعود",
}

# Collections per university: first entry is the Q&A collection (retrieved
# first per AGENTS.md), the rest are document collections ranked together.
# Must stay in sync with the COL_* constants in src.chroma_store (tested).
UNIVERSITY_COLLECTIONS: dict[str, tuple[str, ...]] = {
    "kau": ("kau_qa_verified", "kau_web", "kau_pdf"),
    "ksu": ("ksu_qa", "ksu_txt", "ksu_pdf"),
}


class Settings(BaseSettings):
    """Central configuration for the backend.

    Every field can be overridden by an environment variable of the same name
    (case-insensitive), e.g. ``DEEPSEEK_MODEL_ID`` or ``DEEPINFRA_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Capstone II Arabic RAG Backend"

    # CORS: Angular dev server origin (frontend hardcodes http://localhost:8000).
    cors_origins: list[str] = ["http://localhost:4200"]

    # Secrets (placeholders; real values come from backend/.env).
    deepinfra_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""

    # Generator models: DeepSeek / Qwen / Llama served via DeepInfra
    # (OpenAI-compatible endpoint), Gemini via its free tier.
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"
    deepseek_model_id: str = "deepseek-ai/DeepSeek-V3"
    qwen_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    llama_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    gemini_model_id: str = "gemini-flash-latest"

    # Groq (OpenAI-compatible): gpt-oss is a reasoning model whose reasoning
    # tokens count against the completion cap, so reasoning effort is set low
    # and reasoning output is excluded (verified against Groq docs
    # 2026-10-23: reasoning_effort low/medium/high for gpt-oss; include_reasoning
    # controls exposure; max_tokens is deprecated -> max_completion_tokens).
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_id: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "low"
    groq_include_reasoning: bool = False

    # Mistral (OpenAI-compatible chat/completions).
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model_id: str = "mistral-small-latest"

    # Default failover chain: tried in order, one shared time budget
    # (src/generators.py generate_with_meta). DeepInfra models (deepseek,
    # qwen, llama) remain selectable explicitly via get_generator() but are
    # deliberately NOT part of the default chain.
    default_generation_providers: tuple[str, ...] = ("groq", "mistral", "gemini")

    # Total-request deadline: anchored at the START of /ask handling
    # (before retrieval/rerank), passed down into generate_with_meta as an
    # absolute time.monotonic() timestamp. Retrieval + rerank consume the
    # first share; generation gets the remaining time and no attempt may
    # start unless it fits. The frontend aborts at 20s, so an exhausted
    # deadline must surface as a fast 502 listing each provider's outcome.
    request_deadline_s: float = 17.0
    per_attempt_timeout_s: float = 8.0

    # Embeddings (loaded once at startup in a later stage).
    embedding_model_name: str = "intfloat/multilingual-e5-large"

    # Cross-encoder reranker for the KSU retrieval gate (loaded once at
    # startup, next to the embedder; never lazily). Its predict() returns
    # PROBABILITIES for num_labels==1 — used as received, never re-squashed.
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    # KSU rerank gate: from the merged q&a+txt+pdf dense candidates keep the
    # top-N pool, rerank it, and refuse when nothing reaches the threshold.
    # Rationale (src.ksu_text_clean_diag, TUNE only): with variant-C texts
    # the rerank scores separate cleanly — min RELEVANT top1 0.5855 vs max
    # REFUSAL top1 0.2117, overlap 0 — so 0.40 centers the feasible window.
    ksu_rerank_pool_size: int = 6
    ksu_rerank_refuse_threshold: float = 0.40

    # Gemini REST endpoint and pacing (free-tier friendly).
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_request_delay_seconds: float = 4.0
    gemini_retry_backoff_base_seconds: float = 4.0

    # Local OCR (diagnostics): Tesseract binary with Arabic traineddata.
    tesseract_path: Path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    ocr_low_confidence_threshold: float = 60.0

    # LLM call behaviour: retry with exponential backoff, per-request progress log.
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_retry_backoff_base_seconds: float = 4.0

    # Gemini answer-output cap. Thinking counts against maxOutputTokens on
    # some models, so thinkingConfig (see src/generators.py) must be set
    # beside this cap to protect the answer budget.
    gemini_max_output_tokens: int = 1024

    # Context-assembly relevance floors (cosine), calibrated on Stage 3 data:
    # relevant probes: QA 0.891-0.946, docs 0.793-0.896;
    # off-topic probes: QA 0.758-0.821, docs 0.726-0.782.
    relevance_floor_qa: float = 0.83
    relevance_floor_docs: float = 0.79

    # Confidence gate per side (Stage 4 fix): after the floor, a side passes
    # only if its top hit is a strong leader OR clearly separated from the
    # runner-ups. Calibrated on 28 probes (exact scores): the highest
    # domain-adjacent-wrong leader is the bus query at 0.8672 (gap 0.038);
    # the lowest relevant carrying-side leaders are 0.8698 (housing doc) and
    # 0.8703 (GPA qa). The separating window is only 0.0026 wide -> 0.869 is
    # the cut, with sub-milli margins on both sides. This is inherently
    # fragile: drift should be handled by retrieval-side improvements
    # (reranking / hybrid search), not by nudging this threshold. Refusing a
    # borderline-relevant query is the safe error direction.
    gate_strong_leader: float = 0.869
    gate_gap: float = 0.04

    # Context assembly sizing.
    context_max_qa: int = 3
    context_max_docs: int = 4

    # Filesystem locations (generated files never go into backend/sources/).
    sources_dir: Path = BACKEND_DIR / "sources"
    data_dir: Path = BACKEND_DIR / "data"
    chroma_dir: Path = BACKEND_DIR / "chroma_db"

    # Retrieval / embedding runtime.
    embedding_batch_size: int = 32
    retrieval_top_k: int = 5
    chunk_max_chars: int = 1200
    chunk_target_chars: int = 700

    # KSU-only token cap for chunk splitting (step 0): the multilingual-e5
    # tokenizer packs Arabic text far denser than 1200 chars, permanently
    # truncating 28/128 ksu_pdf chunks at embed time. 450 tokens plus the
    # "passage: " prefix and CLS/SEP stays safely under the 512 budget.
    # KAU keeps the 1200-char setting unchanged (known truncation risk,
    # deferred to the evaluation stage).
    ksu_chunk_max_tokens: int = 450


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance (loaded once per process)."""
    return Settings()


def get_university_gates(university: str) -> dict[str, float]:
    """Per-university gates for the pipeline.

    KAU returns the calibrated floor/leader/gap quartet for
    src.context_assembly (Settings remains the single source of truth).
    KSU returns its rerank-gate values; the pipeline branches on
    university before touching either shape. Unknown universities fail
    loudly instead of silently reusing another university's thresholds.
    """
    if university == "kau":
        settings = get_settings()
        return {
            "floor_qa": settings.relevance_floor_qa,
            "floor_docs": settings.relevance_floor_docs,
            "strong_leader": settings.gate_strong_leader,
            "gap": settings.gate_gap,
        }
    if university == "ksu":
        settings = get_settings()
        return {
            "rerank_threshold": settings.ksu_rerank_refuse_threshold,
            "rerank_pool_size": float(settings.ksu_rerank_pool_size),
        }
    raise ValueError(
        f"No calibrated context gates for university {university!r} yet"
    )


def is_university_available(university: str) -> bool:
    """True only if the university is registered AND its gates are configured.

    Designed for the future API layer (GET /universities): never raises.
    KSU stays unavailable until calibrated gate values exist in this module
    (no hardcoded university check; availability follows the registry).
    """
    if university not in UNIVERSITY_COLLECTIONS:
        return False
    try:
        get_university_gates(university)
    except ValueError:
        return False
    return True
