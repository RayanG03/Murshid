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
    # Smallest remaining budget that may still START a provider attempt
    # (its timeout is then clipped to the remaining time). Before this,
    # an attempt needed the full per_attempt_timeout_s (8s) left: on a
    # laptop CPU where KSU search + rerank took ~9-11s, the LLM was never
    # called and /ask returned 502 (shown as a canned answer by the UI)
    # although Groq usually answers in 1-3s. The attempt still ends
    # before the deadline, so the 17s < 20s frontend guarantee holds.
    # Lowered 3.0 -> 1.5 (live 2026-10-25): Groq answers in 1.0-1.4s on
    # this project's prompts, and a question whose search + rerank took
    # 14.2s was refused generation with 2.8s still left (502 -> canned UI
    # answer).
    min_attempt_seconds: float = 1.5
    # Run one dummy search + rerank per university at startup (src/main.py
    # _warm_up) so the first real question is not slowed by lazy loading.
    warm_up_on_startup: bool = True
    # Per-provider usage counters for the /status page, persisted to
    # data/logs/usage.json (no keys, no request text).
    usage_tracking_persist: bool = True

    # Conversational follow-ups ("summarize that in points", "are there
    # dates?", ...): the frontend sends the last turns and the pipeline
    # rewrites the follow-up into a standalone search query BEFORE
    # retrieval (src/query_rewrite.py). The rewrite shares the request
    # deadline but is additionally capped by query_rewrite_max_seconds so
    # it can never eat the answer's budget; on any failure it falls back
    # to the raw question and the request proceeds unchanged.
    query_rewrite_enabled: bool = True
    query_rewrite_max_history_turns: int = 2
    query_rewrite_max_seconds: float = 4.0
    query_rewrite_max_query_chars: int = 300
    # Per-turn clipping for the history sent by the frontend (bounds the
    # rewrite prompt size regardless of answer length).
    query_rewrite_max_turn_chars: int = 1200
    # Safety net for rewrite topic drift: when the REWRITTEN query is
    # refused by the gate, retry the gate once with the user's own wording
    # if at least this much of the request deadline remains (one more
    # search + rerank, ~5-8s on a laptop CPU). Live case 2026-10-25: a new
    # question after a CS-admission turn was rewritten "... في كلية علوم
    # الحاسب" and refused although the regulation answers it.
    rewrite_fallback_min_remaining_s: float = 9.0

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
    relevance_floor_docs: float = 0.80

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
    gate_strong_leader: float = 0.832
    gate_gap: float = 0.012

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

    # ------------------------------------------------------------------
    # Improvements (2026-10-25). Every switch that changes retrieval scores
    # or gate decisions defaults to the CALIBRATED behavior above; flip it
    # only after src.eval_suite shows it helps (compare with --set).
    # ------------------------------------------------------------------

    # KAU gate: "cosine" = calibrated floor/leader/gap quartet (default);
    # "rerank" = the KSU-style cross-encoder gate. The reranker is already
    # loaded at startup; calibrate kau_rerank_refuse_threshold with
    # `python -m src.eval_suite --university kau --set kau_gate_mode=rerank`.
    # Enabled 2026-10-25 from src.eval_suite (51 held-out questions): the
    # cosine scores OVERLAP (max refuse 0.8624 > min hit 0.8411, so no
    # cosine threshold separates them) while rerank scores separate
    # cleanly (max refuse 0.34 < min hit 0.42; 0.58 with dialect mapping).
    # Refusals 4/6 -> 6/6, hits unchanged. Cost: ~7s per KAU request on a
    # laptop CPU (same as KSU). Set KAU_GATE_MODE=cosine to revert.
    kau_gate_mode: str = "rerank"
    kau_rerank_pool_size: int = 6
    kau_rerank_refuse_threshold: float = 0.40

    # Rerank-gate context filter: after the gate passes, keep only items
    # scoring >= max(min_score, relative * top). Before this, every pool
    # item reached the LLM even with a near-zero rerank probability.
    rerank_context_filter: bool = True
    rerank_context_min_score: float = 0.10
    rerank_context_relative: float = 0.20

    # Follow-ups: also show the rewritten standalone question to the
    # answer model (retrieval already used it; the answer prompt did not).
    pass_standalone_question: bool = True

    # Opt-in fallback AFTER a gate refusal (src/general_fallback.py):
    # "off" | "web" (Gemini + Google Search on the official domain only,
    # links shown) | "llm" (general questions only, labeled "not verified")
    # | "web_then_llm". Off by default: it changes the project's
    # "official sources only" guarantee — enable after team agreement.
    general_fallback_mode: str = "off"
    general_fallback_search_model: str = "gemini-flash-latest"
    # Web search backend for the "web" fallback: "tavily" (official domain
    # enforced by the API's include_domains; page text answered by the
    # normal provider chain), "gemini" (Google Search grounding), or
    # "auto" = tavily when TAVILY_API_KEY is set, else gemini.
    web_search_provider: str = "auto"
    tavily_api_key: str = ""
    # Live 2026-10-25: 3 results x 2500 chars on the raw dialect question
    # found only one program page for "الكليات في المزاحمية" (incomplete
    # answer). Search now uses the normalized query + university name.
    tavily_max_results: int = 5
    tavily_max_chars_per_page: int = 12000
    # "advanced" returns the most relevant chunks per page (2 credits per
    # search; the free plan's monthly credits cover a few hundred questions).
    tavily_search_depth: str = "advanced"
    tavily_chunks_per_source: int = 3
    # Passage selection: pages are split into ~700-char passages, scored
    # with BM25 against the query (src.lexical, no extra model -> no CPU
    # cost), and the best N passages across all pages go to the model.
    web_passage_chars: int = 700
    web_context_passages: int = 8

    # Saudi-dialect -> MSA word mapping on the SEARCH query only (the
    # answer prompt keeps the user's wording). Changes cosine scores, so it
    # is off until evaluated.
    # Enabled 2026-10-25: KSU hits 17/19 -> 18/19, no regressions, no
    # latency cost (src.eval_suite --matrix).
    dialect_normalization: bool = True

    # Hybrid retrieval: add up to N keyword (BM25) candidates per side to
    # the rerank pool (rerank gates only). 0 = off.
    hybrid_lexical_extra: int = 0

    # Numeric grounding guard: numbers in the answer that appear nowhere in
    # the context. "off" | "log" (record in ask.jsonl) | "refuse".
    numeric_guard_mode: str = "log"

    # In-process answer cache for repeated questions without history
    # (protects free-tier provider quotas during demos). 0 disables.
    answer_cache_size: int = 256
    answer_cache_ttl_s: float = 3600.0


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
