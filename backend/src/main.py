"""FastAPI application entry point: /ask, /feedback, /universities, /health.

API contract (confirmed against frontend/src/app/app.ts, which hardcodes
http://localhost:8000):
- POST /ask {question, university, history?} -> {"answer": <arabic text>}; refusals
  return the deterministic CANNOT_CONFIRM string with status 200. The
  request deadline (settings.request_deadline_s) is anchored at the start
  of this handler, BEFORE retrieval/rerank, so an exhausted deadline
  surfaces as a fast 502 listing every provider's outcome — always before
  the frontend's 20s abort (which masks failures as a demo answer).
  ``history`` (optional) is the recent turns [{question, answer}, ...]
  the frontend includes for conversational follow-ups: the pipeline
  rewrites the follow-up into a standalone search query before retrieval
  (src/query_rewrite.py); malformed history is ignored, never a 400.
- POST /feedback {university, answer, feedback} -> 200 OK with an empty
  body; ``answer`` is the FULL answer text, ``feedback`` is "up"|"down"|null
  (null = the user removed their vote).
- GET /universities -> [{id, name, available}]; not called by the current
  frontend (it hardcodes ksu/kau locally) but built per the contract.

KNOWN ISSUE (flagged, deliberately NOT fixed — frontend changes are out of
scope): on any /ask failure or timeout the frontend silently shows a canned
demo answer instead of surfacing the error, so backend problems arrive as
fake answers. These endpoints therefore log every failing request at
WARNING so masking is always visible in the backend logs.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api_logging import JsonlLogger
from src.config import (
    UNIVERSITY_COLLECTIONS,
    UNIVERSITY_DISPLAY_NAMES,
    Settings,
    get_settings,
    is_university_available,
)
from src.generators import GenerationError, get_generation_chain
from src.pipeline import answer_question
from src.query_rewrite import normalize_history

logger = logging.getLogger("capstone.api")

# Per AGENTS.md validation: frontend question input cap.
QUESTION_MAX_CHARS = 1000


def _now_iso() -> str:
    """UTC timestamp for JSONL records (sortable, unambiguous)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _error_response(status: int, message: str) -> JSONResponse:
    """Clear JSON error body, logged at WARNING (the frontend masks failures)."""
    logger.warning("[api] %d: %s", status, message)
    return JSONResponse(status_code=status, content={"error": message})


def _load_resources(app: FastAPI, settings: Settings) -> None:
    """Load process-wide resources exactly ONCE at startup (never lazily).

    Embedding model + KSU reranker (both CPU) plus the Chroma client and
    the default failover chain; request handlers only read app.state.
    """
    from src.chroma_store import get_client
    from src.embeddings import E5Embedder
    from src.retrieval import Retriever
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    logger.info("Starting %s", settings.app_name)

    embedder = E5Embedder(settings.embedding_model_name)
    reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
    client = get_client(settings.chroma_dir)
    app.state.embedder = embedder
    app.state.reranker = reranker
    app.state.chroma = client
    app.state.retriever = Retriever(client, embedder)
    app.state.generation_chain = get_generation_chain(settings=settings)
    logger.info(
        "Startup resources ready in %.1fs (embedding dim=%d, reranker=%s).",
        time.perf_counter() - started,
        embedder.dimension,
        settings.reranker_model_name,
    )


def _lifespan_for(load_resources: bool) -> Callable[[FastAPI], AsyncIterator[None]]:
    """Lifespan factory: real startup loads models once; test mode skips it."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        if load_resources:
            _load_resources(app, settings)
        app.state.ask_logger = JsonlLogger(
            settings.data_dir / "logs", "ask"
        )
        app.state.feedback_logger = JsonlLogger(
            settings.data_dir / "logs", "feedback"
        )
        yield
        logger.info("Shutting down %s", settings.app_name)

    return lifespan


def _register_routes(app: FastAPI) -> None:
    """Attach the endpoint handlers to the app instance."""

    @app.post("/ask")
    def ask(payload: dict[str, Any], request: Request) -> Any:
        """Answer a question from RAG evidence (sync: runs in the threadpool)."""
        started = time.perf_counter()
        # Request deadline anchored HERE — before validation, retrieval and
        # rerank — so generation receives only the remaining time and can
        # never push the total past settings.request_deadline_s (17s < the
        # frontend's 20s abort). monotonic() is immune to system clock jumps.
        deadline = time.monotonic() + get_settings().request_deadline_s
        state = request.app.state

        question = payload.get("question")
        university = payload.get("university")
        if not isinstance(question, str) or not question.strip():
            return _error_response(400, "question must be a non-empty string")
        question = question.strip()
        if len(question) > QUESTION_MAX_CHARS:
            return _error_response(
                400,
                f"question exceeds the maximum of {QUESTION_MAX_CHARS} characters",
            )
        if not isinstance(university, str) or (
            university not in UNIVERSITY_COLLECTIONS
        ):
            return _error_response(
                400,
                f"unknown university {university!r}; expected 'ksu' or 'kau'",
            )
        if not is_university_available(university):
            return _error_response(
                503, f"university {university!r} is not available yet"
            )
        # Conversational follow-ups: validate + clip the history the
        # frontend sent (malformed shapes silently degrade to no history;
        # the request must never 400 because of it).
        settings = get_settings()
        turns = normalize_history(
            payload.get("history"),
            max_turns=settings.query_rewrite_max_history_turns,
            max_chars=settings.query_rewrite_max_turn_chars,
        )

        try:
            result = answer_question(
                question,
                state.retriever,
                state.generation_chain,
                university=university,
                reranker=state.reranker if university == "ksu" else None,
                deadline=deadline,
                history=turns,
            )
        except GenerationError as exc:
            return _error_response(
                502, f"all generation providers failed: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - one endpoint-wide policy
            logger.exception("[api] internal error on /ask")
            return _error_response(
                500, f"internal error: {type(exc).__name__}"
            )

        seconds = {
            "rewrite": round(result.rewrite_seconds, 3),
            "search": round(result.search_seconds, 3),
            "rerank": round(result.rerank_seconds, 3),
            "generation": round(result.generation_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        }
        state.ask_logger.write(
            {
                "timestamp": _now_iso(),
                "university": university,
                "question": question,
                "answer": result.answer,
                "refused": result.refused,
                "provider": result.provider,
                "model": result.model,
                "truncated": result.truncated,
                "seconds": seconds,
            }
        )
        logger.info(
            "[api] /ask university=%s refused=%s provider=%s total=%.2fs",
            university, result.refused, result.provider, seconds["total"],
        )
        # Always the "answer" key ("response" is only a frontend fallback).
        return {"answer": result.answer}

    @app.post("/feedback")
    def feedback(payload: dict[str, Any], request: Request) -> dict[str, str]:
        """Record user feedback; 200 with an empty JSON object (contract)."""
        state = request.app.state
        university = payload.get("university")
        answer_text = payload.get("answer")
        value = payload.get("feedback")
        if not isinstance(university, str) or (
            university not in UNIVERSITY_COLLECTIONS
        ):
            return _error_response(
                400,
                f"unknown university {university!r}; expected 'ksu' or 'kau'",
            )
        if not isinstance(answer_text, str) or not answer_text.strip():
            return _error_response(400, "answer must be a non-empty string")
        if value is not None and value not in ("up", "down"):
            return _error_response(
                400, "feedback must be 'up', 'down' or null"
            )
        question = state.ask_logger.last_assembly(university, answer_text)
        state.feedback_logger.write(
            {
                "timestamp": _now_iso(),
                "university": university,
                "answer": answer_text,
                "feedback": value,
                "matched_question": question,
            }
        )
        return {}

    @app.get("/universities")
    def universities() -> list[dict[str, Any]]:
        """Registry-driven list; no hardcoded university ids or names."""
        return [
            {
                "id": uni,
                "name": UNIVERSITY_DISPLAY_NAMES.get(uni, uni),
                "available": is_university_available(uni),
            }
            for uni in UNIVERSITY_COLLECTIONS
        ]

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report service liveness for smoke checks."""
        return {"status": "ok"}


def create_app(load_resources: bool = True) -> FastAPI:
    """Build and configure the FastAPI application.

    ``load_resources=False`` (for tests): the lifespan skips model loading;
    JSONL writers are still attached so handlers follow one code path, and
    tests then inject fakes into app.state directly.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=_lifespan_for(load_resources),
    )
    # CORS: the Angular dev server origin (frontend base URL is endorsed by
    # its hardcoded http://localhost:8000 target).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)
    return app


app = create_app()
