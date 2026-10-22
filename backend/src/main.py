"""FastAPI application entry point."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings

logger = logging.getLogger("capstone.backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own process-wide resources.

    The embedding model is loaded exactly once here (never lazily) and the
    ChromaDB client is opened; both are attached to app.state.
    """
    settings = get_settings()
    started = time.perf_counter()
    logger.info("Starting %s", settings.app_name)

    from src.chroma_store import get_client
    from src.embeddings import E5Embedder
    from src.retrieval import Retriever

    embedder = E5Embedder(settings.embedding_model_name)
    client = get_client(settings.chroma_dir)
    app.state.embedder = embedder
    app.state.chroma = client
    app.state.retriever = Retriever(client, embedder)
    logger.info("Startup resources ready in %.1fs "
                "(embedding dim=%d).",
                time.perf_counter() - started, embedder.dimension)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/health")
def health() -> dict[str, str]:
    """Report service liveness for smoke checks."""
    return {"status": "ok"}
