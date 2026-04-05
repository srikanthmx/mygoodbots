"""FastAPI application factory for the multi-bot AI gateway."""
from __future__ import annotations

from fastapi import FastAPI

from backend.gateway.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Bot AI Gateway",
        version="0.1.0",
        description="Unified API gateway for the multi-bot AI system",
    )
    app.include_router(router)
    return app


app = create_app()
