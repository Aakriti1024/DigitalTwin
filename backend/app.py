"""
FastAPI application entry point for the Digital Twin backend.

Mounts all API routers and serves the frontend static files.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import API_HOST, API_PORT, FRONTEND_DIR

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.routes import twin, predict, agent, simulation

import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(
        title="Digital Twin — Turbofan Engine Health Monitor",
        description=(
            "REST API for the turbofan engine digital twin system. "
            "Provides real-time health monitoring, RUL prediction, "
            "agentic maintenance recommendations, and what-if simulation."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS (permissive for development) ───────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routers ──────────────────────────────────────────────────────
    app.include_router(twin.router)
    app.include_router(predict.router)
    app.include_router(agent.router)
    app.include_router(simulation.router)

    # ── Static frontend files ────────────────────────────────────────────
    css_dir = os.path.join(FRONTEND_DIR, "css")
    js_dir = os.path.join(FRONTEND_DIR, "js")
    if os.path.isdir(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.isdir(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")
    if os.path.isdir(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        logger.info("Serving frontend static files from: %s", FRONTEND_DIR)
    else:
        logger.warning(
            "Frontend directory not found at '%s' — static files will not be served.",
            FRONTEND_DIR,
        )

    # ── Application-level routes ─────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root():
        """Serve the frontend single-page application."""
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return JSONResponse(
            status_code=200,
            content={
                "message": "Digital Twin API is running. Visit /api/docs for the interactive API documentation.",
                "docs": "/api/docs",
            },
        )

    @app.get("/health", tags=["System"])
    async def health_check():
        """Simple liveness probe."""
        return {"status": "ok", "version": "1.0.0"}

    return app


# ---------------------------------------------------------------------------
# Instantiate application
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Development entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
