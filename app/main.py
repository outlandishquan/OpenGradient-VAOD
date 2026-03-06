"""
Application entrypoint.

Creates the FastAPI app, wires up routes, and runs startup validation.
Start with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import os

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Verifiable AI Output Demo",
    description=(
        "Demonstrates verifiable LLM inference using the OpenGradient SDK "
        "with TEE execution and x402 on-chain settlement."
    ),
    version="1.0.0",
)

# CORS — allow the local dev frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)

# Mount static files BEFORE routes so /static/* is matched first
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    logger.info("Static files mounted from %s", _static_dir)
else:
    logger.warning("Static directory not found: %s", _static_dir)


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Validate configuration on startup.

    The OpenGradient client and Permit2 approval are initialised lazily
    on the first inference call to avoid heavy memory usage during
    startup, which can cause OOM crashes on Render's free tier (512 MB).
    """
    logger.info("=== Verifiable AI Output Demo — starting ===")
    logger.info("Environment : %s", settings.environment)
    logger.info("Python PID  : %s", os.getpid())

    # Validate that the private key is set
    try:
        settings.validate()
        logger.info("Configuration OK — private key is set")
    except RuntimeError as exc:
        logger.warning("Config warning: %s", exc)
        logger.warning(
            "The server will start, but inference calls will fail until "
            "a valid OG_PRIVATE_KEY is set in .env."
        )

    # NOTE: We deliberately do NOT call ensure_approval() or import the
    # OpenGradient client at startup.  The SDK + web3 stack uses ~300 MB
    # of RAM, and Render free tier only provides 512 MB.  Importing
    # eagerly causes an OOM kill (exit code 254).  The client will be
    # initialised lazily on the first /infer request instead.
    logger.info("Startup complete — client will initialise on first request")
