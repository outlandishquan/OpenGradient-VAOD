"""
Application entrypoint.

Creates the FastAPI app, wires up routes, and runs startup validation.
Start with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Validate configuration and pre-warm the OpenGradient client."""
    logger.info("=== Verifiable AI Output Demo — starting ===")
    logger.info("Environment : %s", settings.environment)

    # Validate that the private key is set
    try:
        settings.validate()
        logger.info("Configuration OK")
    except RuntimeError as exc:
        logger.warning("Config warning: %s", exc)
        logger.warning(
            "The server will start, but inference calls will fail until "
            "a valid OG_PRIVATE_KEY is set in .env."
        )
        return

    # Pre-initialise the client and ensure Permit2 approval.
    # This is done eagerly so the first inference call doesn't block.
    try:
        from app.og_client import ensure_approval

        logger.info("Ensuring Permit2 OPG approval (5.0 OPG)…")
        approval = await ensure_approval(opg_amount=5.0)
        logger.info("Permit2 approval: %s", approval)
    except Exception as exc:
        logger.warning(
            "Could not ensure OPG approval at startup: %s  "
            "(inference may still work if allowance is already set)",
            exc,
        )
