"""
FastAPI route handlers.

GET  /       — Serve the HTML UI.
POST /infer  — Accept a prompt, run inference, return JSON.
GET  /health — Simple health-check.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.inference import run_inference, MODEL_MAP, SETTLEMENT_MAP

logger = logging.getLogger(__name__)

router = APIRouter()

# Jinja2 template directory (relative to project root)
templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    """JSON body for the /infer endpoint."""
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: Optional[str] = None
    settlement_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main UI page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": list(MODEL_MAP.keys()),
            "settlement_modes": list(SETTLEMENT_MAP.keys()),
        },
    )


@router.post("/infer")
async def infer(body: InferRequest):
    """Run OpenGradient LLM inference and return structured results.

    The response JSON includes the model output plus all verification
    and payment metadata surfaced by the SDK.
    """
    result = run_inference(
        prompt=body.prompt,
        model_name=body.model,
        settlement_mode=body.settlement_mode,
    )

    status = 200 if result.get("success") else 500
    if result.get("status_code") == 402:
        status = 402
    elif result.get("status_code") == 401:
        status = 401

    return JSONResponse(content=result, status_code=status)


@router.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}
