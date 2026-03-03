"""
Inference service layer.

Orchestrates a single inference call through the OpenGradient SDK and
returns a structured dictionary containing the model output together with
all verification and payment metadata surfaced by the SDK.

VERIFICATION DATA – WHERE IT COMES FROM:
    ``TextGenerationOutput`` (returned by ``client.llm.chat()``) exposes:
      • ``chat_output``       — dict with the assistant message (``content``,
                                ``role``, optionally ``tool_calls``).
      • ``payment_hash``      — x402 payment transaction hash on Base Sepolia.
                                Present when the x402 payment was successfully
                                signed and settled.  This is the primary
                                *verification indicator* — its existence proves
                                the inference was paid for and routed through
                                the OpenGradient TEE network.
      • ``transaction_hash``  — on-chain settlement transaction hash.
                                For TEE providers this may be ``"external"``.
      • ``finish_reason``     — why the model stopped (``stop``, ``length``,
                                ``tool_calls``, etc.).
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import opengradient as og

from app.config import settings
from app.og_client import get_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name → TEE_LLM enum mapping
# ---------------------------------------------------------------------------
MODEL_MAP: Dict[str, og.TEE_LLM] = {
    "GPT_4_1_2025_04_14": og.TEE_LLM.GPT_4_1_2025_04_14,
    "GPT_5": og.TEE_LLM.GPT_5,
    "GPT_5_MINI": og.TEE_LLM.GPT_5_MINI,
    "GPT_5_2": og.TEE_LLM.GPT_5_2,
    "O4_MINI": og.TEE_LLM.O4_MINI,
    "CLAUDE_SONNET_4_5": og.TEE_LLM.CLAUDE_SONNET_4_5,
    "CLAUDE_SONNET_4_6": og.TEE_LLM.CLAUDE_SONNET_4_6,
    "CLAUDE_HAIKU_4_5": og.TEE_LLM.CLAUDE_HAIKU_4_5,
    "CLAUDE_OPUS_4_5": og.TEE_LLM.CLAUDE_OPUS_4_5,
    "CLAUDE_OPUS_4_6": og.TEE_LLM.CLAUDE_OPUS_4_6,
    "GEMINI_2_5_FLASH": og.TEE_LLM.GEMINI_2_5_FLASH,
    "GEMINI_2_5_PRO": og.TEE_LLM.GEMINI_2_5_PRO,
    "GEMINI_2_5_FLASH_LITE": og.TEE_LLM.GEMINI_2_5_FLASH_LITE,
    "GEMINI_3_PRO": og.TEE_LLM.GEMINI_3_PRO,
    "GEMINI_3_FLASH": og.TEE_LLM.GEMINI_3_FLASH,
    "GROK_4": og.TEE_LLM.GROK_4,
    "GROK_4_FAST": og.TEE_LLM.GROK_4_FAST,
    "GROK_4_1_FAST": og.TEE_LLM.GROK_4_1_FAST,
}

# ---------------------------------------------------------------------------
# Settlement mode mapping
# ---------------------------------------------------------------------------
SETTLEMENT_MAP: Dict[str, og.x402SettlementMode] = {
    "SETTLE": og.x402SettlementMode.SETTLE,
    "SETTLE_METADATA": og.x402SettlementMode.SETTLE_METADATA,
    "SETTLE_BATCH": og.x402SettlementMode.SETTLE_BATCH,
}


def _resolve_model(name: Optional[str]) -> og.TEE_LLM:
    """Resolve a model name string to its ``TEE_LLM`` enum value."""
    key = (name or settings.default_model).upper()
    if key not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{key}'. Available: {', '.join(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[key]


def _resolve_settlement(name: Optional[str]) -> og.x402SettlementMode:
    """Resolve a settlement mode string to its enum value."""
    key = (name or "SETTLE_METADATA").upper()
    if key not in SETTLEMENT_MAP:
        raise ValueError(
            f"Unknown settlement mode '{key}'. "
            f"Available: {', '.join(SETTLEMENT_MAP.keys())}"
        )
    return SETTLEMENT_MAP[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    prompt: str,
    model_name: Optional[str] = None,
    settlement_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single LLM inference call and return structured results.

    Parameters
    ----------
    prompt : str
        The user's input text.
    model_name : str, optional
        Key from ``MODEL_MAP``.  Falls back to ``settings.default_model``.
    settlement_mode : str, optional
        Key from ``SETTLEMENT_MAP``.  Falls back to ``SETTLE_METADATA``.

    Returns
    -------
    dict
        A flat dictionary containing the model output, verification
        metadata, and timing information.  See inline comments below for
        the provenance of each field.
    """
    model_enum = _resolve_model(model_name)
    settlement_enum = _resolve_settlement(settlement_mode)
    client = get_client()

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # ---------------------------------------------------------------
        # Call OpenGradient's TEE-verified chat endpoint.
        #
        # Under the hood the SDK:
        #   1. Signs an x402 payment with the wallet's private key.
        #   2. Sends the request to a TEE node on the OG network.
        #   3. The TEE node forwards the prompt to the upstream LLM
        #      provider (OpenAI, Anthropic, Google, xAI) inside an
        #      Intel TDX enclave, providing hardware attestation.
        #   4. The response is returned along with a payment_hash
        #      (the on-chain receipt) and a transaction_hash
        #      (the settlement tx).
        # ---------------------------------------------------------------
        result = client.llm.chat(
            model=model_enum,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant.  Answer concisely."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.max_tokens,
            temperature=0.0,
            x402_settlement_mode=settlement_enum,
        )

        # ---------------------------------------------------------------
        # Extract fields from TextGenerationOutput
        # ---------------------------------------------------------------
        chat_output = result.chat_output or {}
        model_response = chat_output.get("content", "")
        payment_hash = getattr(result, "payment_hash", None)
        transaction_hash = getattr(result, "transaction_hash", None)
        finish_reason = getattr(result, "finish_reason", None)

        # ---------------------------------------------------------------
        # Verification status badge logic.
        #
        # The presence of a payment_hash OR transaction_hash proves the
        # inference was routed through the OpenGradient TEE network.
        # For TEE providers, transaction_hash is typically "external",
        # which still indicates verified TEE execution.
        # ---------------------------------------------------------------
        if payment_hash or transaction_hash:
            verification_status = "verified"
        else:
            verification_status = "unverified"

        return {
            "success": True,
            "prompt": prompt,
            "model": model_enum.value if hasattr(model_enum, "value") else str(model_enum),
            "model_key": model_name or settings.default_model,
            "response": model_response,
            "execution_mode": "TEE (Trusted Execution Environment)",
            "settlement_mode": settlement_enum.name,
            "verification_status": verification_status,
            # --- x402 / settlement metadata (from SDK response) ---
            "payment_hash": payment_hash,
            "transaction_hash": transaction_hash,
            "finish_reason": finish_reason,
            "timestamp": timestamp,
            "environment": settings.environment,
        }

    except Exception as exc:
        # ---------------------------------------------------------------
        # Graceful error handling
        # ---------------------------------------------------------------
        error_msg = str(exc)
        error_type = type(exc).__name__

        # Classify well-known HTTP/SDK errors
        status_code = None
        if "402" in error_msg:
            status_code = 402
            friendly = (
                "Payment Required — your wallet may not have sufficient "
                "$OPG tokens.  Visit https://faucet.opengradient.ai to "
                "top up."
            )
        elif "401" in error_msg or "Unauthorized" in error_msg:
            status_code = 401
            friendly = (
                "Unauthorized — check that OG_PRIVATE_KEY is correct and "
                "the wallet is funded on Base Sepolia."
            )
        elif "timeout" in error_msg.lower() or "connect" in error_msg.lower():
            friendly = (
                "Network error — could not reach the OpenGradient "
                "inference endpoint.  Please try again."
            )
        else:
            friendly = f"{error_type}: {error_msg}"

        logger.error("Inference failed: %s", error_msg, exc_info=True)

        return {
            "success": False,
            "prompt": prompt,
            "error": friendly,
            "error_type": error_type,
            "status_code": status_code,
            "timestamp": timestamp,
            "traceback": traceback.format_exc(),
        }
