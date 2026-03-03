"""
OpenGradient API client wrapper.

Provides a singleton factory for the OpenGradient Client and a helper to
ensure Permit2 token approval before inference calls.

VERIFICATION NOTE:
    The OpenGradient Client automatically handles TEE (Trusted Execution
    Environment) verification for every LLM call.  All requests are routed
    through TEE nodes that provide hardware-attested code execution (Intel
    TDX), ensuring prompts and responses are processed correctly and
    privately.  The x402 payment protocol settles each call on-chain via
    Base Sepolia, producing a `payment_hash` that serves as the
    cryptographic receipt.
"""

from __future__ import annotations

import logging
from typing import Optional

import opengradient as og

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_client: Optional[og.Client] = None


def get_client() -> og.Client:
    """Return (and lazily create) the OpenGradient client singleton.

    The client is initialised with:
      - ``private_key`` — wallet on Base Sepolia holding $OPG tokens.
      - ``email``       — optional, for Model Hub features.

    All LLM inference goes through TEE-verified execution automatically.
    """
    global _client
    if _client is None:
        settings.validate()
        logger.info("Initialising OpenGradient client (env=%s)", settings.environment)
        init_kwargs: dict = {"private_key": settings.private_key}
        if settings.email:
            init_kwargs["email"] = settings.email
        _client = og.Client(**init_kwargs)
        logger.info("OpenGradient client ready")
    return _client


async def ensure_approval(opg_amount: float = 5.0) -> dict:
    """Ensure the wallet has sufficient Permit2 allowance for $OPG.

    This only sends an on-chain approval transaction when the current
    allowance is below the requested amount.  Otherwise it returns
    immediately.

    Returns a dict with allowance info for display in the UI.
    """
    client = get_client()
    result = client.llm.ensure_opg_approval(opg_amount=opg_amount)
    return {
        "allowance_before": str(result.allowance_before),
        "allowance_after": str(result.allowance_after),
        "tx_hash": result.tx_hash,
    }
