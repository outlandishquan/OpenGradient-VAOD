"""
Configuration module.

Loads environment variables from .env and exposes them as a typed Settings
object. The private key is the only required value — everything else has
sensible defaults for the OpenGradient testnet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (one level above this file)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings read from environment variables."""

    # REQUIRED — Base Sepolia wallet private key (holds $OPG tokens for x402).
    private_key: str = field(default_factory=lambda: os.environ.get("OG_PRIVATE_KEY", ""))

    # OPTIONAL — Email for Model Hub auth.
    email: str = field(default_factory=lambda: os.environ.get("OG_EMAIL", ""))

    # OPTIONAL — Human-readable environment label shown in the UI.
    environment: str = field(default_factory=lambda: os.environ.get("OG_ENVIRONMENT", "testnet"))

    # Default model used when the user does not select one.
    default_model: str = "GPT_5_MINI"

    # Default max tokens for inference calls.
    max_tokens: int = 512

    def validate(self) -> None:
        """Raise if critical configuration is missing."""
        if not self.private_key or self.private_key == "0xYOUR_PRIVATE_KEY_HERE":
            raise RuntimeError(
                "OG_PRIVATE_KEY is not set. "
                "Copy .env.example to .env and add your Base Sepolia private key. "
                "Get $OPG tokens from https://faucet.opengradient.ai"
            )


# Module-level singleton — import this wherever settings are needed.
settings = Settings()
