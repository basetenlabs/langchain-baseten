"""Helpers for reasoning-capable Baseten integration tests.

Shared by the standard-suite hook implementation and the provider-specific
reasoning assertions so both files exercise the same model configuration.
"""

from __future__ import annotations

import os

from langchain_baseten import ChatBaseten

DEFAULT_REASONING_MODEL_NAME = "openai/gpt-oss-120b"


def get_reasoning_model() -> ChatBaseten:
    """Build the reasoning-capable model used across Baseten integration tests."""
    return ChatBaseten(
        model=os.environ.get("BASETEN_REASONING_MODEL", DEFAULT_REASONING_MODEL_NAME),
        temperature=0,
        extra_body={"reasoning_effort": "high"},
    )
