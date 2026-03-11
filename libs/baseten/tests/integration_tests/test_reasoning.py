"""Integration tests for Baseten reasoning behavior.

These tests cover Baseten-specific reasoning assertions that are not part of
the shared `ChatModelIntegrationTests` contract.
"""

from __future__ import annotations

import os

import pytest

from tests.integration_tests._reasoning import get_reasoning_model

pytestmark = [
    pytest.mark.requires("baseten_api_key"),
    pytest.mark.skipif(
        not os.environ.get("BASETEN_API_KEY"),
        reason="BASETEN_API_KEY not set",
    ),
]


def test_reasoning_content_integration() -> None:
    """Assert Baseten's provider-specific `reasoning_content` field is exposed."""
    response = get_reasoning_model().invoke("What is 3^3?")

    assert response.content
    assert response.additional_kwargs["reasoning_content"]


def test_reasoning_usage_metadata_integration() -> None:
    """Keep a direct Baseten reasoning metadata assertion outside standard tests."""
    response = get_reasoning_model().invoke("What is 3^3?")

    assert response.usage_metadata is not None
    assert response.usage_metadata["output_token_details"]["reasoning"] is not None
