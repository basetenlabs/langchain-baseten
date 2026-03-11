"""Integration tests for Baseten-specific chat model behavior."""

import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from langchain_baseten import ChatBaseten

pytestmark = [
    pytest.mark.requires("baseten_api_key"),
    pytest.mark.requires("baseten_dedicated_model_url"),
]


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    pytest.skip(f"{name} not set")


def test_chat_baseten_dedicated_url_invoke() -> None:
    """Test `ChatBaseten` with a dedicated model URL."""
    chat = ChatBaseten(
        model="dedicated-model",
        model_url=_get_required_env("BASETEN_DEDICATED_MODEL_URL"),
        baseten_api_key=SecretStr(_get_required_env("BASETEN_API_KEY")),
        temperature=0,
        max_tokens=50,
    )

    response = chat.invoke([HumanMessage(content="Hello from dedicated model!")])

    assert isinstance(response.content, str)
    assert response.content


def test_chat_baseten_dedicated_url_stream() -> None:
    """Test streaming against a dedicated model URL."""
    chat = ChatBaseten(
        model="dedicated-model",
        model_url=_get_required_env("BASETEN_DEDICATED_MODEL_URL"),
        baseten_api_key=SecretStr(_get_required_env("BASETEN_API_KEY")),
        temperature=0,
        max_tokens=30,
        streaming=True,
    )

    chunks = list(chat.stream([HumanMessage(content="Count to 3")]))

    assert chunks
    assert "".join(str(chunk.content) for chunk in chunks)


def test_chat_baseten_dedicated_url_only() -> None:
    """Test model name inference when only a dedicated URL is provided."""
    chat = ChatBaseten(
        model_url=_get_required_env("BASETEN_DEDICATED_MODEL_URL"),
        baseten_api_key=SecretStr(_get_required_env("BASETEN_API_KEY")),
        temperature=0,
        max_tokens=50,
    )

    assert chat.model_name

    response = chat.invoke([HumanMessage(content="Hello from dedicated model!")])

    assert isinstance(response.content, str)
    assert response.content
