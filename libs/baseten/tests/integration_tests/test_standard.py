"""Standard LangChain interface tests.

Provider-specific reasoning assertions live in `test_reasoning.py`, but the
reasoning usage-metadata hooks stay here because `ChatModelIntegrationTests`
discovers and calls them from this standard-suite subclass.
"""

from __future__ import annotations

import base64
import os
import struct
import zlib
from typing import Literal, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_baseten import ChatBaseten
from tests.integration_tests._reasoning import get_reasoning_model

MODEL_NAME = "zai-org/GLM-5"
VISION_MODEL_NAME = "moonshotai/Kimi-K2.5"

pytestmark = [
    pytest.mark.requires("baseten_api_key"),
    pytest.mark.skipif(
        not os.environ.get("BASETEN_API_KEY"),
        reason="BASETEN_API_KEY not set",
    ),
]


class TestBasetenStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatBaseten

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # GLM-5 does not reliably respect tool_choice="any"
        return False

    @property
    def model_override_value(self) -> str:
        return os.environ.get(
            "BASETEN_MODEL_OVERRIDE",
            os.environ.get("BASETEN_REASONING_MODEL", "openai/gpt-oss-120b"),
        )

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """Advertise reasoning metadata support to the shared standard suite."""
        return {"invoke": ["reasoning_output"], "stream": ["reasoning_output"]}

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """Provide a reasoning-capable response for standard usage-metadata tests."""
        prompt = "What is 3^3?"
        model = get_reasoning_model()
        if not stream:
            return model.invoke(prompt)

        full: AIMessageChunk | None = None
        for chunk in model.stream(prompt):
            full = chunk if full is None else full + chunk

        assert full is not None, "Streaming reasoning call returned no chunks"
        return cast(AIMessage, full)


def _make_small_png(width: int = 64, height: int = 64) -> bytes:
    """Generate a minimal valid PNG (solid red) without Pillow."""

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        header
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


class TestBasetenVisionStandard(ChatModelIntegrationTests):
    """Standard suite against Kimi K2.5 (Baseten's only vision Model API).

    Overrides `test_image_inputs` with a small 64x64 PNG because the standard
    suite's hardcoded 1245x1395 image exceeds Kimi K2.5's vision-encoder
    embedding limit (HTTP 413).
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatBaseten

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": VISION_MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        return True


def test_vision_image_inputs() -> None:
    """Test vision via Kimi K2.5 with a small image.

    The standard suite's hardcoded 1245x1395 image exceeds Kimi K2.5's
    vision-encoder embedding limit (HTTP 413), so we test separately with
    a 64x64 solid-red PNG.
    """
    model = ChatBaseten(model=VISION_MODEL_NAME, temperature=0)
    image_data = base64.b64encode(_make_small_png()).decode("utf-8")

    # OpenAI CC format, base64 data
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the dominant color in this image."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ],
    )
    result = model.invoke([msg])
    assert isinstance(result.content, str)
    assert "red" in result.content.lower()

    # Standard LangChain format, base64 data
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the dominant color in this image."},
            {
                "type": "image",
                "base64": image_data,
                "mime_type": "image/png",
            },
        ],
    )
    result = model.invoke([msg])
    assert isinstance(result.content, str)
    assert "red" in result.content.lower()


def test_native_structured_output_integration() -> None:
    """Test Baseten's native JSON-schema structured output path."""
    from pydantic import BaseModel, Field

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    chat = ChatBaseten(
        model=MODEL_NAME,
        temperature=0,
    ).with_structured_output(Joke, method="json_schema")

    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, Joke)

    chunk = None
    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, Joke)
    assert chunk is not None, "Stream returned no chunks - possible API issue"
