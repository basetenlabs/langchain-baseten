"""Test ChatBaseten chat model."""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from openai import BaseModel
from pydantic import SecretStr

from langchain_baseten import ChatBaseten


class MockOpenAIResponse(BaseModel):
    """Mock OpenAI response model."""

    choices: list
    error: None = None

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: Literal["json", "python"] | str = "python",  # noqa: PYI051
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Literal["none", "warn", "error"] | bool = True,
        context: dict[str, Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary, ensuring `reasoning_content` is included."""
        choices_list = []
        for choice in self.choices:
            message_dict = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if hasattr(choice.message, "reasoning_content"):
                message_dict["reasoning_content"] = choice.message.reasoning_content
            choices_list.append({"message": message_dict})

        return {"choices": choices_list, "error": self.error}


def test_chat_baseten_init() -> None:
    """Test ChatBaseten initialization."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
        temperature=0.7,
        max_tokens=100,
    )
    assert chat.model_name == "zai-org/GLM-5.2"
    assert chat.temperature == 0.7
    assert chat.max_tokens == 100


def test_chat_baseten_init_missing_api_key() -> None:
    """Test ChatBaseten initialization with missing API key."""
    original_key = os.environ.get("BASETEN_API_KEY")
    if "BASETEN_API_KEY" in os.environ:
        del os.environ["BASETEN_API_KEY"]

    try:
        with pytest.raises(ValueError, match="BASETEN_API_KEY must be set"):
            ChatBaseten(model="zai-org/GLM-5.2")
    finally:
        if original_key is not None:
            os.environ["BASETEN_API_KEY"] = original_key


def test_chat_baseten_llm_type() -> None:
    """Test ChatBaseten LLM type."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    assert chat._llm_type == "baseten-chat"


def test_chat_baseten_accepts_api_key_alias() -> None:
    """Test ChatBaseten accepts inherited `api_key` alias."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        api_key=SecretStr("test_key"),
    )

    assert chat.baseten_api_key is not None
    assert chat.root_client is not None


def test_chat_baseten_accepts_base_url_alias() -> None:
    """Test ChatBaseten accepts `base_url` alias."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
        base_url="https://proxy.example/v1",
    )

    assert str(chat.root_client.base_url).rstrip("/") == "https://proxy.example/v1"


def test_chat_baseten_reads_base_url_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ChatBaseten reads the base URL from `BASETEN_BASE_URL`."""
    monkeypatch.setenv("BASETEN_BASE_URL", "https://env.example/v1")
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )

    assert str(chat.root_client.base_url).rstrip("/") == "https://env.example/v1"


def test_chat_baseten_falls_back_to_api_base_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ChatBaseten falls back to `BASETEN_API_BASE`."""
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)
    monkeypatch.setenv("BASETEN_API_BASE", "https://api-base.example/v1")
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )

    assert str(chat.root_client.base_url).rstrip("/") == "https://api-base.example/v1"


def test_chat_baseten_base_url_env_wins_over_api_base_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `BASETEN_BASE_URL` takes precedence over `BASETEN_API_BASE`."""
    monkeypatch.setenv("BASETEN_BASE_URL", "https://base-url.example/v1")
    monkeypatch.setenv("BASETEN_API_BASE", "https://api-base.example/v1")
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )

    assert str(chat.root_client.base_url).rstrip("/") == "https://base-url.example/v1"


def test_chat_baseten_base_url_arg_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test an explicit `base_url` overrides `BASETEN_BASE_URL`."""
    monkeypatch.setenv("BASETEN_BASE_URL", "https://env.example/v1")
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
        base_url="https://explicit.example/v1",
    )

    assert str(chat.root_client.base_url).rstrip("/") == "https://explicit.example/v1"


def test_chat_baseten_rejects_openai_api_key_field() -> None:
    """Test ChatBaseten rejects OpenAI-specific API key field names."""
    with pytest.raises(ValueError, match="`openai_api_key` is not supported"):
        kwargs: Any = {
            "model": "zai-org/GLM-5.2",
            "openai_api_key": "test_key",
        }
        ChatBaseten(**kwargs)


def test_chat_baseten_rejects_openai_api_base_field() -> None:
    """Test ChatBaseten rejects OpenAI-specific base URL field names."""
    with pytest.raises(ValueError, match="`openai_api_base` is not supported"):
        kwargs: Any = {
            "model": "zai-org/GLM-5.2",
            "baseten_api_key": SecretStr("test_key"),
            "openai_api_base": "https://proxy.example/v1",
        }
        ChatBaseten(**kwargs)


def test_chat_baseten_rejects_invalid_n() -> None:
    """Test ChatBaseten validates `n`."""
    with pytest.raises(ValueError, match="n must be at least 1"):
        ChatBaseten(
            model="zai-org/GLM-5.2",
            baseten_api_key=SecretStr("test_key"),
            n=0,
        )


def test_chat_baseten_rejects_streaming_with_n_gt_1() -> None:
    """Test ChatBaseten validates streaming with `n > 1`."""
    with pytest.raises(ValueError, match="n must be 1 when streaming"):
        ChatBaseten(
            model="zai-org/GLM-5.2",
            baseten_api_key=SecretStr("test_key"),
            n=2,
            streaming=True,
        )


def test_chat_baseten_dedicated_model_url() -> None:
    """Test ChatBaseten with dedicated model URL."""
    chat = ChatBaseten(
        model="custom-model",
        model_url="https://model-123.api.baseten.co/environments/production/predict",
        baseten_api_key=SecretStr("test_key"),
    )

    # Should use the dedicated URL, converted to OpenAI-compatible format
    expected_base_url = (
        "https://model-123.api.baseten.co/environments/production/sync/v1"
    )
    assert str(chat.root_client.base_url).rstrip("/") == expected_base_url


def test_chat_baseten_model_url_wins_over_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `model_url` overrides `BASETEN_BASE_URL` from the env."""
    monkeypatch.setenv("BASETEN_BASE_URL", "https://env.example/v1")
    chat = ChatBaseten(
        model="custom-model",
        model_url="https://model-123.api.baseten.co/environments/production/predict",
        baseten_api_key=SecretStr("test_key"),
    )

    # The dedicated model URL should win over the env-sourced base URL
    expected_base_url = (
        "https://model-123.api.baseten.co/environments/production/sync/v1"
    )
    assert str(chat.root_client.base_url).rstrip("/") == expected_base_url


def test_chat_baseten_dedicated_model_url_only() -> None:
    """Test ChatBaseten with only dedicated model URL (no model parameter)."""
    chat = ChatBaseten(
        model_url="https://model-456.api.baseten.co/environments/production/sync/v1",
        baseten_api_key=SecretStr("test_key"),
    )

    # Should use the dedicated URL
    expected_base_url = (
        "https://model-456.api.baseten.co/environments/production/sync/v1"
    )
    assert str(chat.root_client.base_url).rstrip("/") == expected_base_url

    # Should extract model name from URL
    assert chat.model_name == "model-456"


def test_chat_baseten_model_apis_default() -> None:
    """Test ChatBaseten uses Model APIs by default."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )

    # Should use the default Model APIs base URL
    expected_base_url = "https://inference.baseten.co/v1"
    assert str(chat.root_client.base_url).rstrip("/") == expected_base_url


def test_chat_baseten_default_model_name() -> None:
    """Test ChatBaseten uses default model name when none provided."""
    chat = ChatBaseten(
        baseten_api_key=SecretStr("test_key"),
    )
    assert chat.model_name == "zai-org/GLM-5.2"


def test_create_chat_result_with_reasoning_content() -> None:
    """Test that reasoning_content is extracted from responses."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    mock_message = MagicMock()
    mock_message.content = "Main content"
    mock_message.reasoning_content = "This is the reasoning content"
    mock_message.role = "assistant"
    mock_response = MockOpenAIResponse(
        choices=[MagicMock(message=mock_message)],
        error=None,
    )

    result = chat._create_chat_result(mock_response)

    assert (
        result.generations[0].message.additional_kwargs.get("reasoning_content")
        == "This is the reasoning content"
    )


def test_convert_chunk_with_reasoning_content() -> None:
    """Test that reasoning_content is extracted from streaming chunks."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    chunk: dict[str, Any] = {
        "choices": [
            {
                "delta": {
                    "content": "Main content",
                    "reasoning_content": "Streaming reasoning content",
                },
            },
        ],
    }

    chunk_result = chat._convert_chunk_to_generation_chunk(
        chunk,
        AIMessageChunk,
        None,
    )

    if chunk_result is None:
        msg = "Expected chunk_result not to be None"
        raise AssertionError(msg)

    assert (
        chunk_result.message.additional_kwargs.get("reasoning_content")
        == "Streaming reasoning content"
    )


def test_convert_chunk_strips_usage_from_content_chunks() -> None:
    """Test that cumulative usage is stripped from content chunks."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    chunk: dict[str, Any] = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
            },
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 1,
            "total_tokens": 13,
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "audio_tokens": 0,
                "reasoning_tokens": 0,
            },
        },
    }

    chunk_result = chat._convert_chunk_to_generation_chunk(
        chunk,
        AIMessageChunk,
        None,
    )

    if chunk_result is None:
        msg = "Expected chunk_result not to be None"
        raise AssertionError(msg)

    assert isinstance(chunk_result.message, AIMessageChunk)
    assert chunk_result.message.content == "Hello"
    assert chunk_result.message.usage_metadata is None


def test_convert_chunk_keeps_usage_for_usage_only_chunks() -> None:
    """Test that usage-only chunks retain their final usage metadata."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    chunk: dict[str, Any] = {
        "choices": [],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
            },
            "completion_tokens_details": {
                "audio_tokens": 0,
                "reasoning_tokens": 0,
            },
        },
    }

    chunk_result = chat._convert_chunk_to_generation_chunk(
        chunk,
        AIMessageChunk,
        None,
    )

    if chunk_result is None:
        msg = "Expected chunk_result not to be None"
        raise AssertionError(msg)

    assert isinstance(chunk_result.message, AIMessageChunk)
    assert chunk_result.message.usage_metadata is not None
    assert chunk_result.message.usage_metadata["input_tokens"] == 12
    assert chunk_result.message.usage_metadata["output_tokens"] == 5
    assert chunk_result.message.usage_metadata["total_tokens"] == 17


def test_stream_usage_aggregation_uses_only_final_usage_chunk() -> None:
    """Test a finish chunk before usage-only metadata does not double-count."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    raw_chunks: list[dict[str, Any]] = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "Hello",
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "total_tokens": 13,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cached_tokens": 0,
                },
                "completion_tokens_details": {
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": " there",
                    },
                    "finish_reason": None,
                },
            ],
            "model": "zai-org/GLM-5.2",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cached_tokens": 0,
                },
                "completion_tokens_details": {
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": None,
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cached_tokens": 0,
                },
                "completion_tokens_details": {
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        },
    ]

    response = MagicMock()
    response.__enter__.return_value = iter(raw_chunks)
    client = MagicMock()
    client.create.return_value = response
    chat.client = client

    full: AIMessageChunk | None = None
    usage_chunks = 0
    for chunk_result in chat._stream([HumanMessage(content="Hello")]):
        message = chunk_result.message
        if not isinstance(message, AIMessageChunk):
            msg = "Expected AIMessageChunk"
            raise AssertionError(msg)
        usage_chunks += message.usage_metadata is not None
        full = message if full is None else full + message

    if full is None:
        msg = "Expected aggregated chunk"
        raise AssertionError(msg)

    assert full.content == "Hello there"
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] == 12
    assert full.usage_metadata["output_tokens"] == 5
    assert full.usage_metadata["total_tokens"] == 17
    assert usage_chunks == 1


def test_stream_usage_aggregation_without_trailing_usage_chunk() -> None:
    """Test usage survives when the stream ends without a usage-only chunk.

    GLM-5.3-Flash-style stream: every content chunk carries cumulative usage
    and no trailing usage-only chunk arrives, so the totals must be emitted
    once the iterator is exhausted.
    """
    chat = ChatBaseten(
        model="zai-org/GLM-5.3-Flash",
        baseten_api_key=SecretStr("test_key"),
    )
    raw_chunks: list[dict[str, Any]] = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "Hello",
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 1,
                "total_tokens": 13,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cached_tokens": 0,
                },
                "completion_tokens_details": {
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": " there",
                    },
                    "finish_reason": None,
                },
            ],
            "model": "zai-org/GLM-5.3-Flash",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cached_tokens": 0,
                },
                "completion_tokens_details": {
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": None,
        },
    ]

    response = MagicMock()
    response.__enter__.return_value = iter(raw_chunks)
    client = MagicMock()
    client.create.return_value = response
    chat.client = client

    full: AIMessageChunk | None = None
    for chunk_result in chat._stream([HumanMessage(content="Hello")]):
        message = chunk_result.message
        if not isinstance(message, AIMessageChunk):
            msg = "Expected AIMessageChunk"
            raise AssertionError(msg)
        full = message if full is None else full + message

    if full is None:
        msg = "Expected aggregated chunk"
        raise AssertionError(msg)

    assert full.content == "Hello there"
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] == 12
    assert full.usage_metadata["output_tokens"] == 5
    assert full.usage_metadata["total_tokens"] == 17


def test_overlapping_streams_keep_usage_separate() -> None:
    """Test concurrent streams on one model do not share token usage state."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.3-Flash",
        baseten_api_key=SecretStr("test_key"),
    )
    first_chunks = [
        {
            "choices": [{"delta": {"content": "first"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        },
        {
            "choices": [{"delta": {"content": None}, "finish_reason": "stop"}],
            "usage": None,
        },
    ]
    second_chunks = [
        {
            "choices": [{"delta": {"content": "second"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
            },
        },
        {
            "choices": [{"delta": {"content": None}, "finish_reason": "stop"}],
            "usage": None,
        },
    ]
    responses = []
    for chunks in (first_chunks, second_chunks):
        response = MagicMock()
        response.__enter__.return_value = iter(chunks)
        responses.append(response)
    client = MagicMock()
    client.create.side_effect = responses
    chat.client = client

    first_stream = chat._stream([HumanMessage(content="first")])
    second_stream = chat._stream([HumanMessage(content="second")])
    first_results = [next(first_stream)]
    second_results = [next(second_stream)]
    first_results.extend(first_stream)
    second_results.extend(second_stream)

    def aggregate(results: list[ChatGenerationChunk]) -> AIMessageChunk:
        full: AIMessageChunk | None = None
        for result in results:
            message = result.message
            if not isinstance(message, AIMessageChunk):
                msg = "Expected AIMessageChunk"
                raise AssertionError(msg)
            full = message if full is None else full + message
        if full is None:
            msg = "Expected streamed chunks"
            raise AssertionError(msg)
        return full

    first_full = aggregate(first_results)
    second_full = aggregate(second_results)

    assert first_full.usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 1,
        "total_tokens": 11,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert second_full.usage_metadata == {
        "input_tokens": 20,
        "output_tokens": 2,
        "total_tokens": 22,
        "input_token_details": {},
        "output_token_details": {},
    }


async def test_overlapping_async_streams_keep_usage_separate() -> None:
    """Test concurrent async tasks on one model do not share token usage state."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.3-Flash",
        baseten_api_key=SecretStr("test_key"),
    )

    def make_chunks(name: str, prompt_tokens: int, completion_tokens: int) -> list:
        return [
            {
                "choices": [{"delta": {"content": name}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
            {
                "choices": [{"delta": {"content": None}, "finish_reason": "stop"}],
                "usage": None,
            },
        ]

    first_chunks = make_chunks("first", 10, 1)
    second_chunks = make_chunks("second", 20, 2)

    class FakeAsyncStream:
        def __init__(self, chunks: list) -> None:
            self._chunks = chunks

        async def __aenter__(self) -> "FakeAsyncStream":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def __aiter__(self) -> "FakeAsyncStream":
            return self

        async def __anext__(self) -> dict:
            if not self._chunks:
                raise StopAsyncIteration
            # Yield to the event loop so the other task interleaves here.
            await asyncio.sleep(0)
            return self._chunks.pop(0)

    client = MagicMock()
    client.create = AsyncMock(
        side_effect=[FakeAsyncStream(first_chunks), FakeAsyncStream(second_chunks)]
    )
    chat.async_client = client

    async def collect(stream: Any) -> list[ChatGenerationChunk]:
        return [chunk async for chunk in stream]

    first_results, second_results = await asyncio.gather(
        collect(chat._astream([HumanMessage(content="first")])),
        collect(chat._astream([HumanMessage(content="second")])),
    )

    def aggregate(results: list[ChatGenerationChunk]) -> AIMessageChunk:
        full: AIMessageChunk | None = None
        for result in results:
            message = result.message
            if not isinstance(message, AIMessageChunk):
                msg = "Expected AIMessageChunk"
                raise AssertionError(msg)
            full = message if full is None else full + message
        if full is None:
            msg = "Expected streamed chunks"
            raise AssertionError(msg)
        return full

    first_full = aggregate(first_results)
    second_full = aggregate(second_results)

    assert first_full.usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 1,
        "total_tokens": 11,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert second_full.usage_metadata == {
        "input_tokens": 20,
        "output_tokens": 2,
        "total_tokens": 22,
        "input_token_details": {},
        "output_token_details": {},
    }


def test_streaming_tool_call_chunks_with_unique_ids_merge_correctly() -> None:
    """Test that tool-call chunks with unique IDs merge into one call."""
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    raw_chunks: list[dict[str, Any]] = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "ls", "arguments": ""},
                                "type": "function",
                            }
                        ],
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-2",
                                "function": {"arguments": '{"path": '},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-3",
                                "function": {"arguments": '"/tmp"'},
                            }
                        ],
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-4",
                                "function": {"arguments": "}"},
                            }
                        ],
                    },
                }
            ],
        },
    ]

    full: AIMessageChunk | None = None
    for raw_chunk in raw_chunks:
        chunk_result = chat._convert_chunk_to_generation_chunk(
            raw_chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        message = chunk_result.message
        if not isinstance(message, AIMessageChunk):
            msg = "Expected AIMessageChunk"
            raise AssertionError(msg)
        full = message if full is None else full + message

    if full is None:
        msg = "Expected aggregated chunk"
        raise AssertionError(msg)

    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["name"] == "ls"
    assert full.tool_calls[0]["args"] == {"path": "/tmp"}  # noqa: S108
    assert full.invalid_tool_calls == []


def test_streaming_tool_call_multi_delta_per_event_merge_correctly() -> None:
    """Test that multiple tool-call deltas in one SSE event are consolidated.

    TRT-LLM may pack multiple deltas for the same tool call into a single SSE
    event. Without consolidation, each becomes a separate `ToolCallChunk` and
    subsequent cross-event merging breaks.
    """
    chat = ChatBaseten(
        model="zai-org/GLM-5.2",
        baseten_api_key=SecretStr("test_key"),
    )
    raw_chunks: list[dict[str, Any]] = [
        # Event 1: two deltas for index=0 in the same SSE event
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "",
                                },
                                "type": "function",
                            },
                            {
                                "index": 0,
                                "id": "call-2",
                                "function": {"arguments": '{"path": '},
                            },
                        ],
                    },
                }
            ],
        },
        # Event 2: single continuation delta
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-3",
                                "function": {"arguments": '"/tmp/test.txt"'},
                            }
                        ],
                    },
                }
            ],
        },
        # Event 3: closing brace + another delta in same event
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-4",
                                "function": {"arguments": "}"},
                            },
                        ],
                    },
                }
            ],
        },
    ]

    full: AIMessageChunk | None = None
    for raw_chunk in raw_chunks:
        chunk_result = chat._convert_chunk_to_generation_chunk(
            raw_chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        message = chunk_result.message
        if not isinstance(message, AIMessageChunk):
            msg = "Expected AIMessageChunk"
            raise AssertionError(msg)
        full = message if full is None else full + message

    if full is None:
        msg = "Expected aggregated chunk"
        raise AssertionError(msg)

    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["name"] == "read_file"
    assert full.tool_calls[0]["args"] == {"path": "/tmp/test.txt"}  # noqa: S108
    assert full.tool_calls[0]["id"] == "call-1"
    assert full.invalid_tool_calls == []


def test_reasoning_effort_is_included_in_payload() -> None:
    """Test that reasoning_effort is passed through in request payloads."""
    chat = ChatBaseten(
        model="openai/gpt-oss-120b",
        baseten_api_key=SecretStr("test_key"),
        reasoning_effort="high",
    )

    payload = chat._get_request_payload([("user", "What is the sum of 2 and 2?")])

    assert payload["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Reasoning-history round-tripping
# ---------------------------------------------------------------------------

_CallMode = Literal["invoke", "ainvoke", "stream", "astream"]
_CALL_MODES: tuple[_CallMode, ...] = ("invoke", "ainvoke", "stream", "astream")


class _FakeBasetenServer:
    """Scripted chat-completions endpoint that records outgoing request bodies.

    Each queued reply is served as a JSON body or as SSE chunks depending on the
    request's `stream` flag, so the same script exercises every call path.
    """

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.authorizations: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.authorizations.append(request.headers["authorization"])
        reply = self.replies.pop(0)
        if body.get("stream"):
            return httpx.Response(
                200,
                content=self._sse(reply),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=self._completion(reply))

    @staticmethod
    def _completion(reply: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply["content"]}
        if "reasoning" in reply:
            message["reasoning_content"] = "".join(reply["reasoning"])
        if reply.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in reply["tool_calls"]
            ]
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if reply.get("tool_calls")
                    else "stop",
                }
            ],
        }

    @staticmethod
    def _sse(reply: dict[str, Any]) -> bytes:
        deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
        # Reasoning arrives split across several deltas to exercise aggregation.
        deltas += [{"reasoning_content": part} for part in reply.get("reasoning", [])]
        if reply["content"]:
            deltas.append({"content": reply["content"]})
        for index, tc in enumerate(reply.get("tool_calls", [])):
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                    ]
                }
            )
        finish = "tool_calls" if reply.get("tool_calls") else "stop"
        events = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish if i == len(deltas) - 1 else None,
                    }
                ],
            }
            for i, delta in enumerate(deltas)
        ]
        # The stream ends at EOF without a `[DONE]` sentinel; the OpenAI SDK
        # handles both, and EOF avoids unawaited-`aclose` warnings in async tests.
        lines = [f"data: {json.dumps(event)}\n\n" for event in events]
        return "".join(lines).encode()


_SYNTHETIC_TEST_KEY = SecretStr("synthetic-test-key")


def _chat_with_server(
    server: _FakeBasetenServer,
    baseten_api_key: SecretStr
    | Callable[[], str]
    | Callable[[], Awaitable[str]] = _SYNTHETIC_TEST_KEY,
) -> ChatBaseten:
    transport = httpx.MockTransport(server.handle)
    return ChatBaseten(
        model="moonshotai/Kimi-K3.1-Pro",
        baseten_api_key=baseten_api_key,
        temperature=0.3,
        max_retries=0,
        stream_usage=False,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
    )


async def _call(
    chat: ChatBaseten | Runnable, messages: list[BaseMessage], mode: _CallMode
) -> AIMessage:
    if mode == "invoke":
        result = chat.invoke(messages)
    elif mode == "ainvoke":
        result = await chat.ainvoke(messages)
    elif mode == "stream":
        chunks = list(chat.stream(messages))
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
    else:
        chunks = [chunk async for chunk in chat.astream(messages)]
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
    assert isinstance(result, AIMessage)
    return result


@pytest.mark.parametrize("mode", _CALL_MODES)
@pytest.mark.parametrize(
    "reasoning_parts",
    [["Let me ", "think ", "about it."], [""]],
    ids=["text", "empty"],
)
async def test_reasoning_content_round_trips_to_next_request(
    mode: _CallMode, reasoning_parts: list[str]
) -> None:
    """Reasoning from a response is sent back on the next request's history."""
    expected_reasoning = "".join(reasoning_parts)
    server = _FakeBasetenServer(
        [
            {"content": "4", "reasoning": reasoning_parts},
            {"content": "8"},
        ]
    )
    chat = _chat_with_server(server)

    history: list[BaseMessage] = [HumanMessage("What is 2 + 2?")]
    ai_message = await _call(chat, history, mode)
    assert ai_message.additional_kwargs["reasoning_content"] == expected_reasoning

    history += [ai_message, HumanMessage("Now double it.")]
    await _call(chat, history, mode)

    sent = server.requests[1]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][1] == {
        "role": "assistant",
        "content": "4",
        "reasoning_content": expected_reasoning,
    }
    # Request parameters are untouched.
    assert sent["model"] == "moonshotai/Kimi-K3.1-Pro"
    assert sent["temperature"] == 0.3
    assert bool(sent.get("stream")) is mode.endswith("stream")


@pytest.mark.parametrize("mode", _CALL_MODES)
async def test_reasoning_content_round_trips_with_tool_call_continuation(
    mode: _CallMode,
) -> None:
    """Reasoning is preserved alongside tool calls and their IDs across turns."""

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    server = _FakeBasetenServer(
        [
            {
                "content": "",
                "reasoning": ["I should ", "call the tool."],
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    }
                ],
            },
            {"content": "It is sunny in Paris.", "reasoning": ["Tool said sunny."]},
            {"content": "You're welcome."},
        ]
    )
    chat = _chat_with_server(server)
    chat_with_tools = chat.bind_tools([get_weather])

    history: list[BaseMessage] = [HumanMessage("Weather in Paris?")]
    tool_turn = await _call(chat_with_tools, history, mode)
    assert tool_turn.tool_calls[0]["id"] == "call_abc123"
    history += [
        tool_turn,
        ToolMessage("Sunny in Paris", tool_call_id="call_abc123"),
    ]

    answer = await _call(chat_with_tools, history, mode)
    history += [answer, HumanMessage("Thanks!")]
    await _call(chat_with_tools, history, mode)

    # Tool continuation request.
    continuation = server.requests[1]["messages"]
    assert [m["role"] for m in continuation] == ["user", "assistant", "tool"]
    assert continuation[1]["reasoning_content"] == "I should call the tool."
    assert continuation[1]["tool_calls"] == [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert continuation[2]["tool_call_id"] == "call_abc123"
    assert server.requests[1]["tools"][0]["function"]["name"] == "get_weather"

    # Later user turn keeps reasoning on *every* earlier assistant message.
    later = server.requests[2]["messages"]
    assert [m["role"] for m in later] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert later[1]["reasoning_content"] == "I should call the tool."
    assert later[1]["tool_calls"][0]["id"] == "call_abc123"
    assert later[2]["tool_call_id"] == "call_abc123"
    assert later[3]["reasoning_content"] == "Tool said sunny."
    assert later[3]["content"] == "It is sunny in Paris."


@pytest.mark.parametrize("mode", _CALL_MODES)
async def test_history_without_reasoning_is_unchanged(mode: _CallMode) -> None:
    """Assistant messages without reasoning serialize exactly as before."""
    server = _FakeBasetenServer([{"content": "4"}, {"content": "8"}])
    chat = _chat_with_server(server)

    history: list[BaseMessage] = [HumanMessage("What is 2 + 2?")]
    ai_message = await _call(chat, history, mode)
    assert "reasoning_content" not in ai_message.additional_kwargs

    history += [ai_message, HumanMessage("Now double it.")]
    await _call(chat, history, mode)

    assert server.requests[1]["messages"] == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Now double it."},
    ]


def test_request_payload_forwards_only_reasoning_content_without_mutation() -> None:
    """Only `reasoning_content` is forwarded; caller messages are not mutated."""
    chat = ChatBaseten(
        model="moonshotai/Kimi-K3.1-Pro",
        baseten_api_key=SecretStr("synthetic-test-key"),
    )
    ai_message = AIMessage(
        content="4",
        additional_kwargs={"reasoning_content": "2 + 2 = 4", "other_field": "x"},
    )
    no_reasoning = AIMessage(
        content="ok", additional_kwargs={"reasoning_content": None}
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "What is 2 + 2?"),
            MessagesPlaceholder("history"),
        ]
    ).invoke({"history": [ai_message, no_reasoning, HumanMessage("Thanks")]})
    before = prompt.model_copy(deep=True)

    # Non-list input goes through the standard input normalization.
    payload = chat._get_request_payload(prompt, stop=["END"])

    assert prompt == before
    assert ai_message.additional_kwargs == {
        "reasoning_content": "2 + 2 = 4",
        "other_field": "x",
    }
    assert payload["stop"] == ["END"]
    assert payload["messages"] == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "2 + 2 = 4"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Thanks"},
    ]


@pytest.mark.parametrize(
    ("model_provider", "expected_reasoning"),
    [("baseten", "2 + 2 = 4"), (None, "2 + 2 = 4"), ("deepseek", None)],
    ids=["baseten", "untagged", "other-provider"],
)
def test_request_payload_forwards_reasoning_only_from_baseten_or_untagged(
    model_provider: str | None, expected_reasoning: str | None
) -> None:
    """Reasoning from another provider's messages is not replayed to Baseten."""
    chat = ChatBaseten(
        model="moonshotai/Kimi-K3.1-Pro",
        baseten_api_key=SecretStr("synthetic-test-key"),
    )
    response_metadata = {"model_provider": model_provider} if model_provider else {}
    ai_message = AIMessage(
        content="4",
        additional_kwargs={"reasoning_content": "2 + 2 = 4"},
        response_metadata=response_metadata,
    )

    payload = chat._get_request_payload([HumanMessage("2 + 2?"), ai_message])

    expected: dict[str, Any] = {"role": "assistant", "content": "4"}
    if expected_reasoning is not None:
        expected["reasoning_content"] = expected_reasoning
    assert payload["messages"][1] == expected


# ---------------------------------------------------------------------------
# Function API keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _CALL_MODES)
async def test_function_api_key_is_called_for_each_request(mode: _CallMode) -> None:
    """A function key is called per request, so a refreshed token is sent."""
    server = _FakeBasetenServer([{"content": "4"}, {"content": "8"}])
    current = {"api_key": "token-1"}

    def get_token() -> str:
        return current["api_key"]

    chat = _chat_with_server(server, baseten_api_key=get_token)

    await _call(chat, [HumanMessage("What is 2 + 2?")], mode)
    current["api_key"] = "token-2"
    await _call(chat, [HumanMessage("What is 4 + 4?")], mode)

    assert server.authorizations == ["Bearer token-1", "Bearer token-2"]


@pytest.mark.parametrize("mode", ["ainvoke", "astream"])
async def test_async_function_api_key_is_called_for_each_request(
    mode: _CallMode,
) -> None:
    """An async function key is awaited per request on async calls."""
    server = _FakeBasetenServer([{"content": "4"}, {"content": "8"}])
    current = {"api_key": "token-1"}

    async def get_token() -> str:
        return current["api_key"]

    chat = _chat_with_server(server, baseten_api_key=get_token)

    await _call(chat, [HumanMessage("What is 2 + 2?")], mode)
    current["api_key"] = "token-2"
    await _call(chat, [HumanMessage("What is 4 + 4?")], mode)

    assert server.authorizations == ["Bearer token-1", "Bearer token-2"]


@pytest.mark.parametrize("mode", ["invoke", "stream"])
async def test_async_function_api_key_rejects_sync_calls(mode: _CallMode) -> None:
    """An async function key has no sync client, so sync calls fail clearly."""
    server = _FakeBasetenServer([{"content": "4"}])

    async def get_token() -> str:
        return "token-1"

    chat = _chat_with_server(server, baseten_api_key=get_token)

    with pytest.raises(ValueError, match="Sync client is not available"):
        await _call(chat, [HumanMessage("What is 2 + 2?")], mode)
    assert server.authorizations == []
