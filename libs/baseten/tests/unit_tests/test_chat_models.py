"""Test ChatBaseten chat model."""

import asyncio
import os
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
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
