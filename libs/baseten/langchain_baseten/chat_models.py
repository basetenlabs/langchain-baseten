"""Baseten chat wrapper."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_baseten.data._profiles import _PROFILES

if TYPE_CHECKING:
    from langchain_core.language_models import ModelProfile, ModelProfileRegistry

DEFAULT_API_BASE = "https://inference.baseten.co/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _normalize_model_url(url: str) -> str:
    """Normalize dedicated model URL to OpenAI-compatible /sync/v1 format."""
    if url.endswith("/predict"):
        return url.replace("/predict", "/sync/v1")
    if url.endswith("/sync"):
        return f"{url}/v1"
    if not url.endswith("/v1"):
        return f"{url.rstrip('/')}/v1"
    return url


def _normalize_tool_call_chunks(chunk: dict[str, Any]) -> dict[str, Any]:
    """Consolidate and fix tool-call deltas so they merge correctly downstream.

    Baseten models served by TensorRT-LLM exhibit two quirks:

    1. A single SSE event may contain multiple tool-call deltas for the same
        logical tool call (same `index`). LangChain turns each delta into a
        separate `ToolCallChunk`, and `merge_lists` never recombines entries
        that originated from the same event.
    2. Every delta gets a unique `id`. `merge_lists` refuses to merge chunks
        whose non-null IDs differ, so cross-event merging also breaks.

    This function folds same-index deltas within an event into one entry and
    nulls out IDs on continuation deltas (those without `function.name`) so
    that `merge_lists` can merge them by `index` alone.
    """
    choices = chunk.get("choices")
    if not choices:
        return chunk
    delta = choices[0].get("delta", {})
    tool_calls = delta.get("tool_calls")
    if not tool_calls:
        return chunk

    # 1. Consolidate same-index deltas within this event
    by_index: dict[int, dict[str, Any]] = {}
    for tc in tool_calls:
        idx = tc.get("index")
        if idx is None:
            continue
        if idx not in by_index:
            by_index[idx] = {**tc}
        else:
            existing = by_index[idx]
            # Merge function fields
            fn_existing = existing.get("function") or {}
            fn_new = tc.get("function") or {}
            merged_fn = {**fn_existing}
            if fn_new.get("name"):
                merged_fn["name"] = fn_new["name"]
            if "arguments" in fn_new:
                merged_fn["arguments"] = (
                    fn_existing.get("arguments", "") + fn_new["arguments"]
                )
            existing["function"] = merged_fn
            # Keep the first non-None id
            if existing.get("id") is None and tc.get("id") is not None:
                existing["id"] = tc["id"]

    # 2. Null out IDs on continuation deltas (no function name)
    new_tool_calls = []
    for tc in by_index.values():
        fn = tc.get("function") or {}
        if not fn.get("name") and tc.get("id") is not None:
            tc = {**tc, "id": None}
        new_tool_calls.append(tc)

    if new_tool_calls == tool_calls:
        return chunk

    # Shallow-copy only the path we changed
    new_chunk = chunk.copy()
    new_choices = [choices[0].copy()]
    new_delta = delta.copy()
    new_delta["tool_calls"] = new_tool_calls
    new_choices[0]["delta"] = new_delta
    new_chunk["choices"] = new_choices
    return new_chunk


def _normalize_stream_usage_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize Baseten stream usage to match OpenAI's final-usage-chunk semantics.

    Baseten currently returns cumulative token usage on every streamed content chunk
    and repeats the final totals in a trailing usage-only chunk. LangChain's chunk
    aggregation sums usage metadata across chunks, so we keep usage only on the
    usage-only chunk and strip it from chunks that also contain choices.
    """
    if chunk.get("usage") and chunk.get("choices"):
        normalized_chunk = chunk.copy()
        normalized_chunk.pop("usage", None)
        return normalized_chunk
    return chunk


class ChatBaseten(BaseChatOpenAI):
    r"""Baseten chat model integration.

    ### Setup

    Install `langchain-baseten` and set the `BASETEN_API_KEY` environment
    variable.

    ```bash
    pip install -U langchain-baseten
    export BASETEN_API_KEY="your-api-key"
    ```

    ### Key init args

    - `model`: Name of Baseten model to use. Optional for dedicated models.
    - `max_tokens`: Max number of tokens to generate.
    - `baseten_api_key`: Baseten API key.

        If not passed in, it is read from `BASETEN_API_KEY`.
    - `baseten_api_base`: Base URL path for API requests for Model APIs.
    - `model_url`: Optional dedicated model URL for deployed models.

        If provided, it overrides `baseten_api_base`. Supports `/predict`,
        `/sync`, or `/sync/v1` endpoints.
    - `request_timeout`: Timeout for requests to the Baseten completion API.
    - `max_retries`: Maximum number of retries to make when generating.

    ### Instantiate

    ```python
    from langchain_baseten import ChatBaseten

    # Option 1: Use Model APIs with model slug (recommended)
    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        temperature=0.7,
        max_tokens=256,
        # Uses default baseten_api_base for Model APIs
    )

    # Option 2: Use dedicated model URL for deployed models
    chat = ChatBaseten(
        model_url="https://model-<id>.api.baseten.co/environments/production/predict",
        temperature=0.7,
        max_tokens=256,
        # model parameter is optional for dedicated models
    )
    ```

    ### Invoke

    ```python
    messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to French.",
        ),
        ("human", "I love programming."),
    ]
    chat.invoke(messages)
    ```

    ```python
    AIMessage(
        content="J'adore la programmation.",
        response_metadata={
            "token_usage": {
                "completion_tokens": 5,
                "prompt_tokens": 31,
                "total_tokens": 36,
            },
            "model_name": "deepseek-ai/DeepSeek-V3-0324",
            "finish_reason": "stop",
        },
    )
    ```

    ### Stream

    ```python
    for chunk in chat.stream(messages):
        print(chunk.content, end="")
    ```

    ```text
    J'adore la programmation.
    ```

    ### Async

    ```python
    await chat.ainvoke(messages)

    # stream:
    # async for chunk in chat.astream(messages):
    #     print(chunk.content, end="")

    # batch:
    # await chat.abatch([messages])
    ```

    ### Tool calling

    ```python
    from pydantic import BaseModel, Field


    class GetWeather(BaseModel):
        '''Get the current weather in a given location'''

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )


    class GetPopulation(BaseModel):
        '''Get the current population in a given location'''

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )


    chat_with_tools = chat.bind_tools([GetWeather, GetPopulation])
    ai_msg = chat_with_tools.invoke(
        "Which city is hotter today and which is bigger: LA or NY?"
    )
    ai_msg.tool_calls
    ```

    ```python
    [
        {
            "name": "GetWeather",
            "args": {"location": "Los Angeles, CA"},
            "id": "call_1",
        },
        {
            "name": "GetWeather",
            "args": {"location": "New York, NY"},
            "id": "call_2",
        },
        {
            "name": "GetPopulation",
            "args": {"location": "Los Angeles, CA"},
            "id": "call_3",
        },
        {
            "name": "GetPopulation",
            "args": {"location": "New York, NY"},
            "id": "call_4",
        },
    ]
    ```

    ### Structured output

    ```python
    from typing import Optional

    from pydantic import BaseModel, Field


    class Joke(BaseModel):
        '''Joke to tell user.'''

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


    structured_chat = chat.with_structured_output(Joke)
    structured_chat.invoke("Tell me a joke about cats")
    ```

    ```python
    Joke(
        setup="Why was the cat sitting on the computer?",
        punchline="To keep an eye on the mouse!",
        rating=None,
    )
    ```

    ### JSON mode

    ```python
    json_chat = chat.bind(response_format={"type": "json_object"})
    ai_msg = json_chat.invoke(
        "Return a JSON object with key 'random_ints' and a value of 10 "
        "random ints in [0-99]"
    )
    ai_msg.content
    ```

    ```python
    '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'
    ```

    ### Response metadata

    ```python
    ai_msg = chat.invoke(messages)
    ai_msg.response_metadata
    ```

    ```python
    {
        "token_usage": {
            "completion_tokens": 5,
            "prompt_tokens": 28,
            "total_tokens": 33,
        },
        "model_name": "deepseek-ai/DeepSeek-V3-0324",
        "finish_reason": "stop",
    }
    ```
    """

    model_name: str = Field(default="", alias="model")
    """Model name to use.

    Optional for dedicated models.
    """

    baseten_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("BASETEN_API_KEY", default=None),
    )
    """Automatically inferred from env var `BASETEN_API_KEY` if not provided."""

    baseten_api_base: str = Field(default=DEFAULT_API_BASE, alias="base_url")
    """Base URL path for API requests.

    Leave as default for Model APIs, or provide dedicated model URL for
    dedicated deployments.
    """

    model_url: str | None = Field(default=None)
    """Optional dedicated model URL for deployed models.

    If provided, this will override `baseten_api_base`!

    Should be in format:
    `https://model-<id>.api.baseten.co/environments/production/predict or /sync/v1`
    """

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_client_config(cls, values: Any) -> Any:
        """Normalize OpenAI-style init args into Baseten-specific fields."""
        if not isinstance(values, dict):
            return values

        normalized = values.copy()

        if "openai_api_key" in normalized:
            msg = (
                "`openai_api_key` is not supported by ChatBaseten. "
                "Use `baseten_api_key` or `api_key` instead."
            )
            raise ValueError(msg)

        if "openai_api_base" in normalized:
            msg = (
                "`openai_api_base` is not supported by ChatBaseten. "
                "Use `baseten_api_base` or `base_url` instead."
            )
            raise ValueError(msg)

        baseten_api_key = normalized.pop("baseten_api_key", None)
        alias_api_key = normalized.pop("api_key", None)

        if alias_api_key is not None:
            if baseten_api_key is not None and alias_api_key != baseten_api_key:
                msg = (
                    "Received conflicting API key values. "
                    "Specify only one of `baseten_api_key` or `api_key`."
                )
                raise ValueError(msg)
            baseten_api_key = alias_api_key

        if baseten_api_key is not None:
            normalized["baseten_api_key"] = baseten_api_key

        baseten_api_base = normalized.pop("baseten_api_base", None)
        alias_api_base = normalized.pop("base_url", None)

        if alias_api_base is not None:
            if baseten_api_base is not None and alias_api_base != baseten_api_base:
                msg = (
                    "Received conflicting base URL values. "
                    "Specify only one of `baseten_api_base` or `base_url`."
                )
                raise ValueError(msg)
            baseten_api_base = alias_api_base

        if baseten_api_base is not None:
            normalized["baseten_api_base"] = baseten_api_base

        return normalized

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "baseten-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"baseten_api_key": "BASETEN_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "baseten"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        # Enable streaming usage metadata by default
        if self.stream_usage is not False:
            self.stream_usage = True

        # Resolve base URL
        if self.model_url:
            base_url = _normalize_model_url(self.model_url)
            # Extract model name from URL if not provided
            if not self.model_name:
                match = re.search(r"model-([a-zA-Z0-9]+)", self.model_url)
                self.model_name = (
                    f"model-{match.group(1)}" if match else "baseten-model"
                )
        else:
            base_url = self.baseten_api_base
            if not self.model_name:
                self.model_name = "deepseek-ai/DeepSeek-V3-0324"

        api_key = (
            self.baseten_api_key.get_secret_value()
            if isinstance(self.baseten_api_key, SecretStr)
            else self.baseten_api_key
        )

        if not api_key:
            msg = (
                "BASETEN_API_KEY must be set. "
                "You can pass it as `baseten_api_key=...` or "
                "set the environment variable `BASETEN_API_KEY`."
            )
            raise ValueError(msg)

        client_params: dict = {
            k: v
            for k, v in {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Create a `ChatResult`, adding Baseten provider metadata."""
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "baseten"

        choices = getattr(response, "choices", None)
        if choices and hasattr(choices[0].message, "reasoning_content"):
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = choices[
                0
            ].message.reasoning_content

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Convert a chunk, adding Baseten provider metadata."""
        chunk = _normalize_tool_call_chunks(chunk)
        chunk = _normalize_stream_usage_chunk(chunk)
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "baseten",
                }
                if (
                    reasoning_content := top.get("delta", {}).get("reasoning_content")
                ) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
        return generation_chunk
