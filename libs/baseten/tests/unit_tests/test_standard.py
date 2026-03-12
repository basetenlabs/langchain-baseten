"""Standard LangChain interface tests."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_baseten import ChatBaseten


class TestBasetenStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatBaseten

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "zai-org/GLM-5",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "BASETEN_API_KEY": "api_key",
            },
            {
                "model": "zai-org/GLM-5",
            },
            {
                "baseten_api_key": "api_key",
            },
        )
