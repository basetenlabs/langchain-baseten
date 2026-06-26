"""Shared fixtures for unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _clear_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear base-URL env vars so tests start from a known, clean environment.

    Without this, a `BASETEN_BASE_URL` or `BASETEN_API_BASE` value exported in
    the developer's or CI's shell would shadow the default and silently break
    tests that assert on the default Model APIs base URL.
    """
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)
    monkeypatch.delenv("BASETEN_API_BASE", raising=False)
