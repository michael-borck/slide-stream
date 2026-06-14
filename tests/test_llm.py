"""Tests for the LLM layer, including OpenAI-compatible (local) backends."""

import pytest
from rich.console import Console

from slide_stream.llm import get_llm_client, query_llm


def test_get_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("nope")


def test_get_llm_client_openai_compatible_uses_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = get_llm_client(
        "openai-compatible", base_url="http://localhost:8080/v1"
    )
    # The OpenAI SDK keeps the base_url on the client.
    assert "localhost:8080" in str(client.base_url)


def test_get_llm_client_openai_compatible_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://192.168.1.5:1234/v1")
    client = get_llm_client("openai-compatible")
    assert "192.168.1.5:1234" in str(client.base_url)


def test_get_llm_client_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        get_llm_client("openai-compatible")


def test_query_llm_openai_compatible_uses_chat_completions(mocker):
    """openai-compatible routes through the chat.completions path."""
    fake_message = mocker.MagicMock()
    fake_message.content = "a flowing narration script"
    fake_choice = mocker.MagicMock()
    fake_choice.message = fake_message
    fake_response = mocker.MagicMock()
    fake_response.choices = [fake_choice]

    fake_client = mocker.MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    result = query_llm(
        fake_client,
        "openai-compatible",
        "Convert these points to a script.",
        Console(),
        model="llama-3.1-8b-instruct",
    )

    assert result == "a flowing narration script"
    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["model"] == "llama-3.1-8b-instruct"


def test_query_llm_returns_none_on_error(mocker):
    """A backend failure is swallowed into None so the caller can degrade."""
    fake_client = mocker.MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("offline")

    result = query_llm(
        fake_client, "openai-compatible", "prompt", Console(), model="m"
    )
    assert result is None
