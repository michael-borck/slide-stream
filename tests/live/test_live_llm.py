"""Live LLM: the configured narration model actually responds.

Skips when no LLM is configured (provider 'none').
    uv run pytest tests/live/test_live_llm.py --run-live -q
"""

import pytest
from rich.console import Console

from slide_stream.llm import get_llm_client, query_llm


def test_live_llm_responds(live_config):
    llm = live_config["providers"].get("llm", {})
    provider = llm.get("provider", "none")
    if provider == "none":
        pytest.skip("no LLM provider configured (providers.llm.provider: none)")

    client = get_llm_client(provider, base_url=llm.get("base_url"))
    reply = query_llm(
        client,
        provider,
        "Reply with exactly one word, lowercase, no punctuation: pineapple",
        Console(),
        model=llm.get("model"),
    )
    assert reply, f"{provider} returned no text"
    assert "pineapple" in reply.lower(), f"unexpected LLM reply: {reply!r}"
    print(f"\nLLM provider exercised: {provider} (model={llm.get('model')})")
