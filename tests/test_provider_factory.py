"""Tests for provider selection, fallback and strict mode in the factory."""

import pytest

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.base import StrictModeError
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.images import (
    TextImageProvider,
)
from slide_stream.providers.tts import (
    GTTSProvider,
    KokoroTTSProvider,
)


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


def _delenv_voicebox(monkeypatch):
    monkeypatch.delenv("VOICEBOX_BASE_URL", raising=False)
    monkeypatch.delenv("VOICEBOX_TOKEN", raising=False)


def test_factory_returns_configured_image_provider(config):
    config["providers"]["images"]["provider"] = "text"
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_falls_back_when_primary_unavailable(config, monkeypatch):
    # dalle3 needs an OpenAI key; without one it must fall back to text.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["providers"]["images"]["provider"] = "dalle3"
    config["providers"]["images"]["fallback"] = "text"
    config["api_keys"] = {}

    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_unknown_image_provider_falls_back_to_text(config):
    config["providers"]["images"]["provider"] = "does-not-exist"
    config["providers"]["images"]["fallback"] = "also-missing"
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


def test_factory_default_tts_falls_back_to_gtts_without_a_server(config, monkeypatch):
    """The default is voicebox, but with no server configured a bare install
    must still render — so the factory quietly returns gTTS."""
    _delenv_voicebox(monkeypatch)
    assert config["providers"]["tts"]["provider"] == "voicebox"
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_unconfigured_default_does_not_error(config, capsys, monkeypatch):
    """A zero-config user must not see a scary ❌; just a friendly hint."""
    _delenv_voicebox(monkeypatch)
    ProviderFactory.create_tts_provider(config)
    captured = capsys.readouterr()
    assert "not available" not in (captured.out + captured.err)


def test_factory_configured_but_broken_tts_is_an_error(config, capsys):
    """A voicebox with a base_url but a real failure is a genuine error."""
    config["providers"]["tts"]["base_url"] = "https://vb.org"
    # base_url set but no profile_id/voice_sample -> is_available() False, and
    # this counts as "set up but unusable", not "never configured".
    ProviderFactory.create_tts_provider(config)
    captured = capsys.readouterr()
    assert "not available" in (captured.out + captured.err)


def test_factory_unconfigured_default_uses_gtts_even_in_strict(config, monkeypatch):
    """Strict protects an explicitly-chosen provider; the unconfigured default
    voicebox was never chosen, so a bare strict install still renders on gTTS
    instead of aborting."""
    _delenv_voicebox(monkeypatch)
    config["settings"]["strict"] = True
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_env_configured_voicebox_is_not_silently_downgraded(
    config, monkeypatch, capsys
):
    """A voicebox configured purely via env vars counts as configured: a
    broken setup is reported as an error instead of quietly using gTTS."""
    monkeypatch.setenv("VOICEBOX_BASE_URL", "https://vb.org")
    # base_url alone (no profile_id/voice_sample) -> unavailable, a real error.
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)  # non-strict still falls back
    captured = capsys.readouterr()
    assert "not available" in (captured.out + captured.err)


def test_factory_env_configured_voicebox_aborts_in_strict(config, monkeypatch):
    """Strict mode must trigger for an env-var-configured (but broken)
    provider rather than silently downgrading to gTTS."""
    monkeypatch.setenv("VOICEBOX_BASE_URL", "https://vb.org")
    config["settings"]["strict"] = True
    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


def test_factory_configured_broken_voicebox_still_aborts_in_strict(config):
    """But a voicebox the user did set up (base_url present) must still abort
    under strict rather than silently swapping to gTTS."""
    config["settings"]["strict"] = True
    config["providers"]["tts"]["base_url"] = "https://vb.org"
    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


def test_factory_unknown_tts_falls_back_to_gtts(config):
    config["providers"]["tts"]["provider"] = "nope"
    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_check_provider_availability_reports_always_on_providers(config):
    availability = ProviderFactory.check_provider_availability(config)
    assert availability["images"]["text"] is True
    assert availability["tts"]["gtts"] is True


def test_provider_listings_contain_known_providers():
    assert "text" in ProviderFactory.list_image_providers()
    assert "dalle3" in ProviderFactory.list_image_providers()
    assert "gtts" in ProviderFactory.list_tts_providers()
    assert "elevenlabs" in ProviderFactory.list_tts_providers()


# --- Availability gating on API keys ---------------------------------------


def test_factory_strict_raises_for_unavailable_tts(config, monkeypatch):
    """In strict mode an unusable TTS provider aborts instead of using gTTS."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["settings"]["strict"] = True
    config["providers"]["tts"]["provider"] = "elevenlabs"

    with pytest.raises(StrictModeError):
        ProviderFactory.create_tts_provider(config)


def test_factory_strict_raises_for_unknown_image_provider(config):
    config["settings"]["strict"] = True
    config["providers"]["images"]["provider"] = "no-such-provider"

    with pytest.raises(StrictModeError):
        ProviderFactory.create_image_provider(config)


def test_factory_non_strict_still_falls_back(config, monkeypatch):
    """Default behaviour is unchanged: unavailable providers fall back."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config["providers"]["tts"]["provider"] = "elevenlabs"

    provider = ProviderFactory.create_tts_provider(config)
    assert isinstance(provider, GTTSProvider)


def test_factory_registers_voicebox():
    assert "voicebox" in ProviderFactory.list_tts_providers()


# --- Voicebox ephemeral voice clone (create -> generate -> delete) ------------


def test_factory_unusable_image_fallback_drops_to_text(config, monkeypatch):
    """A configured fallback that is itself unusable must not be returned;
    the factory drops to the always-available text provider instead."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    config["api_keys"] = {}
    config["providers"]["images"]["provider"] = "dalle3"
    config["providers"]["images"]["fallback"] = "pexels"

    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, TextImageProvider)


# --- gTTS fallback counter (read by cli.py to warn the user) ------------------


def test_tts_session_state_is_per_instance(config):
    """Session state must live on the instance (set in __init__), not on the
    class, so two runs in one process can never share uploads/profiles."""
    from slide_stream.providers.tts import ChatterboxTTSProvider, VoiceboxTTSProvider

    assert "_session_voice" in vars(ChatterboxTTSProvider(config))
    voicebox = VoiceboxTTSProvider(config)
    assert "_session_profile_id" in vars(voicebox)
    assert "_profile_ready" in vars(voicebox)
    assert "_engine" in vars(KokoroTTSProvider(config))


# --- Kokoro output format ------------------------------------------------------
