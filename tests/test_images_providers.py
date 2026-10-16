"""Tests for picture providers: text cards, DALL-E, Gemini, SwarmUI, OpenAI-compatible."""

import pytest
from PIL import Image

from slide_stream.config_loader import DEFAULT_CONFIG
from slide_stream.providers.base import StrictModeError
from slide_stream.providers.factory import ProviderFactory
from slide_stream.providers.images import (
    DalleImageProvider,
    OpenAICompatImageProvider,
    TextImageProvider,
)


@pytest.fixture
def config():
    """A minimal valid config (a fresh copy of the defaults)."""
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


# --- TextImageProvider (no network) ----------------------------------------


def test_text_image_provider_writes_file(config, tmp_path):
    out = tmp_path / "slide.png"
    provider = TextImageProvider(config)

    result = provider.generate_image("Quantum computing", str(out))

    assert result == str(out)
    assert out.exists()
    # Image is created at the configured resolution.
    with Image.open(out) as img:
        assert img.size == tuple(config["settings"]["video"]["resolution"])


def test_text_image_provider_always_available(config):
    assert TextImageProvider(config).is_available() is True
    assert TextImageProvider(config).name == "text"


# --- Provider factory selection / fallback ---------------------------------


def test_dalle_availability_follows_api_key(config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["api_keys"] = {}
    assert DalleImageProvider(config).is_available() is False

    config["api_keys"] = {"openai": "sk-test"}
    assert DalleImageProvider(config).is_available() is True


def test_dalle_falls_back_to_text_on_api_error(config, tmp_path, mocker):
    """If the DALL-E call fails, a text image is produced instead."""
    out = tmp_path / "img.png"
    config["api_keys"] = {"openai": "sk-test"}

    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = DalleImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.exists()  # produced by the text fallback


def test_dalle_fallback_renders_slide_content(config, tmp_path, mocker):
    """When DALL-E fails, the text fallback renders the slide's own content
    (not a generic 'Topic: ...'): two slides sharing a query but differing in
    content produce different fallback images."""
    config["api_keys"] = {"openai": "sk-test"}
    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    provider = DalleImageProvider(config)
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"
    provider.generate_image(
        "same query", str(out_a), slide={"title": "T", "content": ["Apple", "Banana"]}
    )
    provider.generate_image(
        "same query", str(out_b), slide={"title": "T", "content": ["Carrot", "Daikon"]}
    )

    assert out_a.exists() and out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()


def test_factory_selects_openai_compat_image(config):
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "fallback": "text",
        "base_url": "http://localhost:8080/v1",
    }
    provider = ProviderFactory.create_image_provider(config)
    assert isinstance(provider, OpenAICompatImageProvider)


def test_openai_compat_image_handles_b64(config, tmp_path, mocker):
    """Local servers commonly return inline base64 rather than a URL."""
    import base64

    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    item = mocker.MagicMock()
    item.b64_json = base64.b64encode(b"raw-image-bytes").decode()
    item.url = None
    fake_client = mocker.MagicMock()
    fake_client.images.generate.return_value.data = [item]
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"raw-image-bytes"


def test_openai_compat_image_handles_url(config, tmp_path, mocker):
    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    item = mocker.MagicMock()
    item.b64_json = None
    item.url = "http://localhost:8080/image.png"
    fake_client = mocker.MagicMock()
    fake_client.images.generate.return_value.data = [item]
    mocker.patch("openai.OpenAI", return_value=fake_client)

    fake_get = mocker.MagicMock()
    fake_get.content = b"downloaded-bytes"
    mocker.patch(
        "slide_stream.providers.images.requests.get", return_value=fake_get
    )

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"downloaded-bytes"


def test_openai_compat_image_falls_back_to_text_on_error(config, tmp_path, mocker):
    out = tmp_path / "img.png"
    config["providers"]["images"] = {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8080/v1",
    }

    fake_client = mocker.MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("boom")
    mocker.patch("openai.OpenAI", return_value=fake_client)

    result = OpenAICompatImageProvider(config).generate_image("topic", str(out))

    assert result == str(out)
    assert out.exists()  # text fallback produced an image


def test_openai_compat_image_requires_base_url_not_just_key(config, monkeypatch):
    """A bare OpenAI key must NOT make the compat image provider available:
    without a base_url it would silently bill real OpenAI instead of the local
    server. Only an explicit base_url enables it."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config["api_keys"] = {"openai": "sk-test"}
    config["providers"]["images"] = {"provider": "openai-compatible"}
    assert OpenAICompatImageProvider(config).is_available() is False

    config["providers"]["images"]["base_url"] = "http://localhost:8080/v1"
    assert OpenAICompatImageProvider(config).is_available() is True


def test_text_image_provider_renders_slide_content(config, tmp_path):
    """The text provider renders the slide's own content, not a generic
    placeholder. Two slides sharing a query but differing in content produce
    different images (the old code ignored content entirely)."""
    slide_a = {"title": "Same Title", "content": ["Apple", "Banana"]}
    slide_b = {"title": "Same Title", "content": ["Carrot", "Daikon"]}
    out_a = tmp_path / "a.png"
    out_b = tmp_path / "b.png"

    TextImageProvider(config).generate_image("same query", str(out_a), slide=slide_a)
    TextImageProvider(config).generate_image("same query", str(out_b), slide=slide_b)

    assert out_a.exists() and out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()


# --- Strict mode (no silent fallbacks) --------------------------------------


def test_image_strict_raises_instead_of_text_fallback(config, tmp_path, monkeypatch):
    """A failing image provider raises in strict mode rather than silently
    rendering a text slide."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["settings"]["strict"] = True
    out = tmp_path / "slide.png"

    with pytest.raises(StrictModeError):
        DalleImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_image_non_strict_falls_back_to_text(config, tmp_path, monkeypatch):
    """Default behaviour is unchanged: a failing image provider writes a text
    slide image instead."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "slide.png"

    result = DalleImageProvider(config).generate_image("query", str(out))
    assert result == str(out)
    assert out.exists()


# --- Font loading (text slides must not use the tiny bitmap fallback) -------


def test_load_font_returns_scalable_font_at_requested_size():
    """load_font must yield a real scalable font at the asked-for size; the
    old code silently fell back to Pillow's unsized ~10px bitmap font on
    systems without arial.ttf/DejaVuSans.ttf on Pillow's search path (macOS)."""
    from slide_stream.providers.images import load_font

    font = load_font(72)
    assert getattr(font, "size", None) == 72


# --- Kokoro local TTS --------------------------------------------------------


def test_gemini_image_availability_follows_key(config, monkeypatch):
    from slide_stream.providers.images import GeminiImageProvider

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config["api_keys"] = {}
    assert GeminiImageProvider(config).is_available() is False

    config["api_keys"] = {"gemini": "g-test"}
    assert GeminiImageProvider(config).is_available() is True


def test_gemini_image_falls_back_to_text_without_sdk(config, tmp_path, monkeypatch):
    """No google-genai installed -> text fallback (still produces an image)."""
    from slide_stream.providers.images import GeminiImageProvider

    config["api_keys"] = {"gemini": "g-test"}
    out = tmp_path / "img.png"
    # google-genai is not installed in the test env, so the import fails.
    result = GeminiImageProvider(config).generate_image(
        "Quantum", str(out), slide={"title": "Quantum", "content": ["a"]}
    )
    assert result == str(out)
    assert out.exists()  # text card


def test_gemini_image_success_with_mocked_sdk(config, tmp_path, monkeypatch, mocker):
    """Inject a fake google-genai and verify the Imagen bytes are written."""
    import sys

    from slide_stream.providers.images import GeminiImageProvider

    config["api_keys"] = {"gemini": "g-test"}
    config["providers"]["images"]["model"] = "imagen-4.0-fast-generate-001"

    fake_image = mocker.MagicMock()
    fake_image.image.image_bytes = b"\x89PNG-imagen"
    fake_response = mocker.MagicMock(generated_images=[fake_image])
    fake_client = mocker.MagicMock()
    fake_client.models.generate_images.return_value = fake_response

    genai_mod = mocker.MagicMock()
    genai_mod.Client.return_value = fake_client
    types_mod = mocker.MagicMock()
    google_mod = mocker.MagicMock()
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)

    out = tmp_path / "img.png"
    result = GeminiImageProvider(config).generate_image("Neurons", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"\x89PNG-imagen"
    # Configured model + landscape aspect ratio were passed through.
    kwargs = fake_client.models.generate_images.call_args.kwargs
    assert kwargs["model"] == "imagen-4.0-fast-generate-001"


def test_factory_registers_gemini_image_provider():
    assert "gemini" in ProviderFactory.list_image_providers()


# --- SwarmUI image provider (network mocked) ---------------------------------


def test_swarmui_availability_needs_base_url(config, monkeypatch):
    from slide_stream.providers.images import SwarmUIImageProvider

    monkeypatch.delenv("SWARMUI_BASE_URL", raising=False)
    config["providers"]["images"] = {"provider": "swarmui"}
    assert SwarmUIImageProvider(config).is_available() is False

    config["providers"]["images"]["base_url"] = "https://image.example.org"
    assert SwarmUIImageProvider(config).is_available() is True


def test_swarmui_generate_flow(config, tmp_path, mocker):
    """session -> generate -> fetch path -> save bytes; model passed through."""
    from slide_stream.providers.images import SwarmUIImageProvider

    config["providers"]["images"] = {
        "provider": "swarmui",
        "base_url": "https://image.example.org",
        "model": "juggernautXL_v9",
        "steps": 20,
    }

    session_resp = mocker.MagicMock()
    session_resp.json.return_value = {"session_id": "sess-123"}
    gen_resp = mocker.MagicMock()
    gen_resp.json.return_value = {"images": ["View/local/0.png"]}
    img_resp = mocker.MagicMock(content=b"\x89PNG-swarm")

    post = mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp],
    )
    get = mocker.patch(
        "slide_stream.providers.images.requests.get", return_value=img_resp
    )

    out = tmp_path / "img.png"
    result = SwarmUIImageProvider(config).generate_image("a red bicycle", str(out))

    assert result == str(out)
    assert out.read_bytes() == b"\x89PNG-swarm"
    # GenerateText2Image got the session id, model, and steps.
    gen_call = post.call_args_list[1]
    assert gen_call[0][0].endswith("/API/GenerateText2Image")
    body = gen_call[1]["json"]
    assert body["session_id"] == "sess-123"
    assert body["model"] == "juggernautXL_v9"
    assert body["steps"] == 20
    # Image fetched from the returned server path.
    assert get.call_args[0][0] == "https://image.example.org/View/local/0.png"


def test_swarmui_no_model_falls_back_to_text(config, tmp_path, mocker):
    from slide_stream.providers.images import SwarmUIImageProvider

    config["providers"]["images"] = {
        "provider": "swarmui",
        "base_url": "https://image.example.org",
    }
    session_resp = mocker.MagicMock()
    session_resp.json.return_value = {"session_id": "s"}
    gen_resp = mocker.MagicMock()
    gen_resp.json.return_value = {"error": "No model input given."}
    mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp],
    )

    out = tmp_path / "img.png"
    result = SwarmUIImageProvider(config).generate_image(
        "topic", str(out), slide={"title": "T", "content": ["a"]}
    )
    assert result == str(out)
    assert out.exists()  # text fallback rendered


def test_factory_registers_swarmui_provider():
    assert "swarmui" in ProviderFactory.list_image_providers()


# --- Voicebox TTS provider (network mocked) ----------------------------------


def test_swarmui_strict_raises_instead_of_text_fallback(config, tmp_path, mocker):
    from slide_stream.providers.images import SwarmUIImageProvider

    config["settings"]["strict"] = True
    config["providers"]["images"] = {
        "provider": "swarmui", "base_url": "https://image.example.org",
    }
    mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=RuntimeError("server down"),
    )

    out = tmp_path / "img.png"
    with pytest.raises(StrictModeError):
        SwarmUIImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_gemini_strict_raises_instead_of_text_fallback(config, tmp_path):
    from slide_stream.providers.images import GeminiImageProvider

    config["settings"]["strict"] = True
    config["api_keys"] = {"gemini": "g-test"}
    out = tmp_path / "img.png"
    # google-genai is not installed in the test env: the ImportError path must
    # also refuse the text fallback under strict.
    with pytest.raises(StrictModeError):
        GeminiImageProvider(config).generate_image("query", str(out))
    assert not out.exists()


def test_swarmui_headers_tolerate_bearer_prefix(monkeypatch):
    """A token pasted with its scheme ('Bearer xyz') must not produce
    'Bearer Bearer xyz' — strip the redundant prefix."""
    from slide_stream.providers.images import SwarmUIImageProvider

    monkeypatch.delenv("SWARMUI_TOKEN", raising=False)
    prefixed = SwarmUIImageProvider({"providers": {"images": {"api_key": "Bearer abc123"}}})
    assert prefixed._headers() == {"Authorization": "Bearer abc123"}
    plain = SwarmUIImageProvider({"providers": {"images": {"api_key": "abc123"}}})
    assert plain._headers() == {"Authorization": "Bearer abc123"}
    empty = SwarmUIImageProvider({"providers": {"images": {}}})
    assert empty._headers() == {}


def test_swarmui_payload_blocks_text_and_supports_style(config, tmp_path, mocker):
    """Prompts forbid baked-in text; negativeprompt reinforces it; the
    optional style switch swaps the directive."""
    from slide_stream.providers.images import SwarmUIImageProvider

    config["providers"]["images"] = {
        "provider": "swarmui",
        "base_url": "https://image.example.org",
        "model": "juggernautXL_v9",
    }
    session_resp = mocker.MagicMock()
    session_resp.json.return_value = {"session_id": "s"}
    gen_resp = mocker.MagicMock()
    gen_resp.json.return_value = {"images": ["View/local/0.png"]}
    img_resp = mocker.MagicMock(content=b"\x89PNG")
    mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp],
    )
    mocker.patch("slide_stream.providers.images.requests.get", return_value=img_resp)

    out = tmp_path / "img.png"
    post = mocker.patch(
        "slide_stream.providers.images.requests.post",
        side_effect=[session_resp, gen_resp, session_resp, gen_resp],
    )
    mocker.patch("slide_stream.providers.images.requests.get", return_value=img_resp)

    provider = SwarmUIImageProvider(config)
    provider.generate_image("a red bicycle", str(out))
    body = post.call_args_list[1][1]["json"]
    assert "No text, words, letters" in body["prompt"]
    assert "text" in body["negativeprompt"]
    assert "cartoon" not in body["prompt"].lower()

    # Cartoon style swaps the directive.
    config["providers"]["images"]["style"] = "cartoon"
    provider = SwarmUIImageProvider(config)
    provider.generate_image("a red bicycle", str(out))
    body = post.call_args_list[-1][1]["json"]
    assert "cartoon" in body["prompt"].lower()


# --- Voice floor (ADR-0012): configured -> Kokoro (if installed) -> gTTS -----
