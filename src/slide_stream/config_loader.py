"""Configuration loading and management for Slide Stream."""

import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


DEFAULT_CONFIG = {
    "providers": {
        "llm": {
            "provider": "none",
            "model": None,
            "base_url": None  # used by the openai-compatible provider
        },
        "images": {
            "provider": "text",
            "fallback": "text",
            "base_url": None,  # used by the openai-compatible provider
            "folder": None     # used by the 'local' provider (enrich/create)
        },
        "tts": {
            "provider": "gtts",
            "voice": None,
            "base_url": None  # used by the openai-compatible provider
        },
        "avatar": {
            "provider": "none",       # none | precomputed (| musetalk, later)
            "base_url": None,         # HTTP lip-sync service (later phase)
            "base_video": None,       # lecturer base footage (later phase)
            "assets_dir": None        # precomputed head_N.mp4 clips
        }
    },
    "settings": {
        "video": {
            "resolution": [1920, 1080],
            "fps": 24,
            "codec": "libx264",
            "audio_codec": "aac",
            "slide_duration_padding": 1.0,
            "default_slide_duration": 5.0
        },
        "image": {
            "download_timeout": 15,
            "bg_color": "black",
            "title_font_size": 100,
            "content_font_size": 60,
            "font_color": "white",
            "max_line_width": 50
        },
        "narration": {
            # Approximate spoken length per slide in seconds (None = natural
            # length). Also settable per run with --narration-seconds.
            "target_seconds": None,
            # Speaking rate used to convert seconds into a word target.
            "wpm": 150
        },
        "avatar": {
            "position": "bottom-right",  # bottom-left | top-left | top-right
            "size": 0.28,                # circle diameter as fraction of frame height
            "margin": 24,                # px from the frame edges
            "shape": "circle"            # circle only in v1
        },
        "temp_dir": "temp_files",
        "cleanup": True,
        # Strict mode: fail the run when a configured provider is unusable or
        # errors, instead of silently degrading (e.g. ElevenLabs -> gTTS).
        "strict": False
    }
}


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def _first_existing(*candidates: Path) -> Path | None:
    """Return the first path that exists, or None."""
    for path in candidates:
        if path.exists():
            return path
    return None


def find_home_config() -> Path | None:
    """Find the user-level config (~/.slidestream.yaml) — personal defaults
    like a TTS server URL and API keys, shared across all projects."""
    return _first_existing(
        Path.home() / ".slidestream.yaml",
        Path.home() / ".slidestream.yml",
    )


def find_project_config() -> Path | None:
    """Find the project-level config (./slidestream.yaml) — settings for the
    deck at hand."""
    return _first_existing(
        Path("./slidestream.yaml"),
        Path("./slidestream.yml"),
    )


def find_config_file() -> Path | None:
    """Find a single config file (project preferred, then home).

    Retained for backwards compatibility; ``load_config`` layers home and
    project configs rather than using this.
    """
    return find_project_config() or find_home_config()


def _read_config_file(config_file: Path) -> dict[str, Any] | None:
    """Read and parse one YAML config file (None if empty)."""
    try:
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading config file: {e}")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration by layering, later layers winning:

    1. Built-in defaults
    2. User-level config (~/.slidestream.yaml) — personal server URLs / keys
    3. Project-level config — the explicit ``config_path`` if given, otherwise
       ./slidestream.yaml

    So a personal home config can hold your TTS server and API keys once, and
    each project's config only needs its deck-specific overrides.
    """
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)

    # Layer sources: (label, path). Home is always layered underneath; an
    # explicit --config replaces the auto-discovered project file but still
    # sits on top of home.
    sources: list[Path] = []
    if home_config := find_home_config():
        sources.append(home_config)
    if config_path:
        project_config = Path(config_path)
        if not project_config.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        sources.append(project_config)
    elif project_config := find_project_config():
        sources.append(project_config)

    for source in sources:
        file_config = _read_config_file(source)
        if file_config:
            config = merge_configs(config, file_config)
            console.print(f"✅ Loaded configuration from: {source}")

    if not sources:
        console.print("📋 Using default configuration")

    # Expand environment variables
    config = expand_env_vars(config)

    # Validate configuration
    validate_config(config)

    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration structure and required values."""
    required_sections = ["providers", "settings"]

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Validate providers section
    providers = config["providers"]
    required_providers = ["llm", "images", "tts"]

    for provider_type in required_providers:
        if provider_type not in providers:
            raise ConfigurationError(f"Missing provider configuration: {provider_type}")

    # Validate video settings
    video_settings = config["settings"]["video"]
    if not isinstance(video_settings["resolution"], list) or len(video_settings["resolution"]) != 2:
        raise ConfigurationError("Video resolution must be a list of two integers")


def create_example_config() -> str:
    """Create example configuration file content."""
    return """# SlideStream Configuration File
#
# Config is layered, later winning:
#   1. built-in defaults
#   2. ~/.slidestream.yaml   (personal: TTS server URL, API keys — set once)
#   3. ./slidestream.yaml    (this deck's settings; or pass --config FILE)
# So keep your server and keys in the home file and only put per-deck
# overrides here. Any CLI flag (e.g. --voice, --tts-base-url) wins over both.

providers:
  llm:
    provider: openai        # none, openai, gemini, claude, groq, ollama, openai-compatible
    model: gpt-4o-mini     # optional: specific model (or --llm-model on the CLI)
    # API keys come from the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY,
    # GEMINI_API_KEY, GROQ_API_KEY. Narration uses speaker notes as the
    # primary source when present (cleaned up and fitted to the target
    # length); otherwise it writes presenter-style narration from the slide
    # content instead of reading the bullets aloud. Image-only slides are
    # narrated via vision-capable providers (claude, openai, gemini).

  images:
    provider: dalle3        # text, local, dalle3, openai-compatible, pexels, unsplash
    fallback: text         # fallback when primary fails
    # folder: ./images     # for the 'local' provider — matched by filename keywords

  tts:
    provider: elevenlabs   # gtts, kokoro, elevenlabs, openai, openai-compatible
    voice: rachel          # voice ID/name (provider-specific)

  # Talking-head avatar overlay (off by default). 'precomputed' composites
  # ready-made clips named head_1.mp4, head_2.mp4, ... from assets_dir —
  # no GPU or service needed. Enable per run with --avatar, disable with
  # --no-avatar. With strict mode on, every slide must get a head clip.
  # avatar:
  #   provider: precomputed
  #   assets_dir: ./heads

# --- Fully offline TTS (no cloud, no API key) ------------------------------
# Requires: pip install "slide-stream[local-tts]"
# Model files (~340MB) download once to ~/.cache/slide-stream/kokoro/.
# Voices include af_sarah, af_bella (female) and am_adam, am_michael (male).
#
# providers:
#   tts:
#     provider: kokoro
#     voice: af_sarah
# ---------------------------------------------------------------------------

# --- Privacy-first cloned voice (self-hosted Chatterbox) ------------------
# Your reference recording is uploaded under a random UUID name for THIS RUN
# ONLY, so no lecturer-recognisable voice ever exists on the server; a
# server-side cron (contrib/chatterbox/cleanup_uuid_voices.sh) reaps the
# UUID files. Use 10-30s of clean speech (under ~5s fails). List selectable
# voices with: slide-stream voices
#
# providers:
#   tts:
#     provider: chatterbox
#     base_url: https://chatterbox.example.org
#     voice_sample: ./my_voice.wav      # ephemeral clone of YOUR voice
#     # voice: Emily.wav                # ...or a stock server voice instead
#     api_key: "${CHATTERBOX_TOKEN}"    # if your proxy checks a Bearer token
# settings:
#   strict: true                        # never fall back to the wrong voice
# ---------------------------------------------------------------------------

# --- Fully local / self-hosted stack -------------------------------------
# Point every layer at an OpenAI-compatible server (LocalAI, vLLM,
# openedai-speech, Kokoro-FastAPI, ...). No cloud keys required.
#
# providers:
#   llm:
#     provider: openai-compatible
#     base_url: http://localhost:8080/v1
#     model: llama-3.1-8b-instruct
#   images:
#     provider: openai-compatible
#     base_url: http://localhost:8080/v1
#     model: sdxl
#     fallback: text
#   tts:
#     provider: openai-compatible
#     base_url: http://localhost:8000/v1
#     model: tts-1
#     voice: en_US-amy
# -------------------------------------------------------------------------

# API Keys (use environment variables)
api_keys:
  openai: "${OPENAI_API_KEY}"
  elevenlabs: "${ELEVENLABS_API_KEY}"
  pexels: "${PEXELS_API_KEY}"
  unsplash: "${UNSPLASH_ACCESS_KEY}"

settings:
  video:
    resolution: [1920, 1080]
    fps: 24
    codec: libx264
    audio_codec: aac
    slide_duration_padding: 1.0
    default_slide_duration: 5.0

  image:
    download_timeout: 15
    bg_color: black
    title_font_size: 100
    content_font_size: 60
    font_color: white
    max_line_width: 50

  # Narration length target (approximate seconds of speech per slide).
  # Long speaker notes are summarised to fit; thin slides are expanded.
  # Override per run: slide-stream create ... --narration-seconds 30
  #
  # For an exact script instead of LLM-written narration:
  #   --verbatim-notes          speak the PowerPoint speaker notes as written
  #   --script my_script.txt    one block per slide, separated by --- lines
  narration:
    target_seconds: 60
    wpm: 150

  # Talking-head overlay appearance (used when providers.avatar is enabled)
  avatar:
    position: bottom-right   # bottom-left, top-left, top-right
    size: 0.28               # circle diameter as fraction of frame height
    margin: 24               # px from the frame edges

  temp_dir: temp_files
  cleanup: true

  # Fail the run if a configured provider is unusable or errors, instead of
  # silently falling back (e.g. elevenlabs -> gtts, dalle3 -> text image).
  # Recommended when the exact voice matters (e.g. a cloned lecturer voice).
  # Can also be enabled per-run with: slide-stream create --strict ...
  strict: false
"""


def save_example_config(path: str = "slidestream.yaml") -> None:
    """Save an example configuration file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(create_example_config())
    console.print(f"📁 Created example configuration: {path}")
