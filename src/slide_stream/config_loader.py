"""Configuration loading and management for Slide Stream."""

import copy
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
            # Voicebox (self-hosted, multi-engine voice cloning) is the default.
            # It needs a base_url; with none configured, the factory falls back
            # to free gTTS, so a bare install still works out of the box.
            "provider": "voicebox",
            "voice": None,
            "engine": "kokoro",  # voicebox engine (kokoro|chatterbox|qwen|...)
            "base_url": None,    # voicebox / chatterbox / openai-compatible server
        },
        "avatar": {
            "provider": "none",       # none|static|puppet|precomputed|d-id|sadtalker|wav2lip|wan-s2v|comfyui
            "base_url": None,         # comfyui server (sadtalker/wav2lip/comfyui)
            "assets_dir": None,       # precomputed head_N.mp4 clips
            "source_image": None,     # photo: d-id / sadtalker
            "source_video": None,     # short clip: wav2lip
            "source": None,           # comfyui auto-router: photo OR video
            "api_key": None           # bearer/DID key / ${VAR}
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


def expand_env_vars(value: Any, _path: str = "") -> Any:
    """Recursively expand environment variables in configuration values.

    A reference to an unset variable expands to "" (backward compatible),
    but a warning is printed so a typo'd ``${VAR}`` doesn't silently
    disable auth or drop a setting.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        expanded = os.getenv(env_var)
        if expanded is None:
            at_key = f" (config key: {_path})" if _path else ""
            err_console.print(
                f"⚠️  Environment variable '{env_var}' is not set{at_key}; "
                "treating the value as empty."
            )
            return ""
        return expanded
    elif isinstance(value, dict):
        return {
            k: expand_env_vars(v, f"{_path}.{k}" if _path else str(k))
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [
            expand_env_vars(item, f"{_path}[{i}]")
            for i, item in enumerate(value)
        ]
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
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading config file: {e}")
    if data is not None and not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file must be a YAML mapping (key: value pairs), "
            f"got {type(data).__name__}: {config_file}"
        )
    return data


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration by layering, later layers winning:

    1. Built-in defaults
    2. User-level config (~/.slidestream.yaml) — personal server URLs / keys
    3. Project-level config — the explicit ``config_path`` if given, otherwise
       ./slidestream.yaml

    So a personal home config can hold your TTS server and API keys once, and
    each project's config only needs its deck-specific overrides.
    """
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


def _default_block(path: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the DEFAULT_CONFIG mapping at ``path``, or None if absent."""
    node: Any = DEFAULT_CONFIG
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
    _path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Two special cases:

    - A provider block (a mapping with a ``provider`` key, e.g. the blocks
      under ``providers:``) whose ``provider`` value *changes* in the
      override does not inherit the lower layer's other keys — a home
      config's ``voice: rachel`` for elevenlabs must not leak into a
      project's voicebox block. The block restarts from built-in defaults
      plus the override's own keys.
    - A ``None`` override for a key that is a mapping in the base (a user
      commented out all children of a section) keeps the base mapping
      instead of clobbering it with None.
    """
    result = base.copy()

    for key, value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and value is None:
            continue  # null section: keep the lower layer / defaults
        if isinstance(base_value, dict) and isinstance(value, dict):
            key_path = _path + (key,)
            if (
                "provider" in base_value
                and "provider" in value
                and value["provider"] != base_value["provider"]
            ):
                # Provider changed: drop the old provider's sibling keys.
                fresh = copy.deepcopy(_default_block(key_path)) or {}
                # Pin the new provider up front so the recursive merge is a
                # plain deep merge (no re-triggering of this branch).
                fresh["provider"] = value["provider"]
                result[key] = merge_configs(fresh, value, key_path)
            else:
                result[key] = merge_configs(base_value, value, key_path)
        else:
            result[key] = value

    return result


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    """Coerce a config section to a mapping: None becomes {}, anything else
    non-dict is a clean ConfigurationError instead of a raw traceback."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Config section '{name}' must be a mapping (key: value pairs), "
            f"got {type(value).__name__}"
        )
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration structure and required values."""
    required_sections = ["providers", "settings"]

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Validate providers section
    providers = _as_mapping(config["providers"], "providers")
    required_providers = ["llm", "images", "tts"]

    for provider_type in required_providers:
        if provider_type not in providers:
            raise ConfigurationError(f"Missing provider configuration: {provider_type}")

    # Validate video settings
    settings = _as_mapping(config["settings"], "settings")
    video_settings = _as_mapping(settings.get("video"), "settings.video")
    resolution = video_settings.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ConfigurationError("Video resolution must be a list of two integers")


def create_starter_config() -> str:
    """A minimal, valid starter config — what ``slide-stream init`` writes.

    As-is it renders a video with no API keys (text image cards, free gTTS
    narration, no avatar). Every provider variation lives in the full reference
    (``slide-stream init --full`` / ``slidestream.example.yaml``) and the docs,
    so this file stays short and hard to get wrong."""
    return """# SlideStream configuration
#
# This minimal file works as-is with no API keys: text-slide images, free gTTS
# narration, no avatar. Switch a provider on below when you want more.
#
# Config is layered, later winning:
#   1. built-in defaults
#   2. ~/.slidestream.yaml   (personal: server URLs + API keys, set once)
#   3. ./slidestream.yaml    (this deck; or pass --config FILE)
#   4. CLI flags             (e.g. --voice, --tts-provider)
#
# Every provider + option, with examples:  slide-stream init --full
#   (writes the complete slidestream.example.yaml reference)
# Full guide:  docs/USER_GUIDE.md
# Sanity-check a deck + this config any time:  slide-stream doctor <deck>

providers:
  # Rewrites slide content into spoken narration. `none` speaks the slide text
  # as-is (no key needed). For AI narration set a provider + its API key env var:
  #   openai | claude | gemini | groq | ollama | openai-compatible
  llm:
    provider: none

  # Slide pictures. `text` draws clean text cards (no key, always works).
  # Others: dalle3, gemini, swarmui, pexels, unsplash, local, openai-compatible.
  images:
    provider: text

  # Voice. `gtts` is free (needs internet). Offline: kokoro. Self-hosted/cloned:
  # voicebox, chatterbox. Premium: elevenlabs, openai, openai-compatible.
  tts:
    provider: gtts

  # Talking-head overlay, off by default. Set a provider (static, puppet,
  # wan-s2v, sadtalker, wav2lip, d-id, ...) then enable per run with --avatar.
  # List the built-in mascots:  slide-stream avatars
  avatar:
    provider: none

settings:
  # Fail the run if a configured provider is unusable, instead of silently
  # degrading (e.g. a missing voice -> gTTS). Also per run: --strict
  strict: false
"""


def save_starter_config(path: str = "slidestream.yaml") -> None:
    """Write the minimal starter config (owner-only; it may hold secrets later)."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(create_starter_config())
    os.chmod(path, 0o600)
    console.print(f"📁 Created configuration: {path}")


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
    provider: dalle3        # text, local, dalle3, gemini, swarmui, openai-compatible, pexels, unsplash
    fallback: text         # used only if the primary is unavailable at startup
                           # (missing key/base_url); a per-image failure while
                           # rendering always falls straight to a text card.
    # folder: ./images     # for the 'local' provider — matched by filename keywords
    # model: imagen-4.0-fast-generate-001   # for the 'gemini' (Imagen) provider
    # --- self-hosted SwarmUI ---
    # base_url: https://image.example.org
    # model: juggernautXL_v9   # SwarmUI model name
    # steps: 20

  tts:
    provider: elevenlabs   # gtts, kokoro, chatterbox, voicebox, elevenlabs, openai, openai-compatible
    voice: rachel          # voice ID/name (provider-specific)
    # gtts free accents: accent: australian|british|american|canadian|indian|irish
    # self-hosted Voicebox studio (free, multi-engine):
    #   provider: voicebox
    #   base_url: https://voice.example.org
    #   profile_id: "<id from POST /profiles>"   # reuse a persistent profile
    #   engine: kokoro      # or chatterbox / qwen / luxtts / tada
    # ...or clone YOUR voice for one run, then delete it from the server:
    #   provider: voicebox
    #   base_url: https://voice.example.org
    #   voice_sample: ./my_voice.wav        # 10-30s of clean speech
    #   engine: chatterbox
    #   # reference_text omitted -> Voicebox transcribes the clip itself.
    #   # Set it to skip transcription, or transcribe_model to pin Whisper size.
    #   delete_generations: true   # default; also drop rendered audio server-side

  # Talking-head avatar overlay (off by default). Enable per run with --avatar,
  # disable with --no-avatar. List the built-in mascots with: slide-stream avatars
  #
  # No-GPU options (work on any mascot or your own image):
  #   provider: static             # a still image held in the corner
  #   source: owl                  # a built-in name (teddy/panda/koala/robot/
  #                                # wizard/owl), or an image path
  # ...or a crude cartoon mouth-flap driven by the audio loudness:
  #   provider: puppet
  #   source: owl
  #   # mouth: [0.5, 0.6, 0.12, 0.06]   # custom image: mouth region (fractions)
  #
  # 'precomputed' composites ready-made clips named head_1.mp4, head_2.mp4, ...
  # from assets_dir — no GPU or service needed. With strict mode on, every
  # slide must get a head clip.
  #   provider: precomputed
  #   assets_dir: ./heads
  #
  # Animated talking head from a STILL image + the narration audio, via a
  # ComfyUI server. Wan2.2-S2V has no face detector, so it animates the
  # built-in mascots (teddy/owl/robot/wizard) AND human head shots:
  #   provider: wan-s2v
  #   base_url: https://comfyui.example.org
  #   api_key: "${COMFYUI_TOKEN}"   # if the server checks a Bearer token
  #   source: owl                   # a built-in avatar name, or ./me.png
  #   # clip_seconds: 4             # short clip looped under the narration
  #   # full_length: true           # or render the whole narration (slow)
  #
  # Talking-head avatar (self-hosted via a ComfyUI server):
  #   provider: sadtalker           # from a PHOTO (human faces only)
  #   base_url: https://comfyui.example.org
  #   source_image: ./me.png
  # ...or from a short VIDEO (more natural; loops under longer narration):
  #   provider: wav2lip
  #   base_url: https://comfyui.example.org
  #   source_video: ./me_idle.mp4
  # ...or auto-route (photo -> SadTalker, video -> Wav2Lip):
  #   provider: comfyui
  #   base_url: https://comfyui.example.org
  #   source: ./me.png              # or ./me_idle.mp4
  # ...or the hosted BYOK option (bills per minute of video):
  #   provider: d-id
  #   source_image: ./me.png
  #   api_key: "${DID_API_KEY}"

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
    # Users fill this in with API keys — keep it private to the owner.
    os.chmod(path, 0o600)
    console.print(f"📁 Created example configuration: {path}")
