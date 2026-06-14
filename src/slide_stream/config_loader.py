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
            "base_url": None  # used by the openai-compatible provider
        },
        "tts": {
            "provider": "gtts",
            "voice": None,
            "base_url": None  # used by the openai-compatible provider
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
        "temp_dir": "temp_files",
        "cleanup": True
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


def find_config_file() -> Path | None:
    """Find configuration file in standard locations."""
    possible_locations = [
        Path("./slidestream.yaml"),
        Path("./slidestream.yml"),
        Path.home() / ".slidestream.yaml",
        Path.home() / ".slidestream.yml"
    ]

    for location in possible_locations:
        if location.exists():
            return location

    return None


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from file or return defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
    else:
        config_file = find_config_file()

    if config_file:
        try:
            with open(config_file, encoding='utf-8') as f:
                file_config = yaml.safe_load(f)

            if file_config:
                # Deep merge with defaults
                config = merge_configs(config, file_config)
                console.print(f"✅ Loaded configuration from: {config_file}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading config file: {e}")
    else:
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

providers:
  llm:
    provider: openai        # none, openai, gemini, claude, groq, ollama, openai-compatible
    model: gpt-4o-mini     # optional: specific model to use

  images:
    provider: dalle3        # text, dalle3, openai-compatible, pexels, unsplash
    fallback: text         # fallback when primary fails

  tts:
    provider: elevenlabs   # gtts, elevenlabs, openai, openai-compatible
    voice: rachel          # voice ID/name (provider-specific)

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

  temp_dir: temp_files
  cleanup: true
"""


def save_example_config(path: str = "slidestream.yaml") -> None:
    """Save an example configuration file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(create_example_config())
    console.print(f"📁 Created example configuration: {path}")
