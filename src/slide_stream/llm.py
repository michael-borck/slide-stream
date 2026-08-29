"""LLM integration for Slide Stream."""

import base64
import os
import re
from typing import Any

from rich.console import Console

err_console = Console(stderr=True, style="bold red")

# Default Claude model: Haiku is fast, cheap, vision-capable, and more than
# enough for narration writing. Overridable via --llm-model / CLAUDE_MODEL.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"

# Providers whose clients can accept an image alongside the prompt.
VISION_PROVIDERS = ("claude", "openai", "openai-compatible", "ollama", "gemini")

# Reasoning models (Ollama qwen3/deepseek-r1, some Gemma templates) prepend a
# "<think>…</think>" block to their answer. Narration and decks must never
# voice it, so completions are stripped before they reach callers.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str | None) -> str | None:
    """Drop reasoning-model 'thinking' blocks from a completion."""
    if not text:
        return text
    return _THINK_RE.sub("", text).lstrip() or text


def get_llm_client(
    provider: str, base_url: str | None = None, api_key: str | None = None
) -> Any:
    """Get LLM client based on provider.

    For ``openai-compatible`` and ``ollama``, ``base_url`` selects the backend
    (a local server such as Ollama/LocalAI/vLLM/llama.cpp, or any hosted
    OpenAI-compatible API) and ``api_key`` becomes the Bearer token — for an
    Ollama sitting behind an authenticating reverse proxy. Falls back to
    config env vars (``OLLAMA_BASE_URL``/``OLLAMA_TOKEN``/``OPENAI_BASE_URL``/
    ``OPENAI_API_KEY``) when not given.
    """
    if provider == "gemini":
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            # Allow model configuration via environment variable
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "Gemini library not found. Please install with: pip install slide-stream[gemini]"
            )

    elif provider == "openai":
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            return OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library not found. Please install with: pip install slide-stream[openai]"
            )

    elif provider == "claude":
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic library not found. Please install with: pip install slide-stream[claude]"
            )

    elif provider == "groq":
        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            return Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Groq library not found. Please install with: pip install slide-stream[groq]"
            )

    elif provider == "ollama":
        try:
            from openai import OpenAI

            resolved = (
                base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
            ).rstrip("/")
            if not resolved.endswith("/v1"):
                resolved += "/v1"
            # A bearer token is only needed when Ollama sits behind an
            # authenticating proxy; stock Ollama ignores it.
            key = api_key or os.getenv("OLLAMA_TOKEN") or "ollama"
            return OpenAI(base_url=resolved, api_key=key)
        except ImportError:
            raise ImportError(
                "OpenAI library not found. Please install with: pip install slide-stream[openai]"
            )

    elif provider == "openai-compatible":
        try:
            from openai import OpenAI

            resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
            if not resolved_base_url:
                raise ValueError(
                    "openai-compatible LLM requires a base_url (config "
                    "providers.llm.base_url or the OPENAI_BASE_URL env var)."
                )
            # Local servers usually ignore the key; send a placeholder so the
            # client constructs cleanly.
            key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")
            return OpenAI(base_url=resolved_base_url, api_key=key)
        except ImportError:
            raise ImportError(
                "OpenAI library not found. Please install with: pip install slide-stream[openai]"
            )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def query_llm(
    client: Any,
    provider: str,
    prompt_text: str,
    rich_console: Console,
    model: str | None = None,
    think: bool | None = None,
) -> str | None:
    """Query LLM with given prompt.

    ``think=False`` asks Ollama/OpenAI-compatible reasoning models not to
    think (``extra_body={"think": False}``); thinking blocks are stripped from
    every provider's answer regardless.
    """
    rich_console.print("  - Querying LLM...")

    try:
        if provider == "gemini":
            # For Gemini, model is set during client creation, but allow override
            if model:
                # Create a new client with the specified model
                import google.generativeai as genai  # type: ignore[import-untyped]

                temp_client = genai.GenerativeModel(model)  # type: ignore[attr-defined]
                response = temp_client.generate_content(prompt_text)
            else:
                response = client.generate_content(prompt_text)
            return _strip_think(response.text)

        elif provider in ["openai", "ollama", "openai-compatible"]:
            # Use provided model or fallback to environment variable or default
            if model:
                selected_model = model
            elif provider == "openai":
                selected_model = os.getenv(
                    "OPENAI_MODEL", "gpt-4o-mini"
                )  # Updated default
            elif provider == "ollama":
                selected_model = os.getenv(
                    "OLLAMA_MODEL", "llama3.2"
                )  # Updated default
            else:  # openai-compatible
                selected_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

            extra: dict[str, Any] = {}
            if think is False and provider in ("ollama", "openai-compatible"):
                # Ollama reasoning models accept this on the OpenAI-compatible
                # endpoint; servers that ignore it are covered by _strip_think.
                extra["extra_body"] = {"think": False}

            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_text}],
                **extra,
            )
            return _strip_think(response.choices[0].message.content)

        elif provider == "claude":
            # Use provided model or fallback to environment variable or default
            selected_model = model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            response = client.messages.create(
                model=selected_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return _strip_think(response.content[0].text)

        elif provider == "groq":
            # Use provided model or fallback to environment variable or default
            selected_model = model or os.getenv(
                "GROQ_MODEL", "llama-3.1-8b-instant"
            )  # Updated default
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return _strip_think(response.choices[0].message.content)

        return None

    except Exception as e:
        err_console.print(f"  - LLM Error: {e}")
        return None


def query_llm_with_image(
    client: Any,
    provider: str,
    prompt_text: str,
    image_bytes: bytes,
    media_type: str,
    rich_console: Console,
    model: str | None = None,
) -> str | None:
    """Query a vision-capable LLM with an image and a prompt.

    Used for image-only slides: the model describes the image and turns it
    into narration. Returns None when the provider has no vision path or the
    call fails (callers fall back to text-only narration).
    """
    if provider not in VISION_PROVIDERS:
        err_console.print(
            f"  - LLM provider '{provider}' does not support image input; "
            "narrating from the slide title only."
        )
        return None

    rich_console.print("  - Querying LLM (with slide image)...")
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        if provider == "claude":
            selected_model = model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            response = client.messages.create(
                model=selected_model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
            )
            return _strip_think(response.content[0].text)

        elif provider in ["openai", "ollama", "openai-compatible"]:
            if model:
                selected_model = model
            elif provider == "openai":
                selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif provider == "ollama":
                selected_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            else:
                selected_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
            )
            return _strip_think(response.choices[0].message.content)

        elif provider == "gemini":
            response = client.generate_content(
                [{"mime_type": media_type, "data": image_bytes}, prompt_text]
            )
            return _strip_think(response.text)

        return None

    except Exception as e:
        err_console.print(f"  - LLM vision error: {e}")
        return None
