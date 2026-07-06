"""LLM integration for Slide Stream."""

import base64
import os
from typing import Any

from rich.console import Console

err_console = Console(stderr=True, style="bold red")

# Default Claude model: Haiku is fast, cheap, vision-capable, and more than
# enough for narration writing. Overridable via --llm-model / CLAUDE_MODEL.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"

# Providers whose clients can accept an image alongside the prompt.
VISION_PROVIDERS = ("claude", "openai", "openai-compatible", "ollama", "gemini")


def get_llm_client(provider: str, base_url: str | None = None) -> Any:
    """Get LLM client based on provider.

    For ``openai-compatible``, ``base_url`` selects the backend (a local
    server such as LocalAI/vLLM/llama.cpp, or any hosted OpenAI-compatible
    API). Falls back to the ``OPENAI_BASE_URL`` env var when not given.
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

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
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
            api_key = os.getenv("OPENAI_API_KEY", "not-needed")
            return OpenAI(base_url=resolved_base_url, api_key=api_key)
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
) -> str | None:
    """Query LLM with given prompt."""
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
            return response.text

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

            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.choices[0].message.content

        elif provider == "claude":
            # Use provided model or fallback to environment variable or default
            selected_model = model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            response = client.messages.create(
                model=selected_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.content[0].text

        elif provider == "groq":
            # Use provided model or fallback to environment variable or default
            selected_model = model or os.getenv(
                "GROQ_MODEL", "llama-3.1-8b-instant"
            )  # Updated default
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.choices[0].message.content

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
            return response.content[0].text

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
            return response.choices[0].message.content

        elif provider == "gemini":
            response = client.generate_content(
                [{"mime_type": media_type, "data": image_bytes}, prompt_text]
            )
            return response.text

        return None

    except Exception as e:
        err_console.print(f"  - LLM vision error: {e}")
        return None
