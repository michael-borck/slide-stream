"""LLM integration for Slide Stream."""

import os
from typing import Any

from rich.console import Console

err_console = Console(stderr=True, style="bold red")


def get_llm_client(provider: str) -> Any:
    """Get LLM client based on provider."""
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

        elif provider in ["openai", "ollama"]:
            # Use provided model or fallback to environment variable or default
            if model:
                selected_model = model
            elif provider == "openai":
                selected_model = os.getenv(
                    "OPENAI_MODEL", "gpt-4o-mini"
                )  # Updated default
            else:  # ollama
                selected_model = os.getenv(
                    "OLLAMA_MODEL", "llama3.2"
                )  # Updated default

            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.choices[0].message.content

        elif provider == "claude":
            # Use provided model or fallback to environment variable or default
            selected_model = model or os.getenv(
                "CLAUDE_MODEL", "claude-3-5-sonnet-20241022"
            )  # Updated default
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
