# Type Safety Improvement Examples

This document provides concrete examples of how to improve type safety in slide-stream, showing before/after code patterns.

## 1. LLM Client Protocol Design

### Current Implementation (Any usage)

```python
# src/slide_stream/llm.py
def get_llm_client(provider: str) -> Any:
    # Returns different client types
    
def query_llm(client: Any, provider: str, prompt: str, console: Console, model: str | None = None) -> str | None:
    # Works but loses type safety
```

### Improved Implementation (Protocol-based)

```python
from typing import Protocol

class LLMClient(Protocol):
    def generate_response(self, prompt: str, model: str | None = None) -> str:
        """Generate a response from the LLM."""
        ...

class OpenAIClientAdapter:
    def __init__(self, client):
        self._client = client
    
    def generate_response(self, prompt: str, model: str | None = None) -> str:
        response = self._client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

def get_llm_client(provider: str) -> LLMClient:
    # Returns properly typed client
    
def query_llm(client: LLMClient, prompt: str) -> str:
    return client.generate_response(prompt)
```

**Benefits:**
- Type safety for all client interactions
- Uniform interface across providers  
- Easy to test with mock implementations
- Clear contract for what clients must provide

## 2. Slide Data Structure

### Current Implementation (dict with Any)

```python
# src/slide_stream/parser.py
def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
    slides = []
    slides.append({"title": slide_title, "content": content_items})
    return slides
```

### Improved Implementation (TypedDict)

```python
from typing import TypedDict

class Slide(TypedDict):
    title: str
    content: list[str]

def parse_markdown(markdown_text: str) -> list[Slide]:
    slides: list[Slide] = []
    slides.append({"title": slide_title, "content": content_items})
    return slides
```

**Benefits:**
- IntelliSense support for slide dictionary keys
- Compile-time checking of required fields
- Self-documenting data structure
- Easy to extend with additional fields

## 3. MoviePy Type Stubs

### Current Implementation (type: ignore)

```python
# src/slide_stream/media.py
image_clip = ImageClip(image_path, duration=duration).set_position("center")  # type: ignore[attr-defined]
```

### Improved Implementation (Custom Type Stubs)

Create `stubs/moviepy/__init__.pyi`:

```python
from typing import Any

class ImageClip:
    def __init__(self, image: str, duration: float | None = None) -> None: ...
    def set_position(self, position: str | tuple[float, float]) -> ImageClip: ...
    def set_duration(self, duration: float) -> ImageClip: ...
    def resize(self, width: int | None = None, height: int | None = None) -> ImageClip: ...
    def set_audio(self, audio: Any | None) -> ImageClip: ...
    def write_videofile(
        self, 
        filename: str, 
        fps: int = 24, 
        codec: str = "libx264",
        audio_codec: str = "aac",
        logger: Any = None
    ) -> None: ...
    
    @property
    def duration(self) -> float: ...
    @property
    def h(self) -> int: ...
    @property
    def w(self) -> int: ...

class AudioFileClip:
    def __init__(self, filename: str) -> None: ...
    def close(self) -> None: ...
    
    @property
    def duration(self) -> float: ...

def concatenate_videoclips(clips: list[ImageClip]) -> ImageClip: ...
```

Update `pyproject.toml`:
```toml
[tool.basedpyright]
stubPath = "stubs"
```

**Benefits:**
- Remove all MoviePy-related type ignores
- Full type safety for video operations
- Better IDE support and error detection
- Can contribute back to the community

## 4. Error Handling Improvements

### Current Implementation (broad exception handling)

```python
try:
    # LLM operations
    response = client.generate_content(prompt_text)
    return response.text
except Exception as e:
    err_console.print(f"LLM Error: {e}")
    return None
```

### Improved Implementation (specific exception handling)

```python
from openai import OpenAIError
from anthropic import AnthropicError

def query_llm(client: LLMClient, prompt: str) -> str | None:
    try:
        return client.generate_response(prompt)
    except OpenAIError as e:
        err_console.print(f"OpenAI API Error: {e}")
        return None
    except AnthropicError as e:
        err_console.print(f"Claude API Error: {e}")  
        return None
    except Exception as e:
        err_console.print(f"Unexpected LLM Error: {e}")
        return None
```

**Benefits:**
- More specific error messages
- Different handling for different error types
- Better debugging information
- Type-safe error handling

## 5. Configuration Improvements

### Current Implementation (loose configuration)

```python
# Environment variables accessed directly
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

### Improved Implementation (typed configuration)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai", "claude", "gemini", "groq", "ollama"]
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    
    @classmethod
    def from_env(cls, provider: str) -> "LLMConfig":
        return cls(
            provider=provider,  # type: ignore[arg-type]
            api_key=os.getenv(f"{provider.upper()}_API_KEY"),
            model=os.getenv(f"{provider.upper()}_MODEL"),
            base_url=os.getenv(f"{provider.upper()}_BASE_URL")
        )

def get_llm_client(config: LLMConfig) -> LLMClient:
    # Type-safe configuration access
```

**Benefits:**
- Immutable configuration
- Type checking for provider names
- Centralized configuration logic
- Easy to test with different configurations

## 6. Optional Dependency Management

### Current Implementation (try/import with Any)

```python
def get_llm_client(provider: str) -> Any:
    if provider == "gemini":
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
            # ...
        except ImportError:
            raise ImportError("Gemini library not found...")
```

### Improved Implementation (typed optional imports)

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import google.generativeai as genai
    import openai
    import anthropic

class OptionalImportError(ImportError):
    """Raised when an optional dependency is not available."""
    pass

def get_gemini_client() -> "genai.GenerativeModel":
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise OptionalImportError(
            "Google Generative AI not installed. "
            "Install with: pip install slide-stream[gemini]"
        ) from e
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)
```

**Benefits:**
- Better error messages with installation instructions
- Type hints work in development
- Clear separation of concerns
- Custom exception types for better error handling

## Implementation Priority

1. **High Priority**: Slide TypedDict (low risk, high benefit)
2. **Medium Priority**: LLM Protocol design (moderate complexity)
3. **Medium Priority**: Custom type stubs (time-consuming but valuable)
4. **Low Priority**: Configuration dataclasses (nice-to-have)

## Testing Type Improvements

```python
# tests/test_types.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    reveal_type(parse_markdown("# Test"))  # Should show list[Slide]
    reveal_type(get_llm_client("openai"))  # Should show LLMClient

def test_slide_structure():
    slides = parse_markdown("# Test\n- item")
    slide = slides[0]
    
    # These should have full type support:
    assert slide["title"] == "Test"
    assert slide["content"] == ["item"]
    
    # This should be a type error:
    # slide["invalid_key"]  # mypy/basedpyright should catch this
```

Each improvement should be implemented incrementally with proper testing to ensure no regression in functionality.