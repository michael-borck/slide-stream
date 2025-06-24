# Type Safety Documentation

This document tracks all type-related compromises, workarounds, and code smells in the slide-stream project. Maintaining this documentation ensures transparency about type safety decisions and helps with future refactoring.

## Summary

- **Total Type Ignores**: 7
- **Type Guards**: 1
- **Any Usage**: 3 instances
- **Missing Type Stubs**: 3 third-party libraries
- **Legacy Typing**: 0 (all modernized to Python 3.10+ syntax)
- **Cast Usage**: 0
- **hasattr/getattr Patterns**: 0

## Type Ignore Comments

### 1. MoviePy Missing Type Stubs

**Files**: `src/slide_stream/cli.py`, `src/slide_stream/media.py`

```python
# src/slide_stream/cli.py:207
ImageClip(f).set_duration(ImageClip(f).duration)  # type: ignore[attr-defined]

# src/slide_stream/media.py:123  
ImageClip(image_path, duration=duration).set_position("center")  # type: ignore[attr-defined]
```

**Reason**: MoviePy lacks proper type stubs. Methods like `set_duration()`, `set_position()`, `resize()` exist at runtime but are not typed.

**Risk Level**: Low - MoviePy is mature, methods are well-documented

**Future Action**: 
- Monitor for official MoviePy type stubs
- Consider contributing type stubs to typeshed
- Alternative: Create local `.pyi` stub files

### 2. Google Generative AI Missing Type Stubs

**File**: `src/slide_stream/llm.py`

```python
# Lines 15, 20, 23, 98, 100
import google.generativeai as genai  # type: ignore[import-untyped]
genai.configure(api_key=api_key)  # type: ignore[attr-defined]
genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
```

**Reason**: Google's `google-generativeai` package lacks type stubs. The library is relatively new and rapidly evolving.

**Risk Level**: Medium - API may change, but package is official Google library

**Future Action**:
- Check for official type stubs with each package update
- Consider creating custom type stubs for core functionality
- Monitor Google's type annotation roadmap

## Type Guards

### 1. BeautifulSoup Element Type Safety

**File**: `src/slide_stream/parser.py:24`

```python
if isinstance(next_sibling, Tag):
    if next_sibling.name in ["ul", "ol"]:
        # Safe to access Tag-specific attributes
```

**Reason**: BeautifulSoup's `find_next_sibling()` returns `Tag | NavigableString | None`. Only `Tag` objects have `.name` and `.find_all()` methods.

**Benefits**: 
- Runtime safety - prevents AttributeError
- Type safety - basedpyright understands the narrowed type
- Correct behavior - skips text nodes as intended

**Performance Impact**: Minimal - single isinstance check per sibling

## Any Type Usage

### 1. LLM Client Return Type

**File**: `src/slide_stream/llm.py:11`

```python
def get_llm_client(provider: str) -> Any:
```

**Reason**: Different LLM providers return different client types:
- OpenAI: `openai.OpenAI`
- Gemini: `google.generativeai.GenerativeModel` 
- Claude: `anthropic.Anthropic`
- Groq: `groq.Groq`

**Risk Level**: Medium - Loses type safety for client methods

**Future Action**: Consider using Protocol or Union types

### 2. LLM Query Function Parameter

**File**: `src/slide_stream/llm.py:84`

```python
def query_llm(client: Any, provider: str, ...):
```

**Reason**: Must accept different client types from `get_llm_client()`

**Risk Level**: Medium - Coupled to above Any usage

**Future Action**: Same as above - Protocol-based design

### 3. Parser Dictionary Values

**File**: `src/slide_stream/parser.py:9`

```python
def parse_markdown(markdown_text: str) -> list[dict[str, Any]]:
```

**Reason**: Slide content can be strings (titles) or lists (content items)

**Risk Level**: Low - Well-defined structure, could use TypedDict

**Future Action**: Consider TypedDict for slide structure:
```python
class Slide(TypedDict):
    title: str
    content: list[str]
```

## Missing Type Stubs Analysis

### Third-Party Libraries Without Complete Type Support

| Library | Type Coverage | Impact | Workaround |
|---------|---------------|---------|------------|
| `moviepy` | None | High usage | `type: ignore[attr-defined]` |
| `google-generativeai` | None | Optional feature | `type: ignore[import-untyped]` |
| `beautifulsoup4` | Partial | Core functionality | Type guards + proper imports |
| `gtts` | Partial | Limited usage | No issues currently |
| `pillow` | Good | No issues | Well-typed |
| `requests` | Excellent | No issues | Well-typed |

## Code Quality Assessment

### ✅ Good Practices

1. **Modern Type Syntax**: All code uses Python 3.10+ union syntax (`str | None` vs `Optional[str]`)
2. **Explicit Type Guards**: Runtime checks that help both safety and type checking
3. **Documented Ignores**: All `type: ignore` comments include specific error codes
4. **Strategic Ignores**: Type ignores are surgical, not broad suppressions

### ⚠️ Areas for Improvement

1. **Optional Dependency Handling**: Could use protocols for better abstraction
2. **Any Usage**: `llm.py` uses `Any` for client return types - could be more specific
3. **Error Handling**: Some type ignores could be replaced with try/catch blocks

### ✅ Good Patterns (Not Present but Worth Noting)

**No Legacy Typing Syntax**
- No `typing.Union` (uses `|` syntax)
- No `typing.Optional` (uses `| None` syntax)  
- No `typing.List/Dict` (uses `list/dict` builtins)

**No Unsafe Patterns**
- No `cast()` usage - all type narrowing uses proper guards
- No `hasattr()`/`getattr()` defensive programming - dependencies are well-defined
- No broad `# type: ignore` without error codes
- No `# noqa` or other linter suppressions

**No Complex Union Types** 
- Simple, clean type signatures
- Type guards handle complexity rather than complex union annotations

## Recommendations for Future Development

### Short Term (Next Release)

1. **Monitor Type Stub Releases**
   - Check for MoviePy type stubs quarterly
   - Watch Google AI library for typing improvements

2. **Add Integration Tests**
   - Test actual third-party library integration
   - Catch API changes that type ignores might hide

### Medium Term (6 months)

1. **Create Custom Type Stubs**
   ```python
   # stubs/moviepy/editor.pyi
   class ImageClip:
       def set_duration(self, duration: float) -> ImageClip: ...
       def set_position(self, position: str) -> ImageClip: ...
   ```

2. **Protocol-Based Design**
   ```python
   from typing import Protocol
   
   class LLMClient(Protocol):
       def generate(self, prompt: str) -> str: ...
   ```

### Long Term (1 year)

1. **Contribute to Ecosystem**
   - Submit type stubs to DefinitelyTyped/typeshed
   - Contribute to upstream projects

2. **Static Analysis Expansion**
   - Add mypy configuration alongside basedpyright
   - Implement type coverage reporting

## Type Ignore Reference

### Error Codes Used

- `[attr-defined]`: Attribute exists at runtime but not in type definitions
- `[import-untyped]`: Importing module without type information
- `[arg-type]`: Argument type mismatch (not currently used)
- `[return-value]`: Return type mismatch (not currently used)

### Basedpyright Configuration

Current settings in `pyproject.toml`:
```toml
[tool.basedpyright]
pythonVersion = "3.10"
typeCheckingMode = "basic"
reportMissingImports = true
reportMissingTypeStubs = false  # Would be noisy with current dependencies
```

## Monitoring Strategy

### Monthly Checks
1. Review dependency updates for new type stub releases
2. Check if any `type: ignore` comments can be removed

### Quarterly Reviews
1. Assess third-party library type coverage improvements
2. Evaluate creating custom type stubs for heavily used libraries
3. Review this document for accuracy and completeness

### Release Checklist
- [ ] All type ignores documented
- [ ] No new undocumented type compromises
- [ ] basedpyright passes with zero errors/warnings
- [ ] Type safety documentation updated

## Contact and Updates

**Last Updated**: 2024-06-24  
**Next Review**: 2024-09-24  
**Owner**: Project maintainers

For questions about type safety decisions or to propose improvements, please open an issue with the "typing" label.