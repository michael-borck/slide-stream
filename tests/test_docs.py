"""Guards that keep user-facing docs in step with the code."""

import re
from pathlib import Path

from slide_stream.providers.factory import ProviderFactory

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DOC = (ROOT / "docs" / "PROVIDERS.md").read_text(encoding="utf-8")
LLM_PROVIDERS = ("ollama", "openai-compatible", "claude", "openai", "gemini", "groq")


def _table_names(heading: str) -> set[str]:
    """Provider names in the first column of a section's table."""
    section = PROVIDERS_DOC.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([a-z0-9-]+)` \|", section, re.MULTILINE))


def test_every_tts_provider_is_documented():
    assert set(ProviderFactory.list_tts_providers()) <= _table_names("Voice")


def test_every_image_provider_is_documented():
    assert set(ProviderFactory.list_image_providers()) <= _table_names("Pictures")


def test_every_avatar_provider_is_documented():
    names = set(ProviderFactory.list_avatar_providers()) - {"none"}
    assert names <= _table_names("Presenter")


def test_every_llm_provider_is_documented():
    assert set(LLM_PROVIDERS) <= _table_names("Narration writer")


def test_readme_links_are_absolute_for_pypi():
    """The README is the PyPI page, where relative links break."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", readme)
    relative = [link for link in links if not link.startswith(("http://", "https://", "#", "mailto:"))]
    assert relative == []


def test_no_doc_says_ffmpeg_must_be_installed():
    """ffmpeg ships with moviepy; installing it is optional."""
    paths = [ROOT / "README.md", *sorted(p for p in (ROOT / "docs").glob("*.md") if not p.name.startswith("._"))]
    offenders = [
        p.name for p in paths
        if re.search(r"FFmpeg\*?\*?:? (?:is )?(?:required|for video processing)", p.read_text(encoding="utf-8"), re.IGNORECASE)
    ]
    assert offenders == []
