"""AI-vision image renaming for the local image provider.

Ported/adapted from slide-vision. Describes each image in a folder with a
vision-capable LLM and renames it to a keyword slug (e.g. ``a-photo.jpg`` ->
``golden-retriever-in-park.jpg``), so the ``local`` image provider can match
images to slides by filename keywords. Reuses slide-stream's LLM vision layer,
so it works with claude/openai/gemini.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .llm import get_llm_client, query_llm_with_image
from .providers.images import IMAGE_EXTENSIONS

console = Console()

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}

DESCRIBE_PROMPT = (
    "Describe this image in 4-6 words suitable for a filename. Use only "
    "lowercase words separated by hyphens. No punctuation, no file extension."
)


@dataclass
class RenameRecord:
    original: Path
    new_name: str


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:60].strip("-") or "image"


def build_rename_records(
    folder: Path, provider: str, model: str | None = None
) -> list[RenameRecord]:
    """Describe each image and propose a keyword-slug filename."""
    client = get_llm_client(provider)
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    records: list[RenameRecord] = []
    for img in images:
        media_type = _MIME.get(img.suffix.lower(), "image/png")
        description = query_llm_with_image(
            client, provider, DESCRIBE_PROMPT, img.read_bytes(),
            media_type, console, model,
        )
        slug = slugify(description or img.stem)
        records.append(RenameRecord(original=img, new_name=f"{slug}{img.suffix.lower()}"))
    return records


def apply_renames(
    records: list[RenameRecord], dry_run: bool = False
) -> list[tuple[Path, Path]]:
    """Rename files, disambiguating collisions with a numeric suffix."""
    used: set[Path] = set()
    applied: list[tuple[Path, Path]] = []
    for record in records:
        name_path = Path(record.new_name)
        stem, ext = name_path.stem, name_path.suffix
        parent = record.original.parent

        new_path = parent / record.new_name
        count = 1
        while _is_taken(new_path, record.original, used):
            new_path = parent / f"{stem}-{count}{ext}"
            count += 1

        used.add(new_path)
        if not dry_run and record.original != new_path:
            record.original.rename(new_path)
        applied.append((record.original, new_path))
    return applied


def _is_taken(candidate: Path, original: Path, used: set[Path]) -> bool:
    if candidate in used:
        return True
    if candidate == original:
        return False  # renaming a file to its own name is a harmless no-op
    return candidate.exists()


def write_scan_report(folder: Path, applied: list[tuple[Path, Path]]) -> Path:
    lines = ["# Scan Report", "", "| Original | Renamed To |", "|---|---|"]
    for orig, new in applied:
        lines.append(f"| `{orig.name}` | `{new.name}` |")
    report = folder / "scan-report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
