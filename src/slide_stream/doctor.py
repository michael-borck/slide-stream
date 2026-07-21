"""Deck preflight ("doctor"): assess a deck + resolved config before rendering.

Static and offline — it inspects the parsed slides, the layered config, the
render flags, and any local assets (voice sample, embedded images), then reports
warnings and estimates so a user knows what they'll get before spending render
time or money. Powers ``create --dry-run`` and the ``doctor`` command.
"""

import importlib.util
import os
import re
import shutil
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from .avatars import BUILTIN_AVATARS
from .narration import narration_source, strip_stage_directions

# Severity ranks for the summary/exit code.
OK, WARN, BLOCKER = "ok", "warn", "blocker"
_ICON = {OK: "✅", WARN: "⚠️ ", BLOCKER: "❌"}

# Group order in the report.
GROUPS = ["Deck", "Narration", "Voice", "Images", "Avatar", "Providers & environment"]

# Speaker-note fragments that are stage directions, not spoken prose.
_STAGE_DIRECTION = re.compile(
    r"\[[^\]]+\]"                                  # [pause], [click]
    r"|\((?:[^)]*\b(?:pause|click|next|slow|beat|breath|aside|note)\b[^)]*)\)"
    r"|\b(?:pause here|click(?: here| to advance)?|advance|next slide|"
    r"read (?:this )?slowly|take (?:this|it) slow(?:ly|er)?|slow down|"
    r"speaker note|note to self|emphasi[sz]e|gesture|point to|TODO|FIXME)\b"
    r"|https?://",
    re.IGNORECASE,
)

# Per-image cost (USD, rough) for paid image providers, else 0.
_IMAGE_COST = {"dalle3": 0.04, "gemini": 0.02}
# Rough per-slide avatar render minutes (very approximate; for expectations).
_AVATAR_MINUTES = {"wan-s2v": 15.0, "sadtalker": 2.0, "wav2lip": 2.0,
                   "comfyui": 2.0, "d-id": 1.0}
# Which env var / config key each provider needs (for a presence check).
_LLM_KEY = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}
_IMAGE_KEY = {"dalle3": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
              "pexels": "PEXELS_API_KEY", "unsplash": "UNSPLASH_ACCESS_KEY"}
_TTS_KEY = {"elevenlabs": "ELEVENLABS_API_KEY", "openai": "OPENAI_API_KEY"}

# Providers whose SDK ships in an optional extra rather than the core install.
# Maps provider -> (importable module, pip extra) per role, so the preflight can
# tell a user who configured e.g. `claude` but never ran the extra install
# exactly what to `pip install` — instead of a clean report then a render-time
# crash. Providers backed by core `requests` (voicebox, swarmui, pexels, ...)
# are absent here: nothing extra to install.
_LLM_PKG = {"openai": ("openai", "openai"), "claude": ("anthropic", "claude"),
            "gemini": ("google.generativeai", "gemini"), "groq": ("groq", "groq"),
            "ollama": ("openai", "openai"), "openai-compatible": ("openai", "openai")}
_IMAGE_PKG = {"dalle3": ("openai", "openai"), "gemini": ("google.genai", "gemini"),
              "openai-compatible": ("openai", "openai")}
_TTS_PKG = {"elevenlabs": ("elevenlabs", "elevenlabs"), "openai": ("openai", "openai"),
            "openai-compatible": ("openai", "openai"), "kokoro": ("kokoro_onnx", "local-tts")}


def _missing_extra(provider: str, pkg_map: dict[str, tuple[str, str]]) -> str | None:
    """Return the pip extra a configured provider needs but is missing, else None.

    A provider with no entry in ``pkg_map`` needs no extra (core install)."""
    entry = pkg_map.get(provider)
    if entry is None:
        return None
    module, extra = entry
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False  # parent package missing / namespace resolution failed
    return None if found else extra


@dataclass
class Finding:
    group: str
    severity: str
    message: str


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)
    estimates: list[str] = field(default_factory=list)

    def add(self, group: str, severity: str, message: str) -> None:
        self.findings.append(Finding(group, severity, message))

    @property
    def blockers(self) -> int:
        return sum(1 for f in self.findings if f.severity == BLOCKER)

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == WARN)

    def exit_code(self, fail_on_warn: bool = False) -> int:
        if self.blockers:
            return 1
        if fail_on_warn and self.warnings:
            return 1
        return 0


def _wc(text: str) -> int:
    return len(text.split())


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _audio_seconds(path: Path) -> float | None:
    """Duration of an audio file, or None if it can't be read locally."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                rate = w.getframerate()
                return w.getnframes() / rate if rate else None
        except Exception:
            pass
    try:
        from moviepy import AudioFileClip

        with AudioFileClip(str(path)) as a:
            return float(a.duration)
    except Exception:
        return None


def _image_size(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def run_doctor(
    slides: list[dict[str, Any]],
    config: dict[str, Any],
    options: dict[str, Any],
) -> DoctorReport:
    """Assess a parsed deck against the resolved config + render options.

    ``options`` keys: mode ('create'|'enrich'), input_ext, verbatim_notes,
    script_blocks, avatar_enabled, narration_seconds, output_path.
    """
    report = DoctorReport()
    providers = config.get("providers", {})
    settings = config.get("settings", {})
    video = settings.get("video", {})
    narration = settings.get("narration", {})
    wpm = float(narration.get("wpm", 150) or 150)
    target_seconds = options.get("narration_seconds") or narration.get("target_seconds")

    _check_deck(report, slides, options)
    per_slide = _check_narration(report, slides, options, target_seconds, wpm)
    _check_voice(report, providers)
    _check_images(report, slides, providers, video)
    _check_avatar(report, providers, options)
    _check_providers_env(report, providers, settings, options)
    _estimates(report, slides, providers, per_slide, video)
    return report


def _check_deck(report: DoctorReport, slides: list[dict[str, Any]], options: dict[str, Any]) -> None:
    n = len(slides)
    report.add("Deck", OK, f"{n} slide(s)")

    has_notes = [i for i, s in enumerate(slides, 1) if str(s.get("notes", "")).strip()]
    if options.get("input_ext") == ".pptx":
        missing = [i for i in range(1, n + 1) if i not in has_notes]
        if missing:
            report.add("Deck", WARN,
                       f"{len(missing)} slide(s) missing speaker notes: {_ids(missing)}")
        else:
            report.add("Deck", OK, "every slide has speaker notes")
    elif not has_notes:
        report.add("Deck", OK, "Markdown deck (no speaker notes) — narration from content")

    titleless = [i for i, s in enumerate(slides, 1)
                 if not s.get("has_real_title", True) or not str(s.get("title", "")).strip()]
    if titleless:
        report.add("Deck", WARN, f"{len(titleless)} slide(s) without a title: {_ids(titleless)}")

    image_only = [i for i, s in enumerate(slides, 1)
                  if s.get("images") and not s.get("content")
                  and not str(s.get("title", "")).strip()]
    if image_only:
        report.add("Deck", WARN,
                   f"{len(image_only)} image-only slide(s) with no title: {_ids(image_only)} "
                   "— narration leans on the image (needs a vision LLM)")

    empty = [i for i, s in enumerate(slides, 1)
             if not str(s.get("title", "")).strip() and not s.get("content")
             and not str(s.get("notes", "")).strip() and not s.get("images")]
    if empty:
        report.add("Deck", BLOCKER, f"{len(empty)} empty slide(s): {_ids(empty)} — nothing to render")


def _check_narration(report, slides, options, target_seconds, wpm) -> list[float]:
    """Per-slide narration checks; returns per-slide estimated durations."""
    verbatim = options.get("verbatim_notes")
    script_blocks = options.get("script_blocks")
    per_slide: list[float] = []

    if script_blocks is not None:
        report.add("Narration", OK, "source: external --script (verbatim)")
        if len(script_blocks) != len(slides):
            report.add("Narration", WARN,
                       f"script has {len(script_blocks)} block(s) but deck has "
                       f"{len(slides)} slide(s) — matched in order")
    elif verbatim:
        report.add("Narration", OK, "source: speaker notes, spoken verbatim")
    else:
        report.add("Narration", OK, "source: notes/content (LLM-written where available)")

    target_words = int(target_seconds / 60 * wpm) if target_seconds else None
    for i, slide in enumerate(slides, 1):
        notes = str(slide.get("notes", "")).strip()

        if script_blocks is not None:
            block = script_blocks[i - 1] if i - 1 < len(script_blocks) else ""
            words = _wc(block)
        elif verbatim and notes:
            words = _wc(notes)
        elif target_words:
            words = target_words
        else:
            src = narration_source(slide)
            text = notes if src == "notes" else (
                " ".join(str(c) for c in slide.get("content", [])) if src == "content"
                else str(slide.get("title", "")))
            words = _wc(text)
        per_slide.append(max(words, 1) / wpm * 60)

        # Stage directions in notes that would STILL be spoken after the
        # automatic strip (bracketed forms are removed for us; only prose-form
        # cues like "pause here" survive and need a human to fix).
        if notes and (verbatim or narration_source(slide) == "notes"):
            m = _STAGE_DIRECTION.search(strip_stage_directions(notes))
            if m:
                report.add("Narration", WARN,
                           f"slide {i} note has a stage direction "
                           f'"{m.group(0).strip()[:40]}" — it would be spoken')
        # Notes far from the target length.
        if target_words and notes and narration_source(slide) == "notes" and not verbatim:
            w = _wc(notes)
            if w < target_words * 0.5:
                report.add("Narration", WARN,
                           f"slide {i} note is {w} words — short for a "
                           f"~{int(target_seconds)}s target")
            elif w > target_words * 2:
                report.add("Narration", WARN,
                           f"slide {i} note is {w} words — will be summarised to fit")
    return per_slide


def _check_voice(report: DoctorReport, providers: dict[str, Any]) -> None:
    tts = providers.get("tts", {})
    sample = tts.get("voice_sample")
    if not sample:
        return
    path = Path(str(sample))
    if not path.is_file():
        report.add("Voice", BLOCKER, f"voice_sample not found: {sample}")
        return
    secs = _audio_seconds(path)
    if secs is None:
        report.add("Voice", OK, f"voice_sample {path.name} (duration unreadable)")
        return
    if secs < 5:
        report.add("Voice", BLOCKER,
                   f"voice_sample {path.name} is {secs:.1f}s — clones fail under ~5s")
    elif secs < 10:
        report.add("Voice", WARN,
                   f"voice_sample {path.name} is {secs:.1f}s — short (10–30s recommended)")
    elif secs > 60:
        report.add("Voice", WARN,
                   f"voice_sample {path.name} is {secs:.0f}s — longer than needed (10–30s ideal)")
    else:
        report.add("Voice", OK, f"voice_sample {path.name} is {secs:.1f}s")


def _check_images(report, slides, providers, video) -> None:
    images_cfg = providers.get("images", {})
    provider = images_cfg.get("provider", "text")
    report.add("Images", OK, f"provider: {provider}")

    if provider == "local":
        folder = images_cfg.get("folder")
        if not folder or not Path(str(folder)).is_dir():
            report.add("Images", WARN, f"local image folder not found: {folder}")
        else:
            files = [p for p in Path(str(folder)).iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}]
            if len(files) < len(slides):
                report.add("Images", WARN,
                           f"folder has {len(files)} image(s) for {len(slides)} slide(s) — "
                           "some slides may fall back to a text card")

    # Resolution check for images already embedded in the deck (pptx).
    res = video.get("resolution", [1920, 1080])
    vw = res[0] if isinstance(res, list) and res else 1920
    small = []
    for i, slide in enumerate(slides, 1):
        for img in slide.get("images", []) or []:
            data = img.get("data") if isinstance(img, dict) else None
            if not data:
                continue
            size = _image_size(data)
            if size and size[0] < vw * 0.6:
                small.append(f"{i} ({size[0]}×{size[1]})")
    if small:
        report.add("Images", WARN,
                   f"embedded image(s) smaller than the {vw}px frame: {', '.join(small[:5])} "
                   "— they appear small (no upscaling)")


def _check_avatar(report, providers, options) -> None:
    if not options.get("avatar_enabled"):
        return
    avatar = providers.get("avatar", {})
    provider = avatar.get("provider", "none")
    source = avatar.get("source") or avatar.get("source_image") or avatar.get("source_video")
    report.add("Avatar", OK, f"provider: {provider}")

    is_mascot = bool(source and str(source).lower() in BUILTIN_AVATARS)
    if is_mascot and provider in ("sadtalker", "wav2lip", "comfyui"):
        report.add("Avatar", BLOCKER,
                   f"'{source}' is a stylized mascot but '{provider}' needs a human face — "
                   "it won't animate. Use wan-s2v (no face detector) or d-id.")
    if provider in ("sadtalker", "wav2lip", "wan-s2v", "comfyui", "d-id") and not source:
        report.add("Avatar", BLOCKER, f"'{provider}' needs a source image/video — none configured")

    mins = _AVATAR_MINUTES.get(provider)
    if mins and mins >= 5:
        report.add("Avatar", WARN,
                   f"{provider} is ~{mins:.0f} min/slide — a large deck will be very slow; "
                   "consider enabling the avatar on fewer slides")


def _check_providers_env(report, providers, settings, options) -> None:
    strict = settings.get("strict", False)

    def key_present(name: str, cfg: dict[str, Any], keymap: dict[str, str]) -> bool | None:
        env = keymap.get(name)
        if env:
            return bool(os.getenv(env) or cfg.get("api_key"))
        if name in ("openai-compatible", "swarmui", "voicebox", "chatterbox"):
            return bool(cfg.get("base_url") or os.getenv(f"{name.upper().replace('-', '_')}_BASE_URL"))
        return None  # no key needed (text/local/gtts/kokoro/none)

    llm = providers.get("llm", {}).get("provider", "none")
    if llm != "none":
        # A missing SDK aborts the whole render (get_llm_client raises), so it's
        # a blocker regardless of strict mode — flag it before the key check.
        extra = _missing_extra(llm, _LLM_PKG)
        if extra:
            report.add("Providers & environment", BLOCKER,
                       f"LLM '{llm}' package not installed — run: "
                       f'pip install "slide-stream[{extra}]"')
        ok = key_present(llm, providers.get("llm", {}), _LLM_KEY)
        if ok is False:
            report.add("Providers & environment",
                       BLOCKER if strict else WARN,
                       f"LLM '{llm}' has no API key/base_url set — narration will fail"
                       + ("" if strict else " (falls back to un-narrated)"))

    img = providers.get("images", {}).get("provider", "text")
    extra = _missing_extra(img, _IMAGE_PKG)
    if extra:
        report.add("Providers & environment", BLOCKER if strict else WARN,
                   f"image provider '{img}' package not installed — run: "
                   f'pip install "slide-stream[{extra}]"'
                   + ("" if strict else " (falls back to text cards)"))
    ok = key_present(img, providers.get("images", {}), _IMAGE_KEY)
    if ok is False:
        report.add("Providers & environment", WARN,
                   f"image provider '{img}' has no key/base_url — will fall back to text cards")

    tts = providers.get("tts", {}).get("provider", "gtts")
    extra = _missing_extra(tts, _TTS_PKG)
    if extra:
        report.add("Providers & environment", BLOCKER if strict else WARN,
                   f"TTS '{tts}' package not installed — run: "
                   f'pip install "slide-stream[{extra}]"'
                   + ("" if strict else " (falls back to free gTTS)"))
    ok = key_present(tts, providers.get("tts", {}), _TTS_KEY)
    if ok is False:
        report.add("Providers & environment", WARN,
                   f"TTS '{tts}' has no key/base_url — will fall back to free gTTS")

    # ffmpeg is required for voice cloning and avatar compositing.
    needs_ffmpeg = options.get("avatar_enabled") or providers.get("tts", {}).get("voice_sample")
    if needs_ffmpeg and not shutil.which("ffmpeg"):
        report.add("Providers & environment", BLOCKER,
                   "ffmpeg not found — voice cloning & avatars need it "
                   "(brew install ffmpeg / distro package)")

    out = options.get("output_path")
    if out and Path(str(out)).exists():
        report.add("Providers & environment", WARN, f"output {out} exists — it will be overwritten")

    if strict:
        report.add("Providers & environment", OK, "strict mode on — no silent provider fallbacks")


def _estimates(report, slides, providers, per_slide, video) -> None:
    pad = float(video.get("slide_duration_padding", 1.0) or 0)
    total = sum(per_slide) + pad * len(slides)
    report.estimates.append(f"Video length ~{_fmt_dur(total)} ({len(slides)} slides)")

    img_provider = providers.get("images", {}).get("provider", "text")
    per_img = _IMAGE_COST.get(img_provider)
    if per_img:
        report.estimates.append(
            f"Images ~${per_img * len(slides):.2f} ({img_provider}, ~${per_img:.2f}/slide)")

    llm = providers.get("llm", {}).get("provider", "none")
    if llm != "none":
        report.estimates.append(f"LLM: ~{len(slides)} narration call(s) via {llm}")

    avatar = providers.get("avatar", {})
    if avatar.get("provider", "none") != "none":
        mins = _AVATAR_MINUTES.get(avatar["provider"])
        if mins:
            report.estimates.append(
                f"Render ~{_fmt_dur(mins * 60 * len(slides))} — {avatar['provider']} "
                f"~{mins:.0f} min/slide (rough)")


def _ids(nums: list[int]) -> str:
    shown = ", ".join(f"#{n}" for n in nums[:8])
    return shown + (f" …+{len(nums) - 8}" if len(nums) > 8 else "")


def render_report(report: DoctorReport, console: Any, header: str) -> None:
    """Print a grouped, severity-coloured report to the console."""
    from rich.markup import escape
    from rich.panel import Panel

    console.print(Panel.fit(f"[bold cyan]🩺 {header}[/bold cyan]", border_style="green"))
    by_group: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_group.setdefault(f.group, []).append(f)

    style = {OK: "green", WARN: "yellow", BLOCKER: "bold red"}
    for group in GROUPS:
        items = by_group.get(group)
        if not items:
            continue
        console.print(f"\n[bold]{group}[/bold]")
        for f in items:
            # Escape the message: it is plain text and may contain [...] (e.g.
            # a pip extra like slide-stream[claude]), which Rich would otherwise
            # parse as markup and drop.
            console.print(f"  {_ICON[f.severity]} [{style[f.severity]}]{escape(f.message)}[/{style[f.severity]}]")

    if report.estimates:
        console.print("\n[bold]Estimates[/bold]")
        for line in report.estimates:
            console.print(f"  • {line}")

    b, w = report.blockers, report.warnings
    verdict = (f"[bold red]{b} blocker(s)[/bold red] · " if b else "") + \
              (f"[yellow]{w} warning(s)[/yellow]" if w else "[green]no warnings[/green]")
    console.print(f"\n[bold]Summary:[/bold] {verdict}")
