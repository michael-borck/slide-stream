"""Command line interface for Slide Stream."""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from moviepy import VideoFileClip, concatenate_videoclips
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .config_loader import ConfigurationError, load_config, save_example_config
from .llm import get_llm_client, query_llm, query_llm_with_image
from .media import create_video_fragment
from .narration import (
    build_narration_prompt,
    narration_source,
    parse_script_file,
    target_words,
)
from .parser import parse_markdown
from .powerpoint import format_powerpoint_content_for_llm, parse_powerpoint
from .providers.base import StrictModeError
from .providers.factory import ProviderFactory

# Rich Console Initialization
console = Console()
err_console = Console(stderr=True, style="bold red")

# Typer Application Initialization
app = typer.Typer(
    name="slide-stream",
    help="""
    SlideStream: Create professional video presentations from Markdown and PowerPoint files using AI-powered content enhancement.
    """,
    add_completion=False,
    rich_markup_mode="markdown",
    invoke_without_command=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the application version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """SlideStream: create professional video presentations from Markdown and PowerPoint files."""
    if version:
        console.print(
            f"[bold cyan]SlideStream[/bold cyan] version: [yellow]{__version__}[/yellow]"
        )
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # No subcommand given: show help instead of erroring out.
        console.print(ctx.get_help())
        raise typer.Exit()


def _clean_narration(title: str, content: list[Any], notes: str = "") -> str:
    """Build clean, speakable narration from slide text (no markup labels).

    This is the default voiceover used when LLM enhancement is disabled or when
    an LLM call fails, so the audio never reads scaffolding like 'Title:' or
    'Content:' verbatim.
    """
    parts: list[str] = []
    if title.strip():
        parts.append(title.strip())
    for item in content:
        text = str(item).strip()
        if text:
            parts.append(text)
    if notes.strip() and notes.strip() != "Click to add notes":
        parts.append(notes.strip())
    return ". ".join(parts)


def _slide_query(
    title: str, content: list[Any], default_title: str | None
) -> str:
    """Choose a meaningful image/search query for a slide.

    Prefers the title, but falls back to the first content line for slides whose
    title is still the default placeholder (e.g. an untitled 'Slide 3').
    """
    if title and default_title is not None and title == default_title:
        title = ""
    if title.strip():
        return title.strip()
    for item in content:
        text = str(item).strip()
        if text:
            return text
    return title.strip() or "presentation slide"


@app.command()
def create(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the input file (Markdown .md or PowerPoint .pptx).",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Argument(
            help="Filename for the output video.",
        ),
    ],
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (YAML).",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help=(
                "Fail instead of silently falling back when a configured "
                "provider is unusable or errors (e.g. elevenlabs -> gtts)."
            ),
        ),
    ] = False,
    avatar: Annotated[
        bool | None,
        typer.Option(
            "--avatar/--no-avatar",
            help=(
                "Enable or disable the talking-head avatar overlay for this "
                "run, overriding the config file."
            ),
        ),
    ] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider",
            help=(
                "LLM provider for narration (none, openai, gemini, claude, "
                "groq, ollama, openai-compatible), overriding the config file."
            ),
        ),
    ] = None,
    llm_model_option: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="Specific LLM model to use (e.g. claude-haiku-4-5), overriding the config file.",
        ),
    ] = None,
    narration_seconds: Annotated[
        float | None,
        typer.Option(
            "--narration-seconds",
            help=(
                "Approximate spoken length per slide in seconds; notes/content "
                "are summarised or expanded to fit."
            ),
        ),
    ] = None,
    tts_provider_option: Annotated[
        str | None,
        typer.Option(
            "--tts-provider",
            help=(
                "TTS provider (gtts, kokoro, chatterbox, elevenlabs, openai, "
                "openai-compatible), overriding the config file."
            ),
        ),
    ] = None,
    tts_base_url: Annotated[
        str | None,
        typer.Option(
            "--tts-base-url",
            help="TTS server URL (e.g. a Chatterbox or OpenAI-compatible endpoint), overriding the config file.",
        ),
    ] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            "--voice",
            help="Voice name/ID for the TTS provider, overriding the config file.",
        ),
    ] = None,
    voice_sample: Annotated[
        str | None,
        typer.Option(
            "--voice-sample",
            help=(
                "Path to a reference recording for voice cloning (chatterbox); "
                "uploaded ephemerally per run."
            ),
        ),
    ] = None,
    verbatim_notes: Annotated[
        bool,
        typer.Option(
            "--verbatim-notes",
            help=(
                "Speak the PowerPoint speaker notes exactly as written (no LLM "
                "rewriting). Slides without notes fall back to normal narration."
            ),
        ),
    ] = False,
    script_file: Annotated[
        str | None,
        typer.Option(
            "--script",
            help=(
                "Path to a narration script file: one spoken block per slide, "
                "blocks separated by a line of three dashes (---). Used "
                "verbatim, in order, overriding notes and the LLM."
            ),
        ),
    ] = None,
) -> None:
    """Create a video from a Markdown (.md) or PowerPoint (.pptx) file."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 Starting SlideStream! 🚀[/bold cyan]",
            border_style="green",
        )
    )

    # Load configuration
    try:
        config = load_config(config_file)
    except ConfigurationError as e:
        err_console.print(f"Configuration Error: {e}")
        raise typer.Exit(code=1)

    # The CLI flag turns strict mode on for this run; it never turns it off,
    # so a config file with `strict: true` still applies without the flag.
    if strict:
        config["settings"]["strict"] = True

    if llm_provider:
        config["providers"]["llm"]["provider"] = llm_provider
    if llm_model_option:
        config["providers"]["llm"]["model"] = llm_model_option
    if narration_seconds is not None:
        config["settings"].setdefault("narration", {})["target_seconds"] = narration_seconds
    if tts_provider_option:
        config["providers"]["tts"]["provider"] = tts_provider_option
    if tts_base_url:
        config["providers"]["tts"]["base_url"] = tts_base_url
    if voice:
        config["providers"]["tts"]["voice"] = voice
    if voice_sample:
        config["providers"]["tts"]["voice_sample"] = voice_sample

    if avatar is False:
        config["providers"]["avatar"]["provider"] = "none"
    elif avatar is True and config["providers"]["avatar"].get("provider", "none") == "none":
        err_console.print(
            "--avatar requires an avatar provider in the config file "
            "(e.g. providers.avatar.provider: precomputed)."
        )
        raise typer.Exit(code=1)

    # FFmpeg is required to encode video; fail early with an actionable hint.
    if not shutil.which("ffmpeg"):
        err_console.print(
            "FFmpeg was not found on your PATH. SlideStream needs FFmpeg to "
            "encode video. Install it (e.g. 'brew install ffmpeg' or "
            "'sudo apt install ffmpeg') and try again."
        )
        raise typer.Exit(code=1)

    # Check if input file exists
    if not input_path.exists():
        err_console.print(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)

    # Determine file type and validate
    file_extension = input_path.suffix.lower()
    if file_extension not in [".md", ".pptx"]:
        err_console.print(f"Unsupported file type: {file_extension}. Supported: .md, .pptx")
        raise typer.Exit(code=1)

    # Setup a unique temporary directory. Creating our own subdirectory (rather
    # than reusing the configured path directly) guarantees we can never delete
    # files we did not create, and that concurrent runs never collide.
    base_temp_dir = Path(config["settings"]["temp_dir"])
    base_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="slide_stream_", dir=str(base_temp_dir)))

    try:
        # Initialize providers
        try:
            image_provider = ProviderFactory.create_image_provider(config)
            tts_provider = ProviderFactory.create_tts_provider(config)
            avatar_provider = ProviderFactory.create_avatar_provider(config)
        except StrictModeError as e:
            err_console.print(f"{e}")
            raise typer.Exit(code=1)

        # Initialize LLM client
        llm_client = None
        llm_provider_name = config["providers"]["llm"]["provider"]
        llm_model = config["providers"]["llm"]["model"]
        llm_base_url = config["providers"]["llm"].get("base_url")

        if llm_provider_name != "none":
            try:
                llm_client = get_llm_client(llm_provider_name, base_url=llm_base_url)
                console.print(
                    f"✅ LLM Provider: [bold green]{llm_provider_name}[/bold green]"
                )
            except (ImportError, ValueError) as e:
                err_console.print(f"Error initializing LLM: {e}")
                raise typer.Exit(code=1)

        # Parse the input file
        if file_extension == ".md":
            console.print("\n[bold]1. Parsing Markdown...[/bold]")
            with open(input_path, encoding='utf-8') as f:
                markdown_input = f.read()
            if not markdown_input.strip():
                err_console.print("Markdown file is empty. Exiting.")
                raise typer.Exit(code=1)
            slides = parse_markdown(markdown_input)
        else:  # .pptx
            console.print("\n[bold]1. Parsing PowerPoint...[/bold]")
            try:
                slides = parse_powerpoint(input_path)
            except ValueError as e:
                err_console.print(f"Error parsing PowerPoint: {e}")
                raise typer.Exit(code=1)

        if not slides:
            err_console.print(f"No slides found in the {file_extension} file. Exiting.")
            raise typer.Exit(code=1)
        console.print(f"📄 Found [bold yellow]{len(slides)}[/bold yellow] slides.")

        # Optional external narration script (verbatim, one block per slide).
        script_blocks: list[str] | None = None
        if script_file:
            try:
                script_blocks = parse_script_file(script_file)
            except OSError as e:
                err_console.print(f"Could not read script file: {e}")
                raise typer.Exit(code=1)
            if len(script_blocks) != len(slides):
                err_console.print(
                    f"⚠️  Script has {len(script_blocks)} block(s) but the deck "
                    f"has {len(slides)} slide(s). Blocks are matched in order; "
                    "extra slides use their default narration and extra blocks "
                    "are ignored."
                )

        # Process each slide with Rich progress bar
        video_fragments = []
        audio_failed = 0
        llm_narration_failed = 0
        avatar_failed = 0
        avatar_enabled = avatar_provider.name != "none"
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            process_task = progress.add_task(
                "[yellow]Processing Slides...", total=len(slides)
            )

            for i, slide in enumerate(slides):
                slide_num = i + 1
                progress.update(
                    process_task,
                    description=f"[yellow]Processing Slide {slide_num}/{len(slides)}: '{slide['title']}'[/yellow]",
                )

                # raw_text is the structured prompt fed to the LLM only.
                if file_extension == ".md":
                    raw_text = f"Title: {slide['title']}. Content: {' '.join(slide['content'])}"
                else:  # .pptx
                    raw_text = format_powerpoint_content_for_llm(slide)

                # Default narration is clean spoken text (no 'Title:'/'Content:'
                # labels), used when LLM enhancement is off or fails.
                default_title = f"Slide {slide_num}" if file_extension == ".pptx" else None
                effective_title = (
                    slide["title"]
                    if default_title is None or slide["title"] != default_title
                    else ""
                )
                notes = slide.get("notes", "") if file_extension == ".pptx" else ""
                speech_text = _clean_narration(
                    effective_title, slide.get("content", []), notes
                )

                # Meaningful image/search query (falls back past placeholder titles).
                search_query = _slide_query(
                    slide["title"], slide.get("content", []), default_title
                )

                # Verbatim overrides (no LLM): an external script block for
                # this slide, or --verbatim-notes when the slide has notes.
                script_block = (
                    script_blocks[i]
                    if script_blocks is not None and i < len(script_blocks)
                    else None
                )
                if script_block:
                    speech_text = script_block
                elif verbatim_notes and str(slide.get("notes", "")).strip():
                    speech_text = str(slide["notes"]).strip()

                # LLM narration, unless a verbatim source already set the text.
                # Source priority: speaker notes (cleaned up and fitted to the
                # target length) > slide content (written as presenter speech,
                # not read aloud) > slide image (vision) > title only.
                elif llm_client:
                    narration_settings = config["settings"].get("narration", {})
                    words = target_words(
                        narration_settings.get("target_seconds"),
                        narration_settings.get("wpm", 150),
                    )
                    source = narration_source(slide)
                    speech_prompt = build_narration_prompt(
                        slide, source, words, narration_settings.get("wpm", 150)
                    )

                    if source == "image":
                        if slide["title"] == (default_title or ""):
                            err_console.print(
                                f"  - Warning: slide {slide_num} is image-only "
                                "with no title; narration is based on the "
                                "image alone."
                            )
                        image = slide["images"][0]
                        natural_speech = query_llm_with_image(
                            llm_client,
                            llm_provider_name,
                            speech_prompt,
                            image["data"],
                            image["content_type"],
                            console,
                            llm_model,
                        )
                    else:
                        natural_speech = query_llm(
                            llm_client, llm_provider_name, speech_prompt, console, llm_model
                        )

                    if natural_speech:
                        speech_text = natural_speech.strip()
                    else:
                        llm_narration_failed += 1

                    # Only spend an LLM round-trip refining the image query when
                    # the active image provider actually consumes a query; the
                    # text provider renders slide content directly instead.
                    if image_provider.name != "text":
                        search_prompt = f"Generate a concise, descriptive search query for finding a high-quality, relevant image for this topic. Output only the query. Topic:\n\n{raw_text}"
                        improved_query = query_llm(
                            llm_client, llm_provider_name, search_prompt, console, llm_model
                        )
                        if improved_query:
                            search_query = improved_query.strip().replace('"', "")

                # File paths
                img_path = temp_dir / f"slide_{slide_num}.png"
                audio_path = temp_dir / f"slide_{slide_num}.mp3"
                fragment_path = temp_dir / f"fragment_{slide_num}.mp4"

                # Generate image, audio, and video
                strict_mode = config["settings"].get("strict", False)
                try:
                    image_provider.generate_image(
                        search_query, str(img_path), slide=slide
                    )
                except StrictModeError as e:
                    err_console.print(f"Slide {slide_num}: {e}")
                    raise typer.Exit(code=1)
                audio_file = tts_provider.synthesize(speech_text, str(audio_path))
                if audio_file is None:
                    if strict_mode:
                        err_console.print(
                            f"Slide {slide_num}: audio generation failed and "
                            "strict mode is enabled. Aborting."
                        )
                        raise typer.Exit(code=1)
                    audio_failed += 1

                # Talking-head overlay, driven by the slide's narration audio.
                head_path = None
                if avatar_enabled and audio_file:
                    head_path = avatar_provider.generate(
                        str(audio_path),
                        str(temp_dir / f"head_{slide_num}.mp4"),
                        slide_num,
                    )
                    if head_path is None:
                        if strict_mode:
                            err_console.print(
                                f"Slide {slide_num}: avatar generation failed "
                                "and strict mode is enabled. Aborting."
                            )
                            raise typer.Exit(code=1)
                        avatar_failed += 1

                fragment_file = create_video_fragment(
                    str(img_path),
                    str(audio_path) if audio_file else None,
                    str(fragment_path),
                    config,
                    head_video=head_path,
                )

                if fragment_file:
                    video_fragments.append(fragment_file)
                elif strict_mode:
                    err_console.print(
                        f"Slide {slide_num}: video fragment creation failed and "
                        "strict mode is enabled. Aborting."
                    )
                    raise typer.Exit(code=1)

                progress.update(process_task, advance=1)

        # Per-run summary: surface partial output and degradation rather than
        # always declaring total success.
        summary = (
            f"📊 Rendered [bold]{len(video_fragments)}/{len(slides)}[/bold] slides"
            f" · [yellow]{audio_failed}[/yellow] without audio"
        )
        if llm_client:
            summary += f" · [red]{llm_narration_failed}[/red] LLM narration failure(s)"
        if avatar_enabled:
            summary += f" · [yellow]{avatar_failed}[/yellow] slide(s) without avatar"
        if len(video_fragments) < len(slides) or audio_failed or avatar_failed:
            summary += "  [dim](output is incomplete)[/dim]"
        console.print(summary)

        # Combine video fragments
        console.print("\n[bold]2. Combining Video Fragments...[/bold]")
        if video_fragments:
            clips = []
            final_clip = None
            try:
                clips = [VideoFileClip(f) for f in video_fragments]
                final_clip = concatenate_videoclips(clips)

                video_settings = config["settings"]["video"]
                final_clip.write_videofile(
                    output_filename,
                    fps=video_settings["fps"],
                    codec=video_settings["codec"],
                    audio_codec=video_settings["audio_codec"],
                    logger=None,
                )

                console.print(
                    Panel(
                        f"🎉 [bold green]Video creation complete![/bold green] 🎉\n\nOutput file: [yellow]{output_filename}[/yellow]",
                        border_style="green",
                        expand=False,
                    )
                )
            except Exception as e:
                err_console.print(f"Error combining video fragments: {e}")
                raise typer.Exit(code=1)
            finally:
                # Release every clip even if concatenation or encoding raised.
                for clip in clips:
                    clip.close()
                if final_clip is not None:
                    final_clip.close()
        else:
            err_console.print(
                "No video fragments were created, so the final video could not be generated."
            )
            raise typer.Exit(code=1)

    finally:
        # Always remove our unique temp directory on every exit path (success or
        # failure) when cleanup is enabled, so error paths never leak
        # slide_stream_* directories on disk.
        if config.get("settings", {}).get("cleanup", True):
            console.print("\n[bold]3. Cleaning up temporary files...[/bold]")
            shutil.rmtree(temp_dir, ignore_errors=True)
            console.print("✅ Cleanup complete.")


def _parse_deck(input_path: Path) -> list[dict[str, Any]]:
    """Parse a .md or .pptx deck into slide dicts, or exit with an error."""
    ext = input_path.suffix.lower()
    if not input_path.exists():
        err_console.print(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)
    if ext == ".md":
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            err_console.print("Markdown file is empty. Exiting.")
            raise typer.Exit(code=1)
        return parse_markdown(text)
    if ext == ".pptx":
        try:
            return parse_powerpoint(input_path)
        except ValueError as e:
            err_console.print(f"Error parsing PowerPoint: {e}")
            raise typer.Exit(code=1)
    err_console.print(f"Unsupported file type: {ext}. Supported: .md, .pptx")
    raise typer.Exit(code=1)


@app.command()
def enrich(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input deck (Markdown .md or PowerPoint .pptx)."),
    ],
    output_dir: Annotated[
        str,
        typer.Argument(help="Directory for the enriched deck and images."),
    ] = "enriched",
    config_file: Annotated[
        str | None, typer.Option("--config", "-c", help="Config file (YAML).")
    ] = None,
    image_provider_option: Annotated[
        str | None,
        typer.Option(
            "--image-provider",
            help="Image provider (local, dalle3, pexels, unsplash, openai-compatible).",
        ),
    ] = None,
    image_folder: Annotated[
        str | None,
        typer.Option("--image-folder", help="Folder of images for the 'local' provider."),
    ] = None,
    pptx: Annotated[
        bool,
        typer.Option("--pptx", help="Also write an enriched PowerPoint (.pptx)."),
    ] = False,
    zip_output: Annotated[
        bool, typer.Option("--zip", help="Also write a .zip of the output folder.")
    ] = False,
) -> None:
    """Add an image to each slide and write a new deck (no video).

    The output is an editable Markdown deck plus an images/ folder — run
    'create' on it to narrate, or use 'create' directly for a one-pass video.
    """
    console.print(
        Panel.fit("[bold cyan]🖼️  Enriching deck[/bold cyan]", border_style="green")
    )
    try:
        config = load_config(config_file)
    except ConfigurationError as e:
        err_console.print(f"Configuration Error: {e}")
        raise typer.Exit(code=1)

    if image_provider_option:
        config["providers"]["images"]["provider"] = image_provider_option
    if image_folder:
        config["providers"]["images"]["folder"] = image_folder

    slides = _parse_deck(input_path)
    if not slides:
        err_console.print("No slides found. Exiting.")
        raise typer.Exit(code=1)
    console.print(f"📄 Found [bold yellow]{len(slides)}[/bold yellow] slides.")

    try:
        image_provider = ProviderFactory.create_image_provider(config)
    except StrictModeError as e:
        err_console.print(f"{e}")
        raise typer.Exit(code=1)

    from .enrich import enrich_deck

    out = enrich_deck(
        slides,
        image_provider,
        Path(output_dir),
        input_path.stem,
        also_pptx=pptx,
        also_zip=zip_output,
    )
    console.print(
        Panel(
            f"🎉 [bold green]Enriched deck written[/bold green]\n\n"
            f"Markdown: [yellow]{out / (input_path.stem + '.md')}[/yellow]"
            + (f"\nPowerPoint: [yellow]{out / (input_path.stem + '.pptx')}[/yellow]" if pptx else "")
            + f"\nImages: [yellow]{out / 'images'}[/yellow]",
            border_style="green",
            expand=False,
        )
    )


@app.command()
def scan(
    folder: Annotated[
        Path, typer.Argument(help="Folder of images to AI-rename to keyword slugs.")
    ],
    provider: Annotated[
        str,
        typer.Option("--provider", help="Vision LLM provider (claude, openai, gemini)."),
    ] = "claude",
    model: Annotated[
        str | None, typer.Option("--model", help="Specific vision model to use.")
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Actually rename files (default is a dry-run preview).",
        ),
    ] = False,
) -> None:
    """AI-rename images in a folder to keyword slugs for the 'local' provider."""
    if not folder.is_dir():
        err_console.print(f"Not a directory: {folder}")
        raise typer.Exit(code=1)

    from .scan import apply_renames, build_rename_records, write_scan_report

    try:
        records = build_rename_records(folder, provider, model)
    except (ImportError, ValueError) as e:
        err_console.print(f"Vision provider error: {e}")
        raise typer.Exit(code=1)

    if not records:
        console.print("No images found in the folder.")
        return

    applied = apply_renames(records, dry_run=not apply)

    table = Table(title="Dry run — no files changed" if not apply else "Renamed")
    table.add_column("Original", style="cyan")
    table.add_column("→", style="dim")
    table.add_column("New name", style="green")
    for orig, new in applied:
        table.add_row(orig.name, "→", new.name)
    console.print(table)

    if apply:
        report = write_scan_report(folder, applied)
        console.print(f"📝 Report: [yellow]{report}[/yellow]")
    else:
        console.print("\n[dim]Re-run with --apply to rename the files.[/dim]")


@app.command()
def init(
    output_path: Annotated[
        str,
        typer.Argument(help="Path where to create the example configuration file."),
    ] = "slidestream.yaml",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite the file if it already exists.",
        ),
    ] = False,
) -> None:
    """Create an example configuration file."""
    target = Path(output_path)
    if target.exists() and not force:
        err_console.print(
            f"{output_path} already exists. Re-run with --force to overwrite it."
        )
        raise typer.Exit(code=1)
    save_example_config(output_path)


# Ephemeral per-run voice uploads are named <uuid4>.<ext>; they are internal
# plumbing and must never be shown to users as selectable voices.
_UUID_VOICE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\.[a-z0-9]+)?$",
    re.IGNORECASE,
)


def _is_uuid_voice(name: str) -> bool:
    """True for ephemeral UUID-named voice files (hidden from listings)."""
    return bool(_UUID_VOICE_RE.match(name))


@app.command()
def voices(
    config_file: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to configuration file (YAML)."),
    ] = None,
) -> None:
    """List voices available on the configured Chatterbox/TTS server."""
    import requests

    try:
        config = load_config(config_file)
    except ConfigurationError as e:
        err_console.print(f"Configuration Error: {e}")
        raise typer.Exit(code=1)

    tts_config = config.get("providers", {}).get("tts", {})
    base_url = tts_config.get("base_url")
    if not base_url:
        err_console.print(
            "No TTS server configured (providers.tts.base_url). The 'voices' "
            "command lists voices from a Chatterbox or OpenAI-compatible server."
        )
        raise typer.Exit(code=1)
    base_url = base_url.rstrip("/").removesuffix("/v1")

    api_key = tts_config.get("api_key") or config.get("api_keys", {}).get("chatterbox")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def fetch(path: str) -> list[str]:
        response = requests.get(f"{base_url}{path}", headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("voices", data) if isinstance(data, dict) else data
        return [str(v) for v in items]

    try:
        stock = fetch("/v1/audio/voices")
    except Exception as e:
        err_console.print(f"Could not reach TTS server at {base_url}: {e}")
        raise typer.Exit(code=1)

    try:
        references = fetch("/get_reference_files")
    except Exception:
        references = []  # non-Chatterbox servers don't have this endpoint

    table = Table(title=f"Voices on {base_url}")
    table.add_column("Voice", style="cyan")
    table.add_column("Type")
    for name in sorted(stock):
        if not _is_uuid_voice(name):
            table.add_row(name, "stock")
    for name in sorted(references):
        if not _is_uuid_voice(name):
            table.add_row(name, "reference")
    console.print(table)
    console.print(
        "\n[dim]💡 Use providers.tts.voice for any listed voice, or "
        "providers.tts.voice_sample: /path/to/you.wav for an ephemeral, "
        "privacy-first clone of your own voice (10-30s of clean speech).[/dim]"
    )


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Address to bind (127.0.0.1 = local only).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8080,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="Access token (or SLIDESTREAM_TOKEN). Generated if omitted when not local.",
        ),
    ] = None,
    config_file: Annotated[
        str | None, typer.Option("--config", "-c", help="Server config file (YAML).")
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", help="Concurrent renders (keep low; GPU/CPU heavy).")
    ] = 1,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not auto-open a browser.")
    ] = False,
    demo: Annotated[
        bool,
        typer.Option(
            "--demo",
            help="Show a 'hosted demo — install locally for full control' banner (or SLIDESTREAM_DEMO).",
        ),
    ] = False,
) -> None:
    """Launch the web UI: upload a deck + voice + photo, render, download."""
    try:
        import uvicorn

        from .serve import create_app
    except (ImportError, RuntimeError):
        err_console.print(
            'The web UI needs extra packages. Install with: pip install "slide-stream[serve]"'
        )
        raise typer.Exit(code=1)

    try:
        config = load_config(config_file)
    except ConfigurationError as e:
        err_console.print(f"Configuration Error: {e}")
        raise typer.Exit(code=1)

    resolved_token = token or os.getenv("SLIDESTREAM_TOKEN")
    # On a non-local bind with no token, mint one so the server isn't wide open.
    if not resolved_token and host not in ("127.0.0.1", "localhost"):
        import secrets

        resolved_token = secrets.token_urlsafe(24)
        console.print(f"🔑 Generated access token: [bold yellow]{resolved_token}[/bold yellow]")

    app_instance = create_app(
        config=config, token=resolved_token, max_workers=workers,
        demo=demo or None,
    )

    url = f"http://{host}:{port}"
    console.print(
        Panel.fit(
            f"[bold cyan]🎬 SlideStream server[/bold cyan]\n{url}"
            + ("\n[dim]token required[/dim]" if resolved_token else "\n[dim]no token (local)[/dim]"),
            border_style="green",
        )
    )
    if not no_browser and host in ("127.0.0.1", "localhost"):
        import webbrowser

        webbrowser.open(url)

    uvicorn.run(app_instance, host=host, port=port, log_level="warning")


@app.command()
def providers() -> None:
    """List available providers and their status."""
    try:
        config = load_config()
    except ConfigurationError:
        config = {}

    availability = ProviderFactory.check_provider_availability(config)

    console.print("\n[bold cyan]📋 Available Providers[/bold cyan]\n")

    # Image providers
    console.print("[bold]🖼️  Image Providers[/bold]")
    table = Table()
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    image_providers = ProviderFactory.list_image_providers()
    for name, description in image_providers.items():
        status = "✅ Available" if availability.get("images", {}).get(name, False) else "❌ Unavailable"
        table.add_row(name, status, description)

    console.print(table)

    # TTS providers
    console.print("\n[bold]🎙️  Text-to-Speech Providers[/bold]")
    table = Table()
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    tts_providers = ProviderFactory.list_tts_providers()
    for name, description in tts_providers.items():
        status = "✅ Available" if availability.get("tts", {}).get(name, False) else "❌ Unavailable"
        table.add_row(name, status, description)

    console.print(table)

    # Avatar providers
    console.print("\n[bold]🧑 Avatar Providers[/bold]")
    table = Table()
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    avatar_providers = ProviderFactory.list_avatar_providers()
    for name, description in avatar_providers.items():
        status = "✅ Available" if availability.get("avatar", {}).get(name, False) else "❌ Unavailable"
        table.add_row(name, status, description)

    console.print(table)

    console.print("\n[dim]💡 Tip: Use 'slide-stream init' to create a configuration file[/dim]")


if __name__ == "__main__":
    app()
