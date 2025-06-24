"""Command line interface for Slide Stream."""

import time
from pathlib import Path
from typing import Annotated

import typer
from moviepy import ImageClip, concatenate_videoclips
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from . import __version__
from .config import TEMP_DIR
from .llm import get_llm_client, query_llm
from .media import (
    create_text_image,
    create_video_fragment,
    search_and_download_image,
    text_to_speech,
)
from .parser import parse_markdown

# Rich Console Initialization
console = Console()
err_console = Console(stderr=True, style="bold red")

# Typer Application Initialization
app = typer.Typer(
    name="slide-stream",
    help="""
    SlideStream: An AI-powered tool to automatically create video presentations from text and Markdown.
    """,
    add_completion=False,
    rich_markup_mode="markdown",
)


def version_callback(value: bool) -> None:
    """Print the version of the application and exit."""
    if value:
        console.print(
            f"[bold cyan]SlideStream[/bold cyan] version: [yellow]{__version__}[/yellow]"
        )
        raise typer.Exit()


@app.command()
def create(
    input_file: Annotated[
        typer.FileText,
        typer.Option(
            "--input",
            "-i",
            help="Path to the input Markdown file.",
            rich_help_panel="Input/Output Options",
        ),
    ],
    output_filename: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Filename for the output video.",
            rich_help_panel="Input/Output Options",
        ),
    ] = "output_video.mp4",
    llm_provider: Annotated[
        str,
        typer.Option(
            help="Select the LLM provider for text enhancement.",
            rich_help_panel="AI & Content Options",
        ),
    ] = "none",
    image_source: Annotated[
        str,
        typer.Option(
            help="Choose the source for slide images.",
            rich_help_panel="AI & Content Options",
        ),
    ] = "unsplash",
    llm_model: Annotated[
        str | None,
        typer.Option(
            help="Specify the model to use for the LLM provider (optional).",
            rich_help_panel="AI & Content Options",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show application version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Create a video from a Markdown file."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 Starting SlideStream! 🚀[/bold cyan]",
            border_style="green",
        )
    )

    # Read input file
    markdown_input = input_file.read()
    if not markdown_input.strip():
        err_console.print("Input file is empty. Exiting.")
        raise typer.Exit(code=1)

    # Setup temporary directory
    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)

    # Initialize LLM client
    llm_client = None
    if llm_provider != "none":
        try:
            llm_client = get_llm_client(llm_provider)
            console.print(
                f"✅ LLM Provider Initialized: [bold green]{llm_provider}[/bold green]"
            )
        except (ImportError, ValueError) as e:
            err_console.print(f"Error initializing LLM: {e}")
            raise typer.Exit(code=1)

    # Parse the Markdown
    console.print("\n[bold]1. Parsing Markdown...[/bold]")
    slides = parse_markdown(markdown_input)
    if not slides:
        err_console.print("No slides found in the Markdown file. Exiting.")
        raise typer.Exit(code=1)
    console.print(f"📄 Found [bold yellow]{len(slides)}[/bold yellow] slides.")

    # Process each slide with Rich progress bar
    video_fragments = []
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

            raw_text = f"Title: {slide['title']}. Content: {' '.join(slide['content'])}"
            speech_text = raw_text
            search_query = slide["title"]

            # LLM Processing
            if llm_client:
                speech_prompt = f"Convert the following slide points into a natural, flowing script for a voiceover. Speak conversationally. Directly output the script and nothing else.\n\n{raw_text}"
                natural_speech = query_llm(
                    llm_client, llm_provider, speech_prompt, console, llm_model
                )
                if natural_speech:
                    speech_text = natural_speech

                if image_source == "unsplash":
                    search_prompt = f"Generate a concise, descriptive search query for a stock photo website (like Unsplash) to find a high-quality, relevant image for this topic. Output only the query. Topic:\n\n{raw_text}"
                    improved_query = query_llm(
                        llm_client, llm_provider, search_prompt, console, llm_model
                    )
                    if improved_query:
                        search_query = improved_query.strip().replace('"', "")

            # File paths
            img_path = temp_dir / f"slide_{slide_num}.png"
            audio_path = temp_dir / f"slide_{slide_num}.mp3"
            fragment_path = temp_dir / f"fragment_{slide_num}.mp4"

            # Image sourcing, audio, and video creation
            if image_source == "unsplash":
                search_and_download_image(search_query, str(img_path))
            else:
                create_text_image(slide["title"], slide["content"], str(img_path))

            audio_file = text_to_speech(speech_text, str(audio_path))
            fragment_file = create_video_fragment(
                str(img_path),
                str(audio_path) if audio_file else None,
                str(fragment_path),
            )

            if fragment_file:
                video_fragments.append(fragment_file)

            progress.update(process_task, advance=1)
            time.sleep(0.1)  # Small delay for smoother progress bar updates

    # Combine video fragments
    console.print("\n[bold]2. Combining Video Fragments...[/bold]")
    if video_fragments:
        try:
            clips = [
                ImageClip(f).set_duration(ImageClip(f).duration)  # type: ignore[attr-defined]
                for f in video_fragments
            ]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(
                output_filename, fps=24, codec="libx264", audio_codec="aac", logger=None
            )

            # Clean up clips
            for clip in clips:
                clip.close()
            final_clip.close()

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
    else:
        err_console.print(
            "No video fragments were created, so the final video could not be generated."
        )
        raise typer.Exit(code=1)

    # Cleanup
    console.print("\n[bold]3. Cleaning up temporary files...[/bold]")
    try:
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
        temp_dir.rmdir()
        console.print("✅ Cleanup complete.")
    except Exception as e:
        err_console.print(f"Warning: Could not clean up temporary files: {e}")


if __name__ == "__main__":
    app()
