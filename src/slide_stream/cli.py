"""Command line interface for Slide Stream."""

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from moviepy import VideoFileClip, concatenate_videoclips
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .config_loader import ConfigurationError, load_config, save_example_config
from .llm import get_llm_client, query_llm
from .media import create_video_fragment
from .parser import parse_markdown
from .powerpoint import format_powerpoint_content_for_llm, parse_powerpoint
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
        Optional[str],
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (YAML).",
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

    # Check if input file exists
    if not input_path.exists():
        err_console.print(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)
    
    # Determine file type and validate
    file_extension = input_path.suffix.lower()
    if file_extension not in [".md", ".pptx"]:
        err_console.print(f"Unsupported file type: {file_extension}. Supported: .md, .pptx")
        raise typer.Exit(code=1)

    # Setup temporary directory
    temp_dir = Path(config["settings"]["temp_dir"])
    temp_dir.mkdir(exist_ok=True)
    
    # Initialize providers
    image_provider = ProviderFactory.create_image_provider(config)
    tts_provider = ProviderFactory.create_tts_provider(config)

    # Initialize LLM client
    llm_client = None
    llm_provider_name = config["providers"]["llm"]["provider"]
    llm_model = config["providers"]["llm"]["model"]
    
    if llm_provider_name != "none":
        try:
            llm_client = get_llm_client(llm_provider_name)
            console.print(
                f"✅ LLM Provider: [bold green]{llm_provider_name}[/bold green]"
            )
        except (ImportError, ValueError) as e:
            err_console.print(f"Error initializing LLM: {e}")
            raise typer.Exit(code=1)

    # Parse the input file
    if file_extension == ".md":
        console.print("\n[bold]1. Parsing Markdown...[/bold]")
        with open(input_path, 'r', encoding='utf-8') as f:
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

            # Format content based on file type
            if file_extension == ".md":
                raw_text = f"Title: {slide['title']}. Content: {' '.join(slide['content'])}"
            else:  # .pptx
                raw_text = format_powerpoint_content_for_llm(slide)
            
            speech_text = raw_text
            search_query = slide["title"]

            # LLM Processing
            if llm_client:
                if file_extension == ".pptx" and slide.get('notes'):
                    speech_prompt = f"Convert the following PowerPoint slide into a natural, flowing script for a voiceover. Use the speaker notes as guidance for the narrative style. Speak conversationally. Directly output the script and nothing else.\n\n{raw_text}"
                else:
                    speech_prompt = f"Convert the following slide points into a natural, flowing script for a voiceover. Speak conversationally. Directly output the script and nothing else.\n\n{raw_text}"
                natural_speech = query_llm(
                    llm_client, llm_provider_name, speech_prompt, console, llm_model
                )
                if natural_speech:
                    speech_text = natural_speech

                # Generate better image search query using LLM
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
            image_provider.generate_image(search_query, str(img_path))
            audio_file = tts_provider.synthesize(speech_text, str(audio_path))
            fragment_file = create_video_fragment(
                str(img_path),
                str(audio_path) if audio_file else None,
                str(fragment_path),
                config
            )

            if fragment_file:
                video_fragments.append(fragment_file)

            progress.update(process_task, advance=1)
            time.sleep(0.1)  # Small delay for smoother progress bar updates

    # Combine video fragments
    console.print("\n[bold]2. Combining Video Fragments...[/bold]")
    if video_fragments:
        try:
            clips = [VideoFileClip(f) for f in video_fragments]
            final_clip = concatenate_videoclips(clips)
            
            video_settings = config["settings"]["video"]
            final_clip.write_videofile(
                output_filename,
                fps=video_settings["fps"],
                codec=video_settings["codec"],
                audio_codec=video_settings["audio_codec"],
                logger=None
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
    if config["settings"]["cleanup"]:
        console.print("\n[bold]3. Cleaning up temporary files...[/bold]")
        try:
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
            temp_dir.rmdir()
            console.print("✅ Cleanup complete.")
        except Exception as e:
            err_console.print(f"Warning: Could not clean up temporary files: {e}")


@app.command()
def init(
    output_path: Annotated[
        str,
        typer.Argument(help="Path where to create the example configuration file.")
    ] = "slidestream.yaml"
) -> None:
    """Create an example configuration file."""
    save_example_config(output_path)


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
    
    console.print("\n[dim]💡 Tip: Use 'slide-stream init' to create a configuration file[/dim]")


if __name__ == "__main__":
    app()