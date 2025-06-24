import os
import time
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.markdown import Markdown as RichMarkdown
from typing_extensions import Annotated

# --- Assume all previous core functions are in a separate file or below ---
# To keep this clean, let's imagine the functions from the previous script are here.
# (I will include them at the bottom for a complete, runnable script)

# --- Rich Console Initialization ---
console = Console()
err_console = Console(stderr=True, style="bold red")

# --- Typer Application Initialization ---
app = typer.Typer(
    name="slidestream",
    help="""
    SlideStream: An AI-powered tool to automatically create video presentations from text and Markdown.
    """,
    add_completion=False,
    rich_markup_mode="markdown"
)

def version_callback(value: bool):
    """Prints the version of the application and exits."""
    if value:
        console.print("[bold cyan]SlideStream[/bold cyan] version: [yellow]1.0.0[/yellow]")
        raise typer.Exit()

@app.command()
def create(
    input_file: Annotated[typer.FileText,
        typer.Option(
            "--input", "-i",
            help="Path to the input Markdown file.",
            rich_help_panel="Input/Output Options"
        )],
    output_filename: Annotated[str,
        typer.Option(
            "--output", "-o",
            help="Filename for the output video.",
            rich_help_panel="Input/Output Options"
        )] = "output_video.mp4",
    llm_provider: Annotated[str,
        typer.Option(
            help="Select the LLM provider for text enhancement.",
            rich_help_panel="AI & Content Options"
        )] = "none",
    image_source: Annotated[str,
        typer.Option(
            help="Choose the source for slide images.",
            rich_help_panel="AI & Content Options"
        )] = "unsplash",
    version: Annotated[bool,
        typer.Option(
            "--version",
            help="Show application version and exit.",
            callback=version_callback,
            is_eager=True,
        )] = False,
):
    """
    Creates a video from a Markdown file.
    """
    console.print(Panel.fit("[bold cyan]🚀 Starting SlideStream! 🚀[/bold cyan]", border_style="green"))

    markdown_input = input_file.read()
    if not markdown_input.strip():
        err_console.print("Input file is empty. Exiting.")
        raise typer.Exit(code=1)

    # --- Setup Temporary Directory ---
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # --- Initialize LLM Client ---
    llm_client = None
    if llm_provider != 'none':
        try:
            llm_client = get_llm_client(llm_provider)
            console.print(f"✅ LLM Provider Initialized: [bold green]{llm_provider}[/bold green]")
        except (ImportError, ValueError) as e:
            err_console.print(f"Error initializing LLM: {e}")
            raise typer.Exit(code=1)

    # --- Parse the Markdown ---
    console.print("\n[bold]1. Parsing Markdown...[/bold]")
    slides = parse_markdown(markdown_input)
    if not slides:
        err_console.print("No slides found in the Markdown file. Exiting.")
        raise typer.Exit(code=1)
    console.print(f"📄 Found [bold yellow]{len(slides)}[/bold yellow] slides.")


    # --- Process Each Slide with Rich Progress Bar ---
    video_fragments = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        process_task = progress.add_task("[yellow]Processing Slides...", total=len(slides))

        for i, slide in enumerate(slides):
            slide_num = i + 1
            progress.update(process_task, description=f"[yellow]Processing Slide {slide_num}/{len(slides)}: '{slide['title']}'[/yellow]")

            raw_text = f"Title: {slide['title']}. Content: {' '.join(slide['content'])}"
            speech_text = raw_text
            search_query = slide['title']

            # --- LLM Processing ---
            if llm_client:
                speech_prompt = f"Convert the following slide points into a natural, flowing script for a voiceover. Speak conversationally. Directly output the script and nothing else.\n\n{raw_text}"
                natural_speech = query_llm(llm_client, llm_provider, speech_prompt, console)
                if natural_speech:
                    speech_text = natural_speech

                if image_source == 'unsplash':
                    search_prompt = f"Generate a concise, descriptive search query for a stock photo website (like Unsplash) to find a high-quality, relevant image for this topic. Output only the query. Topic:\n\n{raw_text}"
                    improved_query = query_llm(llm_client, llm_provider, search_prompt, console)
                    if improved_query:
                        search_query = improved_query.strip().replace('"', '')

            # --- File Paths ---
            img_path = os.path.join(TEMP_DIR, f"slide_{slide_num}.png")
            audio_path = os.path.join(TEMP_DIR, f"slide_{slide_num}.mp3")
            fragment_path = os.path.join(TEMP_DIR, f"fragment_{slide_num}.mp4")

            # --- Image Sourcing, Audio, and Video Creation ---
            if image_source == 'unsplash':
                search_and_download_image(search_query, img_path)
            else:
                create_text_image(slide['title'], slide['content'], img_path)

            text_to_speech(speech_text, audio_path)
            if create_video_fragment(img_path, audio_path, fragment_path):
                video_fragments.append(fragment_path)

            progress.update(process_task, advance=1)
            time.sleep(0.5) # Small delay for smoother progress bar updates

    console.print("\n[bold]2. Combining Video Fragments...[/bold]")
    if video_fragments:
        final_clip = concatenate_videoclips([ImageClip(f).set_duration(ImageClip(f).duration) for f in video_fragments])
        final_clip.write_videofile(output_filename, fps=24, codec='libx264', audio_codec='aac', logger=None)
        console.print(Panel(f"🎉 [bold green]Video creation complete![/bold green] 🎉\n\nOutput file: [yellow]{output_filename}[/yellow]", border_style="green", expand=False))
    else:
        err_console.print("No video fragments were created, so the final video could not be generated.")

    # --- Cleanup ---
    console.print("\n[bold]3. Cleaning up temporary files...[/bold]")
    for file in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file))
    os.rmdir(TEMP_DIR)
    console.print("✅ Cleanup complete.")

# ==============================================================================
#  HELPER FUNCTIONS (Adapted from previous script to work with Rich console)
# ==============================================================================

import markdown
from bs4 import BeautifulSoup
import requests
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
import textwrap

# --- Configuration ---
VIDEO_RESOLUTION = (1920, 1080)
SLIDE_DURATION_PADDING = 1.0
DEFAULT_SLIDE_DURATION = 5.0
IMAGE_DOWNLOAD_TIMEOUT = 15
TEMP_DIR = "temp_files"
BG_COLOR = "black"

def get_llm_client(provider):
    if provider == 'gemini':
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        except ImportError:
            raise ImportError("Gemini library not found. Please run 'pip install google-generativeai'")
    # ... (include other LLM provider setups: openai, claude, groq, ollama)
    elif provider in ['openai', 'ollama']:
        try:
            from openai import OpenAI
            if provider == 'ollama':
                base_url = os.getenv("OLLAMA_BASE_API")
                if not base_url: raise ValueError("OLLAMA_BASE_API environment variable not set (e.g., http://localhost:11434).")
                return OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
                return OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI library not found. Please run 'pip install openai'")
    # ... and so on for other providers
    return None

def query_llm(client, provider, prompt_text, rich_console):
    rich_console.print("  - Querying LLM...")
    try:
        # ... (LLM querying logic from previous script) ...
        if provider == 'gemini':
            response = client.generate_content(prompt_text)
            return response.text
        elif provider in ['openai', 'ollama', 'groq']:
            # Determine model...
            model = 'gpt-4o' # Simplified for brevity
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt_text}])
            return response.choices[0].message.content
        return None # Placeholder for other providers
    except Exception as e:
        err_console.print(f"  - LLM Error: {e}")
        return None

def parse_markdown(markdown_text):
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    slides = []
    for header in soup.find_all('h1'):
        slide_title = header.get_text()
        content_items = [item.get_text() for item in header.find_next_sibling('ul', 'ol').find_all('li')] if header.find_next_sibling('ul', 'ol') else []
        slides.append({'title': slide_title, 'content': content_items})
    return slides

def search_and_download_image(query, filename):
    try:
        url = f"https://source.unsplash.com/random/{VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}/?{query.replace(' ', ',')}"
        response = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except requests.exceptions.RequestException as e:
        err_console.print(f"  - Image download error: {e}. Using a placeholder.")
        return create_text_image("Image not found", [f"Query: {query}"], filename)

def create_text_image(title, content_items, filename):
    img = Image.new('RGB', VIDEO_RESOLUTION, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 100)
        content_font = ImageFont.truetype("arial.ttf", 60)
    except IOError:
        title_font = ImageFont.load_default()
        content_font = ImageFont.load_default()
    draw.text((VIDEO_RESOLUTION[0] * 0.1, VIDEO_RESOLUTION[1] * 0.1), title, font=title_font, fill='white')
    y_pos = VIDEO_RESOLUTION[1] * 0.3
    for item in content_items:
        wrapped_lines = textwrap.wrap(f"• {item}", width=50)
        for line in wrapped_lines:
            draw.text((VIDEO_RESOLUTION[0] * 0.1, y_pos), line, font=content_font, fill='white')
            y_pos += 70
        y_pos += 30
    img.save(filename)
    return filename

def text_to_speech(text, filename):
    try:
        gTTS(text=text, lang='en').save(filename)
        return filename
    except Exception as e:
        err_console.print(f"  - Audio generation error: {e}")
        return None

def create_video_fragment(image_path, audio_path, output_path):
    try:
        audio_clip = AudioFileClip(audio_path) if audio_path and os.path.exists(audio_path) else None
        duration = (audio_clip.duration + SLIDE_DURATION_PADDING) if audio_clip else DEFAULT_SLIDE_DURATION
        image_clip = ImageClip(image_path, duration=duration).set_position('center')
        image_clip = image_clip.resize(height=VIDEO_RESOLUTION[1]) if image_clip.h > VIDEO_RESOLUTION[1] else image_clip
        image_clip = image_clip.resize(width=VIDEO_RESOLUTION[0]) if image_clip.w > VIDEO_RESOLUTION[0] else image_clip
        final_clip = image_clip.set_audio(audio_clip) if audio_clip else image_clip
        final_clip.write_videofile(output_path, fps=24, codec='libx264', logger=None)
        return output_path
    except Exception as e:
        err_console.print(f"  - Video fragment creation error: {e}")
        return None

if __name__ == "__main__":
    app()

