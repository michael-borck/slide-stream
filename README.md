# Slide Stream 2.0

<!-- BADGES:START -->
[![ai](https://img.shields.io/badge/-ai-ff6f00?style=flat-square)](https://github.com/topics/ai) [![cli-tool](https://img.shields.io/badge/-cli--tool-blue?style=flat-square)](https://github.com/topics/cli-tool) [![markdown-to-video](https://img.shields.io/badge/-markdown--to--video-blue?style=flat-square)](https://github.com/topics/markdown-to-video) [![natural-language-processing](https://img.shields.io/badge/-natural--language--processing-blue?style=flat-square)](https://github.com/topics/natural-language-processing) [![python](https://img.shields.io/badge/-python-3776ab?style=flat-square)](https://github.com/topics/python) [![text-to-speech](https://img.shields.io/badge/-text--to--speech-blue?style=flat-square)](https://github.com/topics/text-to-speech) [![video-creation](https://img.shields.io/badge/-video--creation-blue?style=flat-square)](https://github.com/topics/video-creation) [![image-sourcing](https://img.shields.io/badge/-image--sourcing-blue?style=flat-square)](https://github.com/topics/image-sourcing) [![powerpoint-to-video](https://img.shields.io/badge/-powerpoint--to--video-blue?style=flat-square)](https://github.com/topics/powerpoint-to-video) [![presentation-automation](https://img.shields.io/badge/-presentation--automation-blue?style=flat-square)](https://github.com/topics/presentation-automation)
<!-- BADGES:END -->

🎬 **Professional AI-powered video presentations from Markdown and PowerPoint files.**

[![PyPI version](https://badge.fury.io/py/slide-stream.svg)](https://badge.fury.io/py/slide-stream)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform your content into stunning video presentations with AI-generated images, premium text-to-speech, and smart content enhancement. **Version 2.0** introduces a modern configuration system and professional-grade providers.

## ✨ What's New in 2.0

- 🖼️ **AI Image Generation**: DALL-E 3 creates custom images for your slides
- 🎙️ **Premium Voices**: ElevenLabs delivers studio-quality narration  
- 📸 **Stock Photos**: Pexels & Unsplash integration with API keys
- ⚙️ **Configuration System**: YAML-based setup with environment variables
- 🎯 **Simplified CLI**: Clean commands, no more option overload
- 🔄 **Smart Fallbacks**: Graceful degradation when services unavailable

## 🚀 Quick Start

### 1. Installation

```bash
# Install with all AI providers
pip install slide-stream[all-ai]

# Or install with specific providers
pip install slide-stream[openai,elevenlabs]
```

### 2. Setup Configuration

```bash
# Create configuration file
slide-stream init

# Check available providers
slide-stream providers
```

### 3. Configure API Keys

Set environment variables for the services you want to use:

```bash
# For AI image generation
export OPENAI_API_KEY="your-openai-key"

# For premium text-to-speech  
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# For stock photos (optional)
export PEXELS_API_KEY="your-pexels-key"
export UNSPLASH_ACCESS_KEY="your-unsplash-key"
```

### 4. Create Your First Video

```bash
# Create from Markdown
slide-stream create presentation.md output.mp4

# Create from PowerPoint
slide-stream create slides.pptx video.mp4
```

## 🎯 Usage Examples

### Basic Video Creation

```bash
# Simple creation (uses default config)
slide-stream create slides.md presentation.mp4

# With custom configuration
slide-stream create --config my-config.yaml presentation.pptx video.mp4
```

### Example Markdown File

```markdown
# Welcome to AI-First Development

- Build smarter applications with integrated AI
- Learn practical implementation patterns
- Deploy production-ready solutions

# Why Choose AI-First?

- Faster development cycles
- Enhanced user experiences  
- Competitive advantage in the market

# Getting Started

- Set up your development environment
- Choose the right AI services
- Build your first AI-powered feature
```

## ⚙️ Configuration

SlideStream layers configuration so you set shared things (a TTS server URL,
API keys) once and keep per-deck settings separate. Later layers win:

1. Built-in defaults
2. **`~/.slidestream.yaml`** — personal: your Chatterbox/LLM server URLs and
   API-key references, shared across every project
3. **`./slidestream.yaml`** (or `--config FILE`) — settings for the deck at hand
4. **CLI flags** (`--voice`, `--tts-base-url`, `--narration-seconds`, …) — win
   over everything, for one-off runs

Run `slide-stream init` to write a starter `slidestream.yaml`. API keys are
read from the environment via `${VAR}` expansion, so secrets never live in the
files. Example:

```yaml
# slidestream.yaml
providers:
  llm:
    provider: openai        # Content enhancement
    model: gpt-4o-mini
    
  images:
    provider: dalle3        # AI-generated images
    fallback: text         # Fallback when DALL-E unavailable
    
  tts:
    provider: elevenlabs   # Premium text-to-speech
    voice: rachel          # Voice selection

# API Keys (use environment variables for security)
api_keys:
  openai: "${OPENAI_API_KEY}"
  elevenlabs: "${ELEVENLABS_API_KEY}"
  pexels: "${PEXELS_API_KEY}"
  unsplash: "${UNSPLASH_ACCESS_KEY}"

settings:
  video:
    resolution: [1920, 1080]
    fps: 24
    codec: libx264
  cleanup: true
```

### Configuration Discovery

SlideStream automatically finds your config in this order:
1. `./slidestream.yaml` (current directory)
2. `~/.slidestream.yaml` (home directory)  
3. Built-in defaults

## 🔧 Available Providers

### Image Providers

| Provider | Description | Requirements |
|----------|------------|--------------|
| `dalle3` | AI image generation via DALL-E 3 | OpenAI API key |
| `pexels` | Professional stock photos | Pexels API key |
| `unsplash` | High-quality stock photos | Unsplash API key |
| `text` | Text-based slides (always available) | None |

### Text-to-Speech Providers

| Provider | Description | Requirements |
|----------|------------|--------------|
| `elevenlabs` | Premium AI voices with emotion | ElevenLabs API key |
| `openai` | Natural OpenAI TTS voices | OpenAI API key |
| `gtts` | Google Text-to-Speech (free) | None |

### LLM Providers

| Provider | Description | Requirements |
|----------|------------|--------------|
| `openai` | GPT models for content enhancement | OpenAI API key |
| `gemini` | Google Gemini models | Gemini API key |
| `claude` | Anthropic Claude models | Anthropic API key |
| `groq` | Fast inference with Groq | Groq API key |
| `ollama` | Local models via Ollama | Ollama installation |

## 📋 CLI Commands

### Core Commands

```bash
# Create video presentation
slide-stream create <input_file> <output_file>

# Generate example configuration
slide-stream init [config_file]

# List available providers and their status
slide-stream providers

# Show help
slide-stream --help
```

### Examples

```bash
# Basic usage
slide-stream create slides.md presentation.mp4

# With custom config
slide-stream create --config prod.yaml deck.pptx video.mp4

# Check what's available
slide-stream providers

# Create config file
slide-stream init my-config.yaml
```

## 🎨 Advanced Features

### PowerPoint Integration

- **Slide Content**: Extracts titles, bullet points, and images
- **Speaker Notes**: Uses notes for enhanced AI narration
- **Layouts**: Preserves slide structure and hierarchy

### AI Enhancement

- **Content Improvement**: LLMs enhance slide text for better flow
- **Image Generation**: DALL-E 3 creates relevant, professional images
- **Voice Selection**: Choose from multiple TTS voices and styles

### Professional Output

- **HD Video**: 1920x1080 resolution by default
- **Quality Audio**: Synchronized speech with proper timing
- **Custom Timing**: Configurable slide durations and padding

## 🔑 Getting API Keys

### OpenAI (for DALL-E 3 & GPT)
1. Visit [OpenAI Platform](https://platform.openai.com)
2. Sign up and create an API key
3. Add billing method (pay-per-use)

### ElevenLabs (for Premium TTS)
1. Visit [ElevenLabs](https://elevenlabs.io)
2. Create account and get API key
3. Choose from 900+ voices

### Pexels (for Stock Photos)
1. Visit [Pexels API](https://www.pexels.com/api/)
2. Sign up for free API access
3. Get your API key

### Unsplash (for Stock Photos)
1. Visit [Unsplash Developers](https://unsplash.com/developers)
2. Create application
3. Get your access key

## 📦 Installation Options

```bash
# Core package only
pip install slide-stream

# With specific AI providers
pip install slide-stream[openai]
pip install slide-stream[elevenlabs]
pip install slide-stream[gemini]
pip install slide-stream[claude]
pip install slide-stream[groq]

# All AI providers
pip install slide-stream[all-ai]

# Development dependencies
pip install slide-stream[dev]
```

## 📋 Requirements

- **Python**: 3.10 or higher
- **FFmpeg**: For video processing
- **Internet**: For AI services and stock photos (offline mode available)

### Installing FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 🔧 Configuration Reference

### Provider Options

**Image Providers:**
- `text`: Always available, no setup required
- `local`: Pick images from a local folder by filename keywords (`providers.images.folder`); pair with `slide-stream scan` to AI-name them
- `dalle3`: Requires `OPENAI_API_KEY`
- `gemini`: Google Imagen generation, cheap (~$0.02/image) — `pip install "slide-stream[gemini]"` + `GEMINI_API_KEY`; set `providers.images.model` for the Imagen tier (default Fast)
- `swarmui`: Self-hosted [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) server (`base_url`), free local generation via its native API — set `providers.images.model` (e.g. `juggernautXL_v9`), optional `steps`/`width`/`height`/`api_key`
- `pexels`: Requires `PEXELS_API_KEY`  
- `unsplash`: Requires `UNSPLASH_ACCESS_KEY`

### Enriching a deck with images (`enrich` / `scan`)

Beyond making videos, SlideStream can add an image to each slide and write a
**new editable deck** — no narration, no video, your original untouched:

```bash
# Markdown deck + images/ folder (default). Add --pptx for a PowerPoint too.
slide-stream enrich deck.md out/ --image-provider dalle3
slide-stream enrich deck.md out/ --image-provider local --image-folder ./pics --pptx
```

The output is a real artifact you can review, hand-edit, or narrate as a second
pass (`slide-stream create out/deck.md video.mp4`). For a one-pass video that
adds images *and* narrates internally, just run `create` with an image
provider configured — `enrich` is the deck-only track.

`scan` AI-renames a folder of images to keyword slugs so the `local` provider
can match them to slides (dry-run by default):

```bash
slide-stream scan ./pics --provider claude          # preview renames
slide-stream scan ./pics --provider claude --apply  # actually rename + write report
```

**TTS Providers:**
- `gtts`: Free, always available (needs internet)
- `kokoro`: Fully offline, no API key — `pip install "slide-stream[local-tts]"` (~340MB one-time model download; voices include `af_sarah`, `af_bella`, `am_adam`, `am_michael`)
- `chatterbox`: Voice cloning via a self-hosted [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server) (`base_url`); see "Privacy-first voice cloning" below
- `elevenlabs`: Requires `ELEVENLABS_API_KEY`
- `openai`: Requires `OPENAI_API_KEY`
- `openai-compatible`: Any OpenAI-compatible speech endpoint via `base_url` (local or hosted)

### Privacy-first voice cloning

The `chatterbox` provider narrates videos in your own voice without storing it
on the server between runs:

```yaml
providers:
  tts:
    provider: chatterbox
    base_url: https://chatterbox.example.org
    voice_sample: ./my_voice.wav       # 10-30s of clean speech (<5s fails)
    api_key: "${CHATTERBOX_TOKEN}"     # if your proxy checks a Bearer token
settings:
  strict: true                         # never fall back to the wrong voice
```

- `voice_sample` is uploaded **once per run under a random UUID filename** and
  referenced only for that render — no recognisable voice name ever exists on
  the server, and other users can neither find nor select it.
- Schedule `contrib/chatterbox/cleanup_uuid_voices.sh` on the server (cron) to
  delete UUID files after a grace period. Zero-shot cloning has no training
  step, so re-uploading each run costs only ~1-2 seconds.
- Prefer a stock server voice instead? Set `voice: Emily.wav` (or any name
  from `slide-stream voices`, which lists server voices and hides ephemeral
  UUID uploads).

### Narration

By default (when an LLM provider is configured) SlideStream writes narration
that *complements* the slides instead of reading them aloud, choosing a source
per slide:

1. **Speaker notes** (`.pptx`) — cleaned into speakable prose and fitted to the
   target length (long notes are summarised, thin ones expanded).
2. **Slide content** — turned into what a presenter would *say*, not a recital
   of the bullets.
3. **Slide image** — image-only slides are described by a vision-capable
   provider (`claude`, `openai`, `gemini`) and tied to the slide title.
4. **Title only** — a brief spoken introduction.

Control it per run:

```bash
# Aim for ~30 seconds of narration per slide
slide-stream create deck.pptx out.mp4 --llm-provider claude --narration-seconds 30

# Speak the PowerPoint speaker notes exactly as written (no LLM rewriting)
slide-stream create deck.pptx out.mp4 --verbatim-notes

# Provide your own script: one block per slide, separated by lines of ---
slide-stream create deck.md out.mp4 --script narration.txt
```

`--llm-model` selects a specific model (e.g. `claude-haiku-4-5`, the default for
Claude). API keys are read from the environment (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`). A `--script` file looks
like:

```
Welcome to the course. Today we cover neural networks.
---
A perceptron is the simplest building block of a network.
---
Thanks for watching.
```

**Avatar Providers (talking-head overlay):**
- `none`: Disabled (default)
- `precomputed`: Composites `head_1.mp4`, `head_2.mp4`, … from `providers.avatar.assets_dir` as a circle in a corner of each slide — no GPU or service needed.
- `d-id`: Lip-synced talking head generated from a source image via the [D-ID](https://www.d-id.com/) API (BYOK) — set `providers.avatar.source_image` (lecturer photo) and `api_key`/`DID_API_KEY`. Bills per minute of video (~$1–2/min), so pricier than voice/images.

Enable per run with `--avatar`, disable with `--no-avatar`; appearance via `settings.avatar` (`position`, `size`, `margin`).

**LLM Providers:**
- `none`: No content enhancement
- `openai`: Requires `OPENAI_API_KEY`
- `gemini`: Requires `GEMINI_API_KEY`
- `claude`: Requires `ANTHROPIC_API_KEY`
- `groq`: Requires `GROQ_API_KEY`
- `ollama`: Requires local Ollama installation

### Voice Options

**ElevenLabs Voices:**
- `rachel`: Professional female voice
- `adam`: Clear male voice
- `aria`: Expressive female voice
- (See [ElevenLabs docs](https://elevenlabs.io/docs) for full list)

**OpenAI Voices:**
- `alloy`: Balanced and natural
- `echo`: Clear and articulate  
- `fable`: Warm and engaging
- `nova`: Bright and energetic
- `onyx`: Deep and authoritative
- `shimmer`: Gentle and soothing

## 🤝 Contributing

We welcome contributions! See our documentation:

- **[User Guide](docs/USER_GUIDE.md)** - Comprehensive usage examples
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** - Setup and testing
- **[Type Safety](docs/TYPE_SAFETY.md)** - Code quality standards

## 🆕 Version History

- **2.0.0**: Configuration system, provider architecture, AI image generation
- **1.1.x**: PowerPoint support, bug fixes, stability improvements  
- **1.0.0**: Initial release with Markdown support

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with these excellent tools:
- [Typer](https://typer.tiangolo.com/) - Modern CLI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
- [MoviePy](https://moviepy.readthedocs.io/) - Video processing
- [OpenAI](https://openai.com/) - AI image generation and LLM
- [ElevenLabs](https://elevenlabs.io/) - Premium text-to-speech
- [PyYAML](https://pyyaml.org/) - Configuration parsing

---

**Ready to create professional presentations?** Get started with `pip install slide-stream[all-ai]` 🚀