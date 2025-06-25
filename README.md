# Slide Stream 2.0

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

SlideStream uses YAML configuration files for maximum flexibility:

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
- `dalle3`: Requires `OPENAI_API_KEY`
- `pexels`: Requires `PEXELS_API_KEY`  
- `unsplash`: Requires `UNSPLASH_ACCESS_KEY`

**TTS Providers:**
- `gtts`: Free, always available
- `elevenlabs`: Requires `ELEVENLABS_API_KEY`
- `openai`: Requires `OPENAI_API_KEY`

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