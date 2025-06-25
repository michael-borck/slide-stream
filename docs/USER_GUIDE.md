# SlideStream 2.0 User Guide

Comprehensive guide for creating professional video presentations with SlideStream's AI-powered tools.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration System](#configuration-system)
- [Creating Your First Video](#creating-your-first-video)
- [Working with Providers](#working-with-providers)
- [Advanced Workflows](#advanced-workflows)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### 1. Install SlideStream

```bash
# Install with all AI providers
pip install slide-stream[all-ai]
```

### 2. Create Configuration

```bash
slide-stream init
```

This creates a `slidestream.yaml` file in your current directory with example configuration.

### 3. Set Up API Keys

Edit your configuration file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

### 4. Create Your First Video

```bash
slide-stream create presentation.md output.mp4
```

## Installation

### Core Installation

```bash
pip install slide-stream
```

### With Specific Providers

```bash
# For OpenAI (DALL-E, GPT, TTS)
pip install slide-stream[openai]

# For ElevenLabs premium voices
pip install slide-stream[elevenlabs]

# For Google Gemini
pip install slide-stream[gemini]

# For Anthropic Claude
pip install slide-stream[claude]

# For Groq (fast inference)
pip install slide-stream[groq]

# All AI providers
pip install slide-stream[all-ai]
```

### System Requirements

- **Python 3.10+**
- **FFmpeg** (for video processing)

#### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Configuration System

SlideStream 2.0 uses YAML configuration files for maximum flexibility and maintainability.

### Configuration Discovery

SlideStream searches for configuration in this order:

1. `./slidestream.yaml` (current directory)
2. `~/.slidestream.yaml` (home directory)
3. Built-in defaults

### Basic Configuration

```yaml
# slidestream.yaml
providers:
  llm:
    provider: openai
    model: gpt-4o-mini
    
  images:
    provider: dalle3
    fallback: text
    
  tts:
    provider: elevenlabs
    voice: rachel

api_keys:
  openai: "${OPENAI_API_KEY}"
  elevenlabs: "${ELEVENLABS_API_KEY}"

settings:
  video:
    resolution: [1920, 1080]
    fps: 24
  cleanup: true
```

### Environment Variables

Use environment variables for secure API key management:

```bash
# OpenAI (for DALL-E 3, GPT, and OpenAI TTS)
export OPENAI_API_KEY="sk-..."

# ElevenLabs (for premium TTS)
export ELEVENLABS_API_KEY="..."

# Stock photo providers (optional)
export PEXELS_API_KEY="..."
export UNSPLASH_ACCESS_KEY="..."

# Other LLM providers
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GROQ_API_KEY="..."
```

### Configuration Profiles

Create different configurations for different use cases:

**Basic Profile** (`basic.yaml`):
```yaml
providers:
  llm:
    provider: none
  images:
    provider: text
  tts:
    provider: gtts
```

**Professional Profile** (`pro.yaml`):
```yaml
providers:
  llm:
    provider: openai
    model: gpt-4o
  images:
    provider: dalle3
    fallback: pexels
  tts:
    provider: elevenlabs
    voice: rachel

api_keys:
  openai: "${OPENAI_API_KEY}"
  elevenlabs: "${ELEVENLABS_API_KEY}"
  pexels: "${PEXELS_API_KEY}"
```

Use with:
```bash
slide-stream create --config pro.yaml presentation.md video.mp4
```

## Creating Your First Video

### From Markdown

Create a simple Markdown file:

```markdown
# Welcome to SlideStream

- Create professional video presentations
- Use AI to enhance your content
- Generate videos automatically

# Key Features

- AI-powered image generation
- Premium text-to-speech voices
- Smart content enhancement
- Professional video output

# Getting Started

- Install SlideStream
- Configure your providers
- Create your first video
```

Generate the video:

```bash
slide-stream create presentation.md my-video.mp4
```

### From PowerPoint

SlideStream 2.0 supports PowerPoint files with enhanced features:

```bash
slide-stream create slides.pptx presentation.mp4
```

**PowerPoint Features:**
- Extracts slide titles and content
- Uses speaker notes for enhanced narration
- Preserves slide structure
- Supports complex layouts

### Example Speaker Notes

When using PowerPoint files, add speaker notes for better AI narration:

```
Slide Content: Key Benefits
• Faster development
• Better user experience
• Competitive advantage

Speaker Notes: In this slide, we'll explore the three main benefits of adopting our solution. First, you'll see dramatically faster development cycles, allowing your team to ship features in weeks rather than months. Second, your users will experience a more intuitive and responsive interface. Finally, these improvements will give you a significant competitive advantage in your market.
```

The AI will use these notes to create natural, flowing narration.

## Working with Providers

### Image Providers

#### DALL-E 3 (AI Generation)
```yaml
providers:
  images:
    provider: dalle3
    fallback: text

api_keys:
  openai: "${OPENAI_API_KEY}"
```

**Benefits:**
- Custom images for each slide
- Relevant to your content
- Professional quality
- No licensing concerns

**Requirements:**
- OpenAI API key
- Pay-per-image pricing

#### Stock Photo Providers

**Pexels:**
```yaml
providers:
  images:
    provider: pexels
    fallback: text

api_keys:
  pexels: "${PEXELS_API_KEY}"
```

**Unsplash:**
```yaml
providers:
  images:
    provider: unsplash
    fallback: text

api_keys:
  unsplash: "${UNSPLASH_ACCESS_KEY}"
```

#### Text-Based Images
```yaml
providers:
  images:
    provider: text
```

Always available as a fallback. Creates clean, professional text-based slides.

### Text-to-Speech Providers

#### ElevenLabs (Premium)
```yaml
providers:
  tts:
    provider: elevenlabs
    voice: rachel  # or adam, aria, etc.

api_keys:
  elevenlabs: "${ELEVENLABS_API_KEY}"
```

**Available Voices:**
- `rachel`: Professional female voice
- `adam`: Clear male voice
- `aria`: Expressive female voice
- `josh`: Warm male voice
- And 900+ more voices

#### OpenAI TTS
```yaml
providers:
  tts:
    provider: openai
    voice: nova  # alloy, echo, fable, nova, onyx, shimmer

api_keys:
  openai: "${OPENAI_API_KEY}"
```

#### Google TTS (Free)
```yaml
providers:
  tts:
    provider: gtts
```

Always available, no API key required.

### LLM Providers

#### Content Enhancement

LLMs improve your slide content by:
- Making bullet points flow naturally
- Creating engaging narratives
- Improving clarity and structure
- Generating better image search queries

**OpenAI GPT:**
```yaml
providers:
  llm:
    provider: openai
    model: gpt-4o-mini  # or gpt-4o for higher quality
```

**Google Gemini:**
```yaml
providers:
  llm:
    provider: gemini
    model: gemini-1.5-flash
```

**Anthropic Claude:**
```yaml
providers:
  llm:
    provider: claude
    model: claude-3-5-sonnet-20241022
```

## Advanced Workflows

### Multi-Configuration Workflow

Create different configurations for different scenarios:

```bash
# Quick prototype with free services
slide-stream create --config basic.yaml draft.md prototype.mp4

# High-quality final version
slide-stream create --config premium.yaml final.md presentation.mp4

# Client-specific branding
slide-stream create --config client-brand.yaml proposal.md client-video.mp4
```

### Batch Processing

Process multiple presentations:

```bash
# Create multiple videos
for file in *.md; do
    output="${file%.md}.mp4"
    slide-stream create "$file" "$output"
done
```

### Custom Video Settings

Fine-tune video output:

```yaml
settings:
  video:
    resolution: [1920, 1080]  # 4K: [3840, 2160]
    fps: 30                   # Smooth playback
    codec: libx264            # Compatibility
    audio_codec: aac
    slide_duration_padding: 2.0  # More time per slide
    default_slide_duration: 8.0

  image:
    bg_color: "#1a1a1a"      # Dark theme
    font_color: "#ffffff"
    title_font_size: 120
    content_font_size: 80
```

### Integration with CI/CD

Automate presentation generation:

```yaml
# .github/workflows/presentations.yml
name: Generate Presentations
on:
  push:
    paths: ['presentations/*.md']

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install SlideStream
        run: pip install slide-stream[all-ai]
      
      - name: Generate Videos
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ELEVENLABS_API_KEY: ${{ secrets.ELEVENLABS_API_KEY }}
        run: |
          slide-stream create presentations/quarterly-review.md output/q4-review.mp4
          slide-stream create presentations/product-launch.md output/launch.mp4
```

## Troubleshooting

### Common Issues

#### "Provider not available"
```bash
slide-stream providers
```
Check which providers are available and their status.

#### "API key not found"
- Verify environment variables: `echo $OPENAI_API_KEY`
- Check configuration file syntax
- Ensure API keys are valid and have sufficient credits

#### "FFmpeg not found"
Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

#### Video generation fails
- Check disk space in temp directory
- Verify input file format (.md or .pptx)
- Try with `--config` to use specific configuration

#### Audio/video sync issues
Adjust timing settings:
```yaml
settings:
  video:
    slide_duration_padding: 2.0  # More padding
    default_slide_duration: 6.0  # Longer default
```

### Debug Mode

For detailed error information:

```bash
PYTHONPATH=. python -m slide_stream.cli create --help
```

### Provider Status

Check what's working:

```bash
slide-stream providers
```

Output shows each provider's availability and requirements.

## Best Practices

### Content Creation

**Markdown Structure:**
```markdown
# Clear, Descriptive Titles

- Use bullet points for key ideas
- Keep points concise and focused
- Aim for 3-5 points per slide

# Logical Flow

- Structure your presentation logically
- Use consistent formatting
- Include call-to-action slides
```

**PowerPoint Tips:**
- Use speaker notes for detailed explanations
- Keep slide content brief
- Use consistent layouts
- Include relevant images in slides

### Configuration Management

**Environment Variables:**
```bash
# .env file (don't commit to git)
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
PEXELS_API_KEY=...
```

**Version Control:**
- Commit configuration files
- Use environment variables for secrets
- Create different configs for different environments

### Performance Optimization

**Fast Generation:**
```yaml
providers:
  llm:
    provider: groq  # Fastest inference
    model: llama-3.1-8b-instant
  images:
    provider: text  # No API calls
  tts:
    provider: gtts  # Free and fast
```

**High Quality:**
```yaml
providers:
  llm:
    provider: openai
    model: gpt-4o  # Best quality
  images:
    provider: dalle3
  tts:
    provider: elevenlabs
    voice: rachel
```

### Cost Management

**Monitor Usage:**
- OpenAI: Check usage dashboard
- ElevenLabs: Monitor character usage
- Set usage alerts

**Optimize Costs:**
- Use text images for drafts
- Switch to premium providers for final versions
- Batch process multiple presentations

### Quality Guidelines

**For Professional Presentations:**
- Use DALL-E 3 or stock photos for images
- Use ElevenLabs or OpenAI TTS for voices
- Enable LLM content enhancement
- Use higher resolution (1920x1080 or 4K)

**For Internal/Draft Use:**
- Text-based images are sufficient
- gTTS provides adequate quality
- Disable LLM enhancement for speed

### Security Best Practices

- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate API keys regularly
- Monitor API usage for unusual activity
- Use least-privilege API permissions

---

This guide covers the essential workflows for SlideStream 2.0. For development setup and contributing, see [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md).