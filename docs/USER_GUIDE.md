# SlideStream User Guide

A comprehensive guide to using SlideStream for creating AI-powered video presentations from Markdown and PowerPoint files.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Input Formats](#input-formats)
5. [Command Reference](#command-reference)
6. [AI Providers](#ai-providers)
7. [Image Sources](#image-sources)
8. [Workflow Examples](#workflow-examples)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)
11. [Tips & Best Practices](#tips--best-practices)

## Overview

SlideStream is an AI-powered command-line tool that automatically creates professional video presentations from:
- 📝 **Markdown files** (.md)
- 📊 **PowerPoint files** (.pptx)

### Key Features

- 🤖 **AI Enhancement**: Improve content using OpenAI, Gemini, Claude, Groq, or Ollama
- 📋 **Speaker Notes Support**: Use PowerPoint speaker notes for enhanced AI narration
- 🖼️ **Smart Images**: Automatic image sourcing from Unsplash or generate text-based slides
- 🎙️ **Text-to-Speech**: Natural narration using Google Text-to-Speech
- 🎨 **Professional Output**: High-quality MP4 videos with configurable settings

## Installation

### Basic Installation

```bash
pip install slide-stream
```

### With AI Provider Support

Choose the AI providers you want to use:

```bash
# For OpenAI GPT models
pip install slide-stream[openai]

# For Google Gemini
pip install slide-stream[gemini]

# For Anthropic Claude
pip install slide-stream[claude]

# For Groq (fast inference)
pip install slide-stream[groq]

# For all AI providers
pip install slide-stream[all-ai]
```

### System Requirements

- **Python**: 3.10 or higher
- **FFmpeg**: Required for video processing
- **Internet connection**: For Unsplash images and AI providers (optional)

#### Installing FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) or use chocolatey:
```bash
choco install ffmpeg
```

## Quick Start

### 1. Create a Simple Markdown File

Create `my-presentation.md`:
```markdown
# Welcome to My Presentation

- This is the first point
- Here's another important point
- And a final thought

# Second Slide

- More content here
- Additional information
- Conclusion
```

### 2. Generate Your Video

```bash
slide-stream --input my-presentation.md --output my-video.mp4
```

### 3. View Your Video

Open `my-video.mp4` in any video player to see your AI-generated presentation!

## Input Formats

### Markdown Files (.md)

SlideStream uses level 1 headers (`#`) to create new slides:

```markdown
# Slide 1 Title
- Bullet point 1
- Bullet point 2
- Bullet point 3

# Slide 2 Title
- More content
- Additional points

# Final Slide
- Conclusion
- Thank you
```

**Features:**
- Each `# Header` creates a new slide
- Bullet points become slide content
- Simple and clean formatting
- Great for quick presentations

### PowerPoint Files (.pptx)

SlideStream can import PowerPoint presentations:

**Supported Elements:**
- Slide titles
- Text content (bullet points, paragraphs)
- Speaker notes (used for enhanced AI narration)

**PowerPoint Features:**
- **Speaker Notes**: Automatically used by AI to create natural narration
- **Slide Content**: Bullet points and text are extracted
- **Layout Agnostic**: Works with any PowerPoint layout

**Example PowerPoint to Video:**
```bash
slide-stream --input presentation.pptx --output video.mp4
```

## Command Reference

### Basic Syntax

```bash
slide-stream [OPTIONS]
```

### Required Options

| Option | Short | Description |
|--------|--------|-------------|
| `--input` | `-i` | Path to input file (.md or .pptx) |
| `--output` | `-o` | Output video filename (default: output_video.mp4) |

### AI & Content Options

| Option | Default | Description |
|--------|---------|-------------|
| `--llm-provider` | `none` | AI provider: `none`, `openai`, `gemini`, `claude`, `groq`, `ollama` |
| `--llm-model` | (auto) | Specific model to use (optional) |
| `--image-source` | `unsplash` | Image source: `unsplash` or `text` |

### Examples

```bash
# Basic usage
slide-stream -i slides.md -o presentation.mp4

# With AI enhancement
slide-stream -i slides.md -o presentation.mp4 --llm-provider openai

# Text-only slides (no internet required)
slide-stream -i slides.md -o presentation.mp4 --image-source text

# PowerPoint with specific model
slide-stream -i presentation.pptx -o video.mp4 --llm-provider openai --llm-model gpt-4o
```

## AI Providers

### Configuration

Set environment variables for your chosen AI provider:

#### OpenAI
```bash
export OPENAI_API_KEY="your-openai-key"
export OPENAI_MODEL="gpt-4o-mini"  # Optional: default model
```

#### Google Gemini
```bash
export GEMINI_API_KEY="your-gemini-key"
export GEMINI_MODEL="gemini-1.5-flash"  # Optional: default model
```

#### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your-claude-key"
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"  # Optional: default model
```

#### Groq
```bash
export GROQ_API_KEY="your-groq-key"
export GROQ_MODEL="llama-3.1-8b-instant"  # Optional: default model
```

#### Ollama (Local)
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2"  # Optional: default model
```

### AI Enhancement Features

When using AI providers, SlideStream:

1. **Improves Speech**: Converts bullet points into natural, flowing narration
2. **Enhances Search**: Generates better image search queries for Unsplash
3. **Uses Speaker Notes**: For PowerPoint files, incorporates speaker notes into narration
4. **Maintains Context**: Ensures coherent presentation flow

## Image Sources

### Unsplash (Default)
- **Pros**: High-quality, professional stock photos
- **Cons**: Requires internet connection
- **Usage**: Automatic image search based on slide content

```bash
slide-stream -i slides.md -o video.mp4 --image-source unsplash
```

### Text-Only
- **Pros**: No internet required, fast processing
- **Cons**: Less visually appealing
- **Usage**: Creates text-based slides with title and bullet points

```bash
slide-stream -i slides.md -o video.mp4 --image-source text
```

## Workflow Examples

### 1. Simple Markdown Presentation

**Input**: `simple.md`
```markdown
# Introduction
- Welcome to our company
- Today's agenda
- Questions at the end

# Our Services
- Web development
- Mobile applications
- Cloud solutions

# Contact Us
- email@company.com
- +1-555-0123
- Thank you for your time
```

**Command**:
```bash
slide-stream -i simple.md -o company-intro.mp4 --image-source text
```

### 2. AI-Enhanced Presentation

**Input**: `pitch.md`
```markdown
# Product Launch
- Revolutionary new app
- Solves major industry problem
- Ready for market

# Market Opportunity
- $10B market size
- Growing 15% annually
- Underserved segment

# Our Solution
- Innovative technology
- User-friendly design
- Competitive pricing
```

**Setup**:
```bash
export OPENAI_API_KEY="your-key-here"
```

**Command**:
```bash
slide-stream -i pitch.md -o pitch-presentation.mp4 \
  --llm-provider openai \
  --image-source unsplash
```

### 3. PowerPoint with Speaker Notes

**Input**: `presentation.pptx` (with speaker notes)

**Command**:
```bash
slide-stream -i presentation.pptx -o enhanced-video.mp4 \
  --llm-provider openai \
  --image-source unsplash
```

**What happens**:
- Extracts slide content and speaker notes
- AI uses speaker notes to create natural narration
- Finds relevant images for each slide
- Generates professional video output

### 4. Batch Processing

Process multiple files:

```bash
# Process all markdown files in directory
for file in *.md; do
  slide-stream -i "$file" -o "${file%.md}.mp4" --image-source text
done

# Process with AI enhancement
for file in *.pptx; do
  slide-stream -i "$file" -o "${file%.pptx}-video.mp4" \
    --llm-provider openai --image-source unsplash
done
```

## Advanced Usage

### Custom Model Selection

Different models offer different capabilities:

```bash
# Fast, cost-effective
slide-stream -i slides.md -o video.mp4 \
  --llm-provider openai --llm-model gpt-4o-mini

# High-quality, more expensive
slide-stream -i slides.md -o video.mp4 \
  --llm-provider openai --llm-model gpt-4o

# Ultra-fast inference with Groq
slide-stream -i slides.md -o video.mp4 \
  --llm-provider groq --llm-model llama-3.1-8b-instant
```

### Local AI with Ollama

Run AI models locally for privacy:

1. **Install Ollama**: [Follow Ollama installation guide](https://ollama.ai)

2. **Pull a model**:
```bash
ollama pull llama3.2
```

3. **Use with SlideStream**:
```bash
slide-stream -i slides.md -o video.mp4 \
  --llm-provider ollama --llm-model llama3.2
```

### Environment Configuration File

Create `.env` file in your project:
```bash
# .env file
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
GEMINI_API_KEY=your-gemini-key
ANTHROPIC_API_KEY=your-claude-key
```

Load before running:
```bash
source .env
slide-stream -i slides.md -o video.mp4 --llm-provider openai
```

## Troubleshooting

### Common Issues

#### 1. "Input file not found"
```bash
# Error: Input file not found: slides.md
```
**Solution**: Check file path and ensure file exists
```bash
ls -la slides.md  # Verify file exists
slide-stream -i ./slides.md -o video.mp4  # Use explicit path
```

#### 2. "Unsupported file type"
```bash
# Error: Unsupported file type: .txt. Supported: .md, .pptx
```
**Solution**: Use supported file formats (.md or .pptx)

#### 3. "Error initializing LLM"
```bash
# Error initializing LLM: Missing API key
```
**Solution**: Set environment variable for your AI provider
```bash
export OPENAI_API_KEY="your-key-here"
```

#### 4. "No slides found"
```bash
# No slides found in the .md file. Exiting.
```
**Solution**: Ensure your Markdown uses `# Header` for slide titles
```markdown
# Slide Title  ← This creates a slide
- Content here

## Subtitle   ← This does NOT create a slide
```

#### 5. FFmpeg Issues
```bash
# Error: FFmpeg not found
```
**Solution**: Install FFmpeg (see Installation section)

#### 6. Network/Image Issues
```bash
# Error downloading image from Unsplash
```
**Solution**: Use text-only images or check internet connection
```bash
slide-stream -i slides.md -o video.mp4 --image-source text
```

### Performance Tips

#### 1. Use Text Images for Testing
```bash
# Fast processing, no network calls
slide-stream -i slides.md -o video.mp4 --image-source text
```

#### 2. Choose Appropriate AI Models
```bash
# Fast and cost-effective
--llm-provider openai --llm-model gpt-4o-mini

# Local processing (no API costs)
--llm-provider ollama --llm-model llama3.2
```

#### 3. Optimize PowerPoint Files
- Keep slide content concise
- Use clear, descriptive speaker notes
- Avoid complex layouts

### Debug Mode

For detailed output, check the console during processing:
- Green messages: Success
- Yellow messages: Warnings
- Red messages: Errors

## Tips & Best Practices

### Content Creation

#### Markdown Best Practices
```markdown
# Clear, Descriptive Titles
- Keep bullet points concise
- Use parallel structure
- Aim for 3-5 points per slide

# Avoid
- Really long bullet points that go on and on and become difficult to read
- Too many nested levels
- Complex formatting
```

#### PowerPoint Best Practices
- **Speaker Notes**: Add detailed speaker notes for better AI narration
- **Simple Layouts**: Use standard layouts (Title + Content works best)
- **Clear Text**: Ensure text is extractable (avoid text in images)

### AI Enhancement Tips

#### 1. Content Quality
- Clear, descriptive slide titles help AI find better images
- Detailed speaker notes result in better narration
- Consistent terminology improves coherence

#### 2. Model Selection
- **gpt-4o-mini**: Fast, cost-effective, good quality
- **gpt-4o**: Highest quality, more expensive
- **gemini-1.5-flash**: Good balance, competitive pricing
- **claude-3-5-sonnet**: Excellent for creative content
- **groq models**: Ultra-fast inference

#### 3. Image Search Optimization
```markdown
# Good: Specific, descriptive
# "Modern Office Workspace"
# "Data Analytics Dashboard"
# "Team Collaboration Meeting"

# Poor: Generic, vague
# "Business"
# "Technology"
# "People"
```

### Workflow Optimization

#### 1. Development Workflow
```bash
# 1. Create content with text images (fast)
slide-stream -i draft.md -o preview.mp4 --image-source text

# 2. Review and refine content

# 3. Generate final version with AI and images
slide-stream -i final.md -o presentation.mp4 \
  --llm-provider openai --image-source unsplash
```

#### 2. Team Workflow
```bash
# Content creators use PowerPoint with speaker notes
# Developers convert to video
slide-stream -i team-presentation.pptx -o final-video.mp4 \
  --llm-provider openai --image-source unsplash
```

#### 3. Production Workflow
```bash
# Automated processing
#!/bin/bash
for presentation in presentations/*.pptx; do
  output="videos/$(basename "$presentation" .pptx).mp4"
  slide-stream -i "$presentation" -o "$output" \
    --llm-provider openai --image-source unsplash
done
```

### Quality Considerations

#### Video Output
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 24 fps
- **Codec**: H.264 (MP4)
- **Audio**: AAC compression

#### Duration
- Each slide duration is based on text-to-speech narration length
- Longer, more detailed content = longer slides
- Typical: 30-60 seconds per slide

#### File Sizes
- Text images: ~5-10 MB per minute
- Unsplash images: ~20-50 MB per minute
- Depends on slide count and image complexity

---

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/michael-borck/slide-stream/issues)
- **Documentation**: [Project README](../README.md)
- **Type Safety**: [Type Safety Documentation](TYPE_SAFETY.md)

Happy presenting! 🎬✨