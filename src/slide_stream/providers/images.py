"""Image provider implementations."""

import os
import textwrap
from typing import Any, Dict, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from .base import ImageProvider

console = Console()
err_console = Console(stderr=True, style="bold red")


class TextImageProvider(ImageProvider):
    """Generate text-based images for slides."""
    
    @property
    def name(self) -> str:
        return "text"
    
    def is_available(self) -> bool:
        return True
    
    def generate_image(self, query: str, filename: str) -> str:
        """Create a text-based image."""
        settings = self.config.get("settings", {}).get("image", {})
        video_settings = self.config.get("settings", {}).get("video", {})
        
        resolution = tuple(video_settings.get("resolution", [1920, 1080]))
        bg_color = settings.get("bg_color", "black")
        font_color = settings.get("font_color", "white")
        title_font_size = settings.get("title_font_size", 100)
        content_font_size = settings.get("content_font_size", 60)
        max_line_width = settings.get("max_line_width", 50)
        
        img = Image.new("RGB", resolution, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
            content_font = ImageFont.truetype("arial.ttf", content_font_size)
        except OSError:
            try:
                title_font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
                content_font = ImageFont.truetype("DejaVuSans.ttf", content_font_size)
            except OSError:
                title_font = ImageFont.load_default()
                content_font = ImageFont.load_default()
        
        # Draw title
        title = "Generated Image"
        draw.text(
            (resolution[0] * 0.1, resolution[1] * 0.1),
            title,
            font=title_font,
            fill=font_color,
        )
        
        # Draw content
        content_items = [f"Topic: {query}"]
        y_pos = resolution[1] * 0.3
        
        for item in content_items:
            wrapped_lines = textwrap.wrap(f"• {item}", width=max_line_width)
            for line in wrapped_lines:
                draw.text(
                    (resolution[0] * 0.1, y_pos),
                    line,
                    font=content_font,
                    fill=font_color,
                )
                y_pos += 70
            y_pos += 30
        
        img.save(filename)
        console.print(f"  - Generated text image: {query}")
        return filename


class DalleImageProvider(ImageProvider):
    """Generate images using DALL-E 3 via OpenAI API."""
    
    @property
    def name(self) -> str:
        return "dalle3"
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        api_keys = self.config.get("api_keys", {})
        openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        return bool(openai_key)
    
    def generate_image(self, query: str, filename: str) -> str:
        """Generate image using DALL-E 3."""
        try:
            from openai import OpenAI
            
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = OpenAI(api_key=api_key)
            
            # Create a descriptive prompt for presentation images
            prompt = f"A professional, clean image for a presentation slide about: {query}. High quality, suitable for business presentation, no text overlay."
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",  # Close to 1920x1080 aspect ratio
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download the generated image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            
            console.print(f"  - Generated DALL-E image: {query}")
            return filename
            
        except ImportError:
            err_console.print("  - OpenAI library not installed. Install with: pip install openai")
            return self._fallback_to_text(query, filename)
        except Exception as e:
            err_console.print(f"  - DALL-E error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename)
    
    def _fallback_to_text(self, query: str, filename: str) -> str:
        """Fallback to text image generation."""
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename)


class PexelsImageProvider(ImageProvider):
    """Download images from Pexels stock photo service."""
    
    @property
    def name(self) -> str:
        return "pexels"
    
    def is_available(self) -> bool:
        """Check if Pexels API key is available."""
        api_keys = self.config.get("api_keys", {})
        pexels_key = api_keys.get("pexels") or os.getenv("PEXELS_API_KEY")
        return bool(pexels_key)
    
    def generate_image(self, query: str, filename: str) -> str:
        """Download image from Pexels."""
        try:
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("pexels") or os.getenv("PEXELS_API_KEY")
            
            if not api_key:
                raise ValueError("Pexels API key not found")
            
            settings = self.config.get("settings", {}).get("image", {})
            timeout = settings.get("download_timeout", 15)
            
            headers = {"Authorization": api_key}
            
            # Search for images
            search_url = "https://api.pexels.com/v1/search"
            params = {
                "query": query,
                "per_page": 1,
                "orientation": "landscape"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data.get("photos"):
                raise ValueError("No images found")
            
            # Get the first image
            photo = data["photos"][0]
            image_url = photo["src"]["large2x"]  # High resolution
            
            # Download the image
            img_response = requests.get(image_url, timeout=timeout)
            img_response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            
            console.print(f"  - Downloaded Pexels image: {query}")
            return filename
            
        except Exception as e:
            err_console.print(f"  - Pexels error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename)
    
    def _fallback_to_text(self, query: str, filename: str) -> str:
        """Fallback to text image generation."""
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename)


class UnsplashImageProvider(ImageProvider):
    """Download images from Unsplash stock photo service."""
    
    @property
    def name(self) -> str:
        return "unsplash"
    
    def is_available(self) -> bool:
        """Check if Unsplash API key is available."""
        api_keys = self.config.get("api_keys", {})
        unsplash_key = api_keys.get("unsplash") or os.getenv("UNSPLASH_ACCESS_KEY")
        return bool(unsplash_key)
    
    def generate_image(self, query: str, filename: str) -> str:
        """Download image from Unsplash."""
        try:
            api_keys = self.config.get("api_keys", {})
            api_key = api_keys.get("unsplash") or os.getenv("UNSPLASH_ACCESS_KEY")
            
            if not api_key:
                raise ValueError("Unsplash API key not found")
            
            settings = self.config.get("settings", {}).get("image", {})
            timeout = settings.get("download_timeout", 15)
            video_settings = self.config.get("settings", {}).get("video", {})
            resolution = video_settings.get("resolution", [1920, 1080])
            
            headers = {"Authorization": f"Client-ID {api_key}"}
            
            # Search for images
            search_url = "https://api.unsplash.com/search/photos"
            params = {
                "query": query,
                "per_page": 1,
                "orientation": "landscape"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            if not data.get("results"):
                raise ValueError("No images found")
            
            # Get the first image
            photo = data["results"][0]
            image_url = f"{photo['urls']['raw']}&w={resolution[0]}&h={resolution[1]}&fit=crop"
            
            # Download the image
            img_response = requests.get(image_url, timeout=timeout)
            img_response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(img_response.content)
            
            console.print(f"  - Downloaded Unsplash image: {query}")
            return filename
            
        except Exception as e:
            err_console.print(f"  - Unsplash error: {e}. Using text fallback.")
            return self._fallback_to_text(query, filename)
    
    def _fallback_to_text(self, query: str, filename: str) -> str:
        """Fallback to text image generation."""
        text_provider = TextImageProvider(self.config)
        return text_provider.generate_image(query, filename)