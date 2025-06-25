"""Abstract base classes for providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ImageProvider(ABC):
    """Abstract base class for image providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
    
    @abstractmethod
    def generate_image(self, query: str, filename: str) -> str:
        """Generate or download an image based on query.
        
        Args:
            query: Search query or prompt for image generation
            filename: Target filename to save the image
            
        Returns:
            Path to the saved image file
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
    
    @abstractmethod
    def synthesize(self, text: str, filename: str) -> Optional[str]:
        """Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            filename: Target filename to save the audio
            
        Returns:
            Path to the saved audio file, or None if failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
    
    @abstractmethod
    def generate_text(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Generate text based on prompt.
        
        Args:
            prompt: Input prompt for text generation
            model: Optional specific model to use
            
        Returns:
            Generated text or None if failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available."""
        pass