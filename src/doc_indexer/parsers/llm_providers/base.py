"""
Base class for LLM providers.
"""

from abc import ABC, abstractmethod

from PIL import Image


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze an image using LLM and return extracted content.

        Args:
            image: PIL Image to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Extracted content as string
        """
        ...

    @abstractmethod
    async def analyze_text(self, text: str, prompt: str) -> str:
        """
        Analyze text using LLM for enhanced extraction.

        Args:
            text: Text to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Enhanced/structured text
        """
        ...
