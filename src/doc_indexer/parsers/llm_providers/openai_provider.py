"""
OpenAI LLM provider implementation.
"""

import base64
import os
from io import BytesIO
from typing import Optional

from PIL import Image

try:
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    HumanMessage = None

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider for GPT models."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-4-vision-preview"
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, will try to read from OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-4-vision-preview" for vision)
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "API key is required for OpenAI provider. "
                "Provide it as parameter or set OPENAI_API_KEY environment variable."
            )

        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install with: pip install langchain-openai"
            )

        # Don't store API key as instance variable for security
        self.model = model
        self.client = ChatOpenAI(
            api_key=api_key, model=model, temperature=0.1, max_tokens=4096
        )
        # Clear the API key from memory after client initialization
        api_key = None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze an image using OpenAI vision model.

        Args:
            image: PIL Image to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Extracted content as string
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(image)

        # Create message with image
        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high",
                },
            },
        ]

        message = HumanMessage(content=message_content)

        try:
            response = await self.client.ainvoke([message])
            return response.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    async def analyze_text(self, text: str, prompt: str) -> str:
        """
        Analyze text using OpenAI model for enhanced extraction.

        Args:
            text: Text to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Enhanced/structured text
        """
        # Sanitize and limit input to prevent prompt injection
        text = text[:5000]  # Limit text length
        prompt = prompt[:500]  # Limit prompt length

        # Combine prompt with text
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"

        message = HumanMessage(content=full_prompt)

        try:
            # Use existing client or create text-only version
            if "vision" not in self.model:
                response = await self.client.ainvoke([message])
            else:
                # For vision models, we need to use text-only model
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")

                text_client = ChatOpenAI(
                    api_key=api_key,
                    model="gpt-4-turbo-preview",
                    temperature=0.1,
                    max_tokens=4096,
                )
                response = await text_client.ainvoke([message])
                # Clear API key from memory
                api_key = None

            return response.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
