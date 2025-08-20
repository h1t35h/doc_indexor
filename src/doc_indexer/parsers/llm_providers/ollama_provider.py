"""
Ollama LLM provider implementation.
"""

import base64
import ssl
from io import BytesIO

import aiohttp
from PIL import Image

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local model inference."""

    def __init__(
        self, model: str = "llava", base_url: str = "http://localhost:11434"
    ) -> None:
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "llava" for vision, "llama2" for text)
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def _create_secure_session(
        self, timeout_seconds: int = 30
    ) -> aiohttp.ClientSession:
        """Create a secure aiohttp session with proper configuration.

        Args:
            timeout_seconds: Request timeout in seconds

        Returns:
            Configured ClientSession
        """
        # Create SSL context for HTTPS connections
        ssl_context = None
        if self.base_url.startswith("https://"):
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

        # Configure connection limits and timeouts
        connector = aiohttp.TCPConnector(
            limit=10,  # Total connection pool limit
            limit_per_host=5,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache timeout
            ssl=ssl_context,
        )

        timeout = aiohttp.ClientTimeout(
            total=timeout_seconds, connect=10, sock_read=timeout_seconds
        )

        # Create session with security headers
        headers = {"User-Agent": "DocIndexer/1.0", "Accept": "application/json"}

        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            raise_for_status=False,  # Handle status codes manually
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """
        Analyze an image using Ollama vision model.

        Args:
            image: PIL Image to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Extracted content as string
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(image)

        # Prepare the request
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image_base64]}],
            "stream": False,
        }

        try:
            async with self._create_secure_session(timeout_seconds=60) as session:
                async with session.post(self.chat_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Ollama API error: {response.status} - {error_text}"
                        )
        except aiohttp.ClientError as e:
            raise Exception(f"Failed to connect to Ollama: {e}")

    async def analyze_text(self, text: str, prompt: str) -> str:
        """
        Analyze text using Ollama model for enhanced extraction.

        Args:
            text: Text to analyze
            prompt: Prompt to guide the analysis

        Returns:
            Enhanced/structured text
        """
        # Combine prompt with text
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"

        payload = {
            "model": self.model.replace(
                "llava", "llama2"
            ),  # Use text model for text analysis
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
        }

        try:
            async with self._create_secure_session(timeout_seconds=30) as session:
                async with session.post(self.chat_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Ollama API error: {response.status} - {error_text}"
                        )
        except aiohttp.ClientError as e:
            raise Exception(f"Failed to connect to Ollama: {e}")

    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            async with self._create_secure_session(timeout_seconds=5) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception:
            return False
