"""
Configuration management for document parsers.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ParserConfig:
    """Configuration for document parsers."""

    # LLM Provider Configuration
    llm_provider: str = "ollama"  # "ollama" or "openai"

    # Ollama Configuration
    ollama_model: str = (
        "llava"  # Deprecated - use ollama_image_model and ollama_text_model
    )
    ollama_image_model: Optional[str] = (
        None  # Model for image/vision tasks (e.g., llava, bakllava)
    )
    ollama_text_model: Optional[str] = (
        None  # Model for text processing (e.g., llama2, mistral)
    )
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-vision-preview"

    # Parsing Configuration
    parsing_mode: str = "hybrid"  # "text_only", "hybrid", or "llm_only"
    enable_ocr_fallback: bool = True
    max_pages_per_batch: int = 10
    llm_timeout_seconds: int = 30

    # Image Processing
    image_quality: int = 85  # JPEG quality for image compression
    max_image_size: tuple = (1920, 1080)  # Max dimensions for images sent to LLM

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)
        self.ollama_image_model = os.getenv(
            "OLLAMA_IMAGE_MODEL", self.ollama_image_model
        )
        self.ollama_text_model = os.getenv("OLLAMA_TEXT_MODEL", self.ollama_text_model)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.parsing_mode = os.getenv("PARSING_MODE", self.parsing_mode)
        self.enable_ocr_fallback = os.getenv("ENABLE_OCR", "true").lower() == "true"
        self.max_pages_per_batch = int(
            os.getenv("MAX_PAGES_PER_BATCH", str(self.max_pages_per_batch))
        )
        self.llm_timeout_seconds = int(
            os.getenv("LLM_TIMEOUT_SECONDS", str(self.llm_timeout_seconds))
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.llm_provider not in ["ollama", "openai"]:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

        if self.parsing_mode not in ["text_only", "hybrid", "llm_only"]:
            raise ValueError(f"Invalid parsing mode: {self.parsing_mode}")

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")

        if self.max_pages_per_batch < 1:
            raise ValueError("max_pages_per_batch must be at least 1")

        if self.llm_timeout_seconds < 1:
            raise ValueError("llm_timeout_seconds must be at least 1")
