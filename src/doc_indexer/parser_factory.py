"""
Factory for creating document parsers with LLM support.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Document, DocumentMetadata
from .parsers.base import BaseParser
from .parsers.config import ParserConfig
from .parsers.llm_providers.ollama_provider import OllamaProvider
from .parsers.llm_providers.openai_provider import OpenAIProvider
from .parsers.pdf_parser import PDFParser
from .parsers.powerpoint_parser import PowerPointParser
from .parsers.strategies.llm_enhanced import LLMEnhancedStrategy
from .parsers.strategies.text_only import TextOnlyStrategy
from .parsers.word_parser import WordParser


class DocumentParser:
    """Main document parser that delegates to specific parsers with LLM support."""

    def __init__(self, parser_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize parser with configuration."""
        config_dict = parser_config or {}

        self.llm_provider_name = config_dict.get("llm_provider", "none")
        self.parsing_mode = config_dict.get("parsing_mode", "text_only")
        self.extract_images = config_dict.get("extract_images", True)

        self.config = ParserConfig()
        self.config.parsing_mode = self.parsing_mode
        self.config.max_pages_per_batch = 5

        if self.llm_provider_name != "none" and self.parsing_mode != "text_only":
            if self.llm_provider_name == "ollama":
                # Support new separate models for image and text
                image_model = config_dict.get("ollama_image_model")
                text_model = config_dict.get("ollama_text_model")

                # Fallback to llm_model if specific models not provided
                if not image_model and not text_model:
                    model = config_dict.get("llm_model", "llava:latest")
                    image_model = model
                    text_model = model

                base_url = config_dict.get("ollama_url", "http://localhost:11434")
                llm_provider = OllamaProvider(
                    image_model=image_model, text_model=text_model, base_url=base_url
                )
            elif self.llm_provider_name == "openai":
                import os

                model = config_dict.get("llm_model", "gpt-4-vision-preview")

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found in environment variables"
                    )
                llm_provider = OpenAIProvider(api_key=api_key, model=model)
            else:
                raise ValueError(f"Unknown LLM provider: {self.llm_provider_name}")

            self.strategy = LLMEnhancedStrategy(llm_provider, self.config)
        else:
            self.strategy = TextOnlyStrategy()

        self._init_parsers()

    def _init_parsers(self) -> None:
        """Initialize specific parsers with the configured strategy."""
        parser_args = {
            "parsing_strategy": self.strategy,
            "extract_images": self.extract_images,
        }

        self.parsers: Dict[str, BaseParser] = {
            ".pdf": PDFParser(**parser_args),
            ".docx": WordParser(**parser_args),
            ".doc": WordParser(**parser_args),
            ".pptx": PowerPointParser(**parser_args),
            ".ppt": PowerPointParser(**parser_args),
        }

    def parse(self, file_path: Path) -> Document:
        """Parse a document based on its file extension."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()
        if file_extension not in self.parsers:
            raise ValueError(f"Unsupported file type: {file_extension}")

        parser = self.parsers[file_extension]
        document = asyncio.run(parser.parse(file_path))

        if not document.metadata:
            document.metadata = DocumentMetadata(
                filename=file_path.name,
                file_type=file_extension[1:],
                file_path=str(file_path.absolute()),
                file_size=file_path.stat().st_size,
            )

        return document

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file type is supported."""
        return file_path.suffix.lower() in self.parsers
