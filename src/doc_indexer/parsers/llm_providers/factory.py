"""
Factory for creating LLM providers.
"""

from typing import Optional

from doc_indexer.parsers.config import ParserConfig

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    @staticmethod
    def create(config: ParserConfig) -> LLMProvider:
        """
        Create an LLM provider based on configuration.

        Args:
            config: Parser configuration

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is unknown or misconfigured
        """
        provider_name = config.llm_provider.lower()

        if provider_name == "ollama":
            # Use new image_model and text_model if available, fallback to ollama_model
            image_model = config.ollama_image_model or config.ollama_model
            text_model = config.ollama_text_model or config.ollama_model

            return OllamaProvider(
                image_model=image_model,
                text_model=text_model,
                base_url=config.ollama_base_url,
            )
        elif provider_name == "openai":
            if not config.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return OpenAIProvider(
                api_key=config.openai_api_key, model=config.openai_model
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")

    @staticmethod
    def create_with_fallback(config: ParserConfig) -> Optional[LLMProvider]:
        """
        Create an LLM provider with fallback to Ollama if primary fails.

        Args:
            config: Parser configuration

        Returns:
            LLM provider instance or None if all fail
        """
        try:
            return LLMProviderFactory.create(config)
        except Exception:
            # Try fallback to Ollama
            if config.llm_provider != "ollama":
                try:
                    return OllamaProvider()
                except Exception:
                    pass
            return None
