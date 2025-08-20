"""
Tests for configurable Ollama models via CLI.
"""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from doc_indexer.cli import main
from doc_indexer.parsers.config import ParserConfig
from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider


class TestOllamaModelConfiguration:
    """Tests for configurable Ollama models."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory with test files."""
        # Create test files
        (tmp_path / "test.pdf").touch()
        return str(tmp_path)

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_ollama_image_model_cli_parameter(
        self, mock_indexer_class, runner, temp_dir
    ):
        """Test --ollama-image-model CLI parameter."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 1
        mock_indexer.get_stats.return_value = {"total_documents": 1}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(
            main,
            [
                "index",
                temp_dir,
                "--llm-provider",
                "ollama",
                "--ollama-image-model",
                "llava:13b",
            ],
        )

        assert result.exit_code == 0
        # Verify the parser config was passed correctly
        call_args = mock_indexer_class.call_args
        parser_config = call_args[1]["parser_config"]
        assert parser_config["ollama_image_model"] == "llava:13b"

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_ollama_text_model_cli_parameter(
        self, mock_indexer_class, runner, temp_dir
    ):
        """Test --ollama-text-model CLI parameter."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 1
        mock_indexer.get_stats.return_value = {"total_documents": 1}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(
            main,
            [
                "index",
                temp_dir,
                "--llm-provider",
                "ollama",
                "--ollama-text-model",
                "llama2:70b",
            ],
        )

        assert result.exit_code == 0
        # Verify the parser config was passed correctly
        call_args = mock_indexer_class.call_args
        parser_config = call_args[1]["parser_config"]
        assert parser_config["ollama_text_model"] == "llama2:70b"

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_both_ollama_models_configured(self, mock_indexer_class, runner, temp_dir):
        """Test configuring both image and text models."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 1
        mock_indexer.get_stats.return_value = {"total_documents": 1}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(
            main,
            [
                "index",
                temp_dir,
                "--llm-provider",
                "ollama",
                "--ollama-image-model",
                "llava:34b",
                "--ollama-text-model",
                "mistral:latest",
                "--parsing-mode",
                "hybrid",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_indexer_class.call_args
        parser_config = call_args[1]["parser_config"]
        assert parser_config["ollama_image_model"] == "llava:34b"
        assert parser_config["ollama_text_model"] == "mistral:latest"

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_ollama_models_with_default_fallback(
        self, mock_indexer_class, runner, temp_dir
    ):
        """Test that models have sensible defaults when not specified."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 1
        mock_indexer.get_stats.return_value = {"total_documents": 1}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["index", temp_dir, "--llm-provider", "ollama"])

        assert result.exit_code == 0
        # Should not have these keys if not specified
        call_args = mock_indexer_class.call_args
        parser_config = call_args[1]["parser_config"]
        # These keys should not be present if not specified
        assert (
            "ollama_image_model" not in parser_config
            or parser_config["ollama_image_model"] is None
        )
        assert (
            "ollama_text_model" not in parser_config
            or parser_config["ollama_text_model"] is None
        )

    def test_parser_config_with_ollama_models(self):
        """Test ParserConfig properly handles Ollama model settings."""
        config = ParserConfig()
        config.llm_provider = "ollama"
        config.ollama_image_model = "llava:custom"
        config.ollama_text_model = "llama3:custom"

        assert config.ollama_image_model == "llava:custom"
        assert config.ollama_text_model == "llama3:custom"

    @pytest.mark.asyncio
    async def test_ollama_provider_uses_correct_models(self):
        """Test OllamaProvider uses specified models for different tasks."""
        # Test with custom image model
        provider = OllamaProvider(
            image_model="llava:34b",
            text_model="llama2:70b",
            base_url="http://localhost:11434",
        )

        assert provider.image_model == "llava:34b"
        assert provider.text_model == "llama2:70b"

    @pytest.mark.asyncio
    async def test_ollama_provider_defaults(self):
        """Test OllamaProvider has sensible defaults."""
        provider = OllamaProvider()

        # Should have default models
        assert provider.image_model == "llava"  # Default image model
        assert provider.text_model == "llama2"  # Default text model

    @patch("doc_indexer.parsers.llm_providers.factory.OllamaProvider")
    def test_factory_creates_provider_with_models(self, mock_provider_class):
        """Test LLMProviderFactory passes model configuration correctly."""
        from doc_indexer.parsers.llm_providers.factory import LLMProviderFactory

        config = ParserConfig()
        config.llm_provider = "ollama"
        config.ollama_image_model = "bakllava"
        config.ollama_text_model = "mixtral"
        config.ollama_base_url = "http://localhost:11434"

        LLMProviderFactory.create(config)

        mock_provider_class.assert_called_once_with(
            image_model="bakllava",
            text_model="mixtral",
            base_url="http://localhost:11434",
        )

    def test_cli_help_shows_ollama_model_options(self, runner):
        """Test that CLI help text includes Ollama model options."""
        result = runner.invoke(main, ["index", "--help"])

        assert result.exit_code == 0
        assert "--ollama-image-model" in result.output
        assert "--ollama-text-model" in result.output
        # Check descriptions are present
        assert "vision" in result.output.lower() or "image" in result.output.lower()
