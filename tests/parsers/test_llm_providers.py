"""
Tests for LLM provider implementations.
"""

import base64
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image


class TestLLMProviders:
    """Tests for LLM provider implementations."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new("RGB", (100, 100), color="white")

    @pytest.fixture
    def config(self):
        """Create a mock configuration."""
        from doc_indexer.parsers.config import ParserConfig

        config = ParserConfig()
        config.llm_provider = "ollama"
        config.ollama_model = "llava"
        config.ollama_base_url = "http://localhost:11434"
        config.openai_api_key = "test-key"
        config.openai_model = "gpt-4-vision-preview"
        return config


class TestOllamaProvider:
    """Tests for Ollama LLM provider."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new("RGB", (100, 100), color="white")

    @pytest.fixture
    def ollama_provider(self):
        """Create an Ollama provider instance."""
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        return OllamaProvider(model="llava", base_url="http://localhost:11434")

    @pytest.mark.asyncio
    async def test_ollama_initialization(self):
        """Test Ollama provider initialization."""
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        provider = OllamaProvider(model="llava", base_url="http://localhost:11434")
        assert provider.model == "llava"
        assert provider.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_ollama_analyze_image(self, sample_image):
        """Test Ollama image analysis."""
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        with patch(
            "doc_indexer.parsers.llm_providers.ollama_provider.aiohttp"
        ) as mock_aiohttp:
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"message": {"content": "Extracted text from image"}}
            )

            # Create a proper async context manager for post
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            # Create a proper async context manager for session
            mock_session = AsyncMock()
            mock_session.post = Mock(return_value=mock_post)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)

            # Mock ClientSession and ClientError
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_cm)
            mock_aiohttp.ClientError = Exception
            mock_aiohttp.ClientTimeout = Mock(return_value=None)

            provider = OllamaProvider(model="llava")
            result = await provider.analyze_image(sample_image, "Extract text")

            assert result == "Extracted text from image"

    @pytest.mark.asyncio
    async def test_ollama_analyze_text(self):
        """Test Ollama text analysis."""
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        with patch(
            "doc_indexer.parsers.llm_providers.ollama_provider.aiohttp"
        ) as mock_aiohttp:
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"message": {"content": "Enhanced text"}}
            )

            # Create a proper async context manager for post
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            # Create a proper async context manager for session
            mock_session = AsyncMock()
            mock_session.post = Mock(return_value=mock_post)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)

            # Mock ClientSession and ClientError
            mock_aiohttp.ClientSession = Mock(return_value=mock_session_cm)
            mock_aiohttp.ClientError = Exception
            mock_aiohttp.ClientTimeout = Mock(return_value=None)

            provider = OllamaProvider(model="llama2")
            result = await provider.analyze_text("Sample text", "Enhance this text")

            assert result == "Enhanced text"

    @pytest.mark.asyncio
    async def test_ollama_image_to_base64(self, ollama_provider, sample_image):
        """Test image to base64 conversion."""
        base64_str = ollama_provider._image_to_base64(sample_image)

        # Verify it's valid base64
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.llm_providers.ollama_provider.aiohttp.ClientSession")
    async def test_ollama_error_handling(self, mock_session_class, sample_image):
        """Test Ollama error handling."""
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        # Mock aiohttp session with error
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(
            side_effect=Exception("Failed to connect to Ollama")
        )
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        provider = OllamaProvider(model="llava")

        with pytest.raises(Exception, match="Failed to connect to Ollama"):
            await provider.analyze_image(sample_image, "Extract text")


class TestOpenAIProvider:
    """Tests for OpenAI LLM provider."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return Image.new("RGB", (100, 100), color="white")

    @pytest.fixture
    def openai_provider(self):
        """Create an OpenAI provider instance."""
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key="test-key", model="gpt-4-vision-preview")

    @pytest.mark.asyncio
    async def test_openai_initialization(self):
        """Test OpenAI provider initialization."""
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key", model="gpt-4-vision-preview")
        # API key is no longer stored as instance variable for security
        assert provider.model == "gpt-4-vision-preview"

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.llm_providers.openai_provider.ChatOpenAI")
    async def test_openai_analyze_image(self, mock_openai_class, sample_image):
        """Test OpenAI image analysis."""
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Extracted text from image"
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")
        result = await provider.analyze_image(sample_image, "Extract text")

        assert result == "Extracted text from image"
        mock_client.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("doc_indexer.parsers.llm_providers.openai_provider.ChatOpenAI")
    async def test_openai_analyze_text(self, mock_openai_class):
        """Test OpenAI text analysis."""
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = "Enhanced text"
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
        result = await provider.analyze_text("Sample text", "Enhance this text")

        assert result == "Enhanced text"
        mock_client.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_missing_api_key(self):
        """Test OpenAI provider with missing API key."""
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        with pytest.raises(ValueError, match="API key is required"):
            OpenAIProvider(api_key="", model="gpt-4-vision-preview")


class TestLLMProviderFactory:
    """Tests for LLM provider factory."""

    @pytest.fixture
    def config(self):
        """Create a mock configuration."""
        from doc_indexer.parsers.config import ParserConfig

        config = ParserConfig()
        config.llm_provider = "ollama"
        config.ollama_model = "llava"
        config.ollama_base_url = "http://localhost:11434"
        config.openai_api_key = "test-key"
        config.openai_model = "gpt-4-vision-preview"
        return config

    @pytest.mark.asyncio
    async def test_factory_create_ollama(self, config):
        """Test factory creation of Ollama provider."""
        from doc_indexer.parsers.llm_providers.factory import LLMProviderFactory
        from doc_indexer.parsers.llm_providers.ollama_provider import OllamaProvider

        config.llm_provider = "ollama"
        provider = LLMProviderFactory.create(config)

        assert isinstance(provider, OllamaProvider)

    @pytest.mark.asyncio
    async def test_factory_create_openai(self, config):
        """Test factory creation of OpenAI provider."""
        from doc_indexer.parsers.llm_providers.factory import LLMProviderFactory
        from doc_indexer.parsers.llm_providers.openai_provider import OpenAIProvider

        config.llm_provider = "openai"
        config.openai_api_key = "test-key"
        provider = LLMProviderFactory.create(config)

        assert isinstance(provider, OpenAIProvider)

    @pytest.mark.asyncio
    async def test_factory_invalid_provider(self, config):
        """Test factory with invalid provider name."""
        from doc_indexer.parsers.llm_providers.factory import LLMProviderFactory

        config.llm_provider = "invalid"

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create(config)

    @pytest.mark.asyncio
    async def test_factory_openai_no_api_key(self, config):
        """Test factory OpenAI creation without API key."""
        from doc_indexer.parsers.llm_providers.factory import LLMProviderFactory

        config.llm_provider = "openai"
        config.openai_api_key = ""

        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            LLMProviderFactory.create(config)
