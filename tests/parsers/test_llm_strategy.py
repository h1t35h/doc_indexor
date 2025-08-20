"""
Tests for parsing strategy implementations.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

# Removed TestParsingStrategies class - fixtures moved to individual test classes


class TestTextOnlyStrategy:
    """Tests for text-only parsing strategy."""

    @pytest.fixture
    def mock_pages(self):
        """Create mock page content."""
        from doc_indexer.parsers.base import PageContent

        return [
            PageContent(
                page_number=1,
                text="Page 1 text",
                image=Image.new("RGB", (100, 100)),
                tables=[{"header": ["Col1"], "rows": [["Data1"]]}],
            ),
            PageContent(
                page_number=2,
                text="Page 2 text",
                image=None,
                tables=[],
            ),
        ]

    @pytest.mark.asyncio
    async def test_text_only_process_pages(self, mock_pages):
        """Test text-only strategy page processing."""
        from doc_indexer.parsers.strategies.text_only import TextOnlyStrategy

        strategy = TextOnlyStrategy()
        result = await strategy.process_pages(mock_pages)

        assert "Page 1 text" in result
        assert "Page 2 text" in result
        assert "Col1" in result
        assert "Data1" in result

    @pytest.mark.asyncio
    async def test_text_only_empty_pages(self):
        """Test text-only strategy with empty pages."""
        from doc_indexer.parsers.strategies.text_only import TextOnlyStrategy

        strategy = TextOnlyStrategy()
        result = await strategy.process_pages([])

        assert result == ""

    @pytest.mark.asyncio
    async def test_text_only_with_tables(self):
        """Test text-only strategy with table extraction."""
        from doc_indexer.parsers.base import PageContent
        from doc_indexer.parsers.strategies.text_only import TextOnlyStrategy

        pages = [
            PageContent(
                page_number=1,
                text="",
                image=None,
                tables=[
                    {
                        "header": ["Name", "Age"],
                        "rows": [["Alice", "30"], ["Bob", "25"]],
                    }
                ],
            )
        ]

        strategy = TextOnlyStrategy()
        result = await strategy.process_pages(pages)

        assert "Name" in result
        assert "Age" in result
        assert "Alice" in result
        assert "30" in result


class TestLLMEnhancedStrategy:
    """Tests for LLM-enhanced parsing strategy."""

    @pytest.fixture
    def mock_pages(self):
        """Create mock page content."""
        from doc_indexer.parsers.base import PageContent

        return [
            PageContent(
                page_number=1,
                text="Page 1 text",
                image=Image.new("RGB", (100, 100)),
                tables=[{"header": ["Col1"], "rows": [["Data1"]]}],
            ),
            PageContent(
                page_number=2,
                text="Page 2 text",
                image=None,
                tables=[],
            ),
        ]

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.analyze_image = AsyncMock(return_value="Image analysis result")
        provider.analyze_text = AsyncMock(return_value="Text analysis result")
        return provider

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        from doc_indexer.parsers.config import ParserConfig

        config = ParserConfig()
        config.parsing_mode = "hybrid"
        return config

    @pytest.mark.asyncio
    async def test_llm_enhanced_initialization(self, mock_llm_provider, mock_config):
        """Test LLM-enhanced strategy initialization."""
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        assert strategy.llm == mock_llm_provider
        assert strategy.config == mock_config

    @pytest.mark.asyncio
    async def test_llm_enhanced_process_pages_hybrid(
        self, mock_pages, mock_llm_provider, mock_config
    ):
        """Test LLM-enhanced strategy in hybrid mode."""
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        mock_config.parsing_mode = "hybrid"
        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        result = await strategy.process_pages(mock_pages)

        # Should analyze both image and text for page 1
        assert mock_llm_provider.analyze_image.called
        assert mock_llm_provider.analyze_text.called
        assert "Image analysis result" in result
        assert "Text analysis result" in result

    @pytest.mark.asyncio
    async def test_llm_enhanced_process_pages_llm_only(
        self, mock_pages, mock_llm_provider, mock_config
    ):
        """Test LLM-enhanced strategy in LLM-only mode."""
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        mock_config.parsing_mode = "llm_only"
        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        result = await strategy.process_pages(mock_pages)

        # Should only use LLM analysis
        assert mock_llm_provider.analyze_image.called
        assert "Image analysis result" in result

    @pytest.mark.asyncio
    async def test_llm_enhanced_error_handling(
        self, mock_pages, mock_llm_provider, mock_config
    ):
        """Test LLM-enhanced strategy error handling."""
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        # Make LLM provider raise an error on image analysis only
        mock_llm_provider.analyze_image = AsyncMock(side_effect=Exception("LLM error"))

        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        # Should handle error gracefully and still process text
        result = await strategy.process_pages(mock_pages)
        # Check that text analysis still worked
        assert "Text analysis result" in result
        # Check that pages are included
        assert "Page 1" in result or "Page 2" in result

    @pytest.mark.asyncio
    async def test_llm_enhanced_prompt_building(self, mock_llm_provider, mock_config):
        """Test LLM prompt building."""
        from doc_indexer.parsers.base import PageContent
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        page = PageContent(
            page_number=1,
            text="Sample text",
            image=Image.new("RGB", (100, 100)),
            tables=[],
        )

        prompt = strategy._build_extraction_prompt(page)

        assert "extract" in prompt.lower()
        assert "text" in prompt.lower()
        assert "table" in prompt.lower() or "structure" in prompt.lower()

    @pytest.mark.asyncio
    async def test_llm_enhanced_combine_results(self, mock_llm_provider, mock_config):
        """Test result combination in LLM-enhanced strategy."""
        from doc_indexer.parsers.strategies.llm_enhanced import LLMEnhancedStrategy

        strategy = LLMEnhancedStrategy(
            llm_provider=mock_llm_provider, config=mock_config
        )

        results = [
            "Page 1: Content A",
            "Page 2: Content B",
            "Page 3: Content C",
        ]

        combined = strategy._combine_results(results)

        assert "Content A" in combined
        assert "Content B" in combined
        assert "Content C" in combined
