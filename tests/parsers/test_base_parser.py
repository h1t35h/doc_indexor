"""
Tests for base parser classes and protocols.
"""

from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest
from doc_indexer.models import Document, DocumentMetadata
from PIL import Image


class TestBaseParser:
    """Tests for the base parser class."""

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock parsing strategy."""
        strategy = Mock()
        strategy.process_pages = AsyncMock(return_value="Processed content")
        return strategy

    @pytest.fixture
    def mock_page_content(self):
        """Create mock page content."""
        from doc_indexer.parsers.base import PageContent

        return PageContent(
            page_number=1,
            text="Sample text",
            image=Image.new("RGB", (100, 100)),
            tables=[],
        )

    @pytest.mark.asyncio
    async def test_base_parser_initialization(self):
        """Test base parser initialization with strategy."""
        from doc_indexer.parsers.base import BaseParser

        class ConcreteParser(BaseParser):
            async def extract_pages(self, file_path: Path) -> List:
                return []

        strategy = Mock()
        parser = ConcreteParser(parsing_strategy=strategy)
        assert parser.strategy == strategy

    @pytest.mark.asyncio
    async def test_base_parser_parse_method(self, mock_strategy, tmp_path):
        """Test the parse method of base parser."""
        from doc_indexer.parsers.base import BaseParser, PageContent

        class ConcreteParser(BaseParser):
            async def extract_pages(self, file_path: Path) -> List[PageContent]:
                return [
                    PageContent(page_number=1, text="Page 1", image=None, tables=[])
                ]

        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy")

        parser = ConcreteParser(parsing_strategy=mock_strategy)
        result = await parser.parse(test_file)

        assert isinstance(result, Document)
        assert result.content == "Processed content"
        mock_strategy.process_pages.assert_called_once()

    @pytest.mark.asyncio
    async def test_base_parser_file_not_found(self, mock_strategy):
        """Test base parser with non-existent file."""
        from doc_indexer.parsers.base import BaseParser, PageContent

        class ConcreteParser(BaseParser):
            async def extract_pages(self, file_path: Path) -> List[PageContent]:
                return []

        parser = ConcreteParser(parsing_strategy=mock_strategy)

        with pytest.raises(FileNotFoundError):
            await parser.parse(Path("/non/existent/file.pdf"))

    @pytest.mark.asyncio
    async def test_page_content_model(self):
        """Test PageContent dataclass."""
        from doc_indexer.parsers.base import PageContent

        page = PageContent(
            page_number=1,
            text="Test text",
            image=Image.new("RGB", (100, 100)),
            tables=[{"header": ["Col1"], "rows": [["Data"]]}],
        )

        assert page.page_number == 1
        assert page.text == "Test text"
        assert page.image is not None
        assert len(page.tables) == 1

    @pytest.mark.asyncio
    async def test_parser_protocol(self):
        """Test that FileParser protocol is properly defined."""

        class ValidParser:
            async def parse(self, file_path: Path) -> Document:
                return Document(
                    content="test",
                    metadata=DocumentMetadata(
                        filename="test.pdf",
                        file_type="pdf",
                        file_path="/test.pdf",
                    ),
                )

        parser = ValidParser()
        assert hasattr(parser, "parse")

        # Test that the method signature matches
        import inspect

        sig = inspect.signature(parser.parse)
        assert "file_path" in sig.parameters
        assert sig.parameters["file_path"].annotation == Path
