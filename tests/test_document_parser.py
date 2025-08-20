"""
Tests for document parser functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from doc_indexer.models import Document, DocumentMetadata
from doc_indexer.parser_factory import DocumentParser
from doc_indexer.parsers import PDFParser, PowerPointParser, WordParser
from doc_indexer.parsers.strategies.text_only import TextOnlyStrategy


class TestDocumentParser:
    """Tests for the main DocumentParser class."""

    @pytest.fixture
    def parser(self):
        """Create a DocumentParser instance."""
        return DocumentParser()

    def test_parser_initialization(self, parser):
        """Test that parser initializes with correct parsers."""
        assert parser is not None
        assert hasattr(parser, "parsers")
        assert ".pdf" in parser.parsers
        assert ".docx" in parser.parsers
        assert ".pptx" in parser.parsers

    def test_parse_pdf_file(self, parser):
        """Test parsing a PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with patch.object(PDFParser, "parse") as mock_parse:
            mock_parse.return_value = Document(
                content="Test PDF content",
                metadata=DocumentMetadata(
                    filename="test.pdf", file_type="pdf", file_path=str(tmp_path)
                ),
            )

            result = parser.parse(tmp_path)

            assert result is not None
            assert result.content == "Test PDF content"
            assert result.metadata.file_type == "pdf"
            mock_parse.assert_called_once()

        tmp_path.unlink()

    def test_parse_word_file(self, parser):
        """Test parsing a Word document."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with patch.object(WordParser, "parse") as mock_parse:
            mock_parse.return_value = Document(
                content="Test Word content",
                metadata=DocumentMetadata(
                    filename="test.docx", file_type="docx", file_path=str(tmp_path)
                ),
            )

            result = parser.parse(tmp_path)

            assert result is not None
            assert result.content == "Test Word content"
            assert result.metadata.file_type == "docx"
            mock_parse.assert_called_once()

        tmp_path.unlink()

    def test_parse_powerpoint_file(self, parser):
        """Test parsing a PowerPoint presentation."""
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with patch.object(PowerPointParser, "parse") as mock_parse:
            mock_parse.return_value = Document(
                content="Test PowerPoint content",
                metadata=DocumentMetadata(
                    filename="test.pptx", file_type="pptx", file_path=str(tmp_path)
                ),
            )

            result = parser.parse(tmp_path)

            assert result is not None
            assert result.content == "Test PowerPoint content"
            assert result.metadata.file_type == "pptx"
            mock_parse.assert_called_once()

        tmp_path.unlink()

    def test_parse_unsupported_file_type(self, parser):
        """Test that unsupported file types raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(tmp_path)

        tmp_path.unlink()

    def test_parse_nonexistent_file(self, parser):
        """Test that parsing nonexistent file raises an error."""
        fake_path = Path("/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            parser.parse(fake_path)


class TestPDFParser:
    """Tests for PDF parser."""

    @pytest.fixture
    def parser(self):
        """Create a PDFParser instance."""
        strategy = TextOnlyStrategy()
        return PDFParser(parsing_strategy=strategy)

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.pdf_parser.PdfReader")
    async def test_parse_simple_pdf(self, mock_pdf_reader, parser):
        """Test parsing a simple PDF."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page 1 content"

        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = await parser.parse(tmp_path)

        assert result.content == "Page 1 content"
        assert result.metadata.filename == tmp_path.name
        assert result.metadata.file_type == "pdf"

        tmp_path.unlink()

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.pdf_parser.PdfReader")
    async def test_parse_multi_page_pdf(self, mock_pdf_reader, parser):
        """Test parsing a multi-page PDF."""
        mock_pages = []
        for i in range(3):
            page = Mock()
            page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(page)

        mock_reader = Mock()
        mock_reader.pages = mock_pages
        mock_pdf_reader.return_value = mock_reader

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = await parser.parse(tmp_path)

        assert "Page 1 content" in result.content
        assert "Page 2 content" in result.content
        assert "Page 3 content" in result.content

        tmp_path.unlink()


class TestWordParser:
    """Tests for Word document parser."""

    @pytest.fixture
    def parser(self):
        """Create a WordParser instance."""
        strategy = TextOnlyStrategy()
        return WordParser(parsing_strategy=strategy)

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.word_parser.DocxDocument")
    async def test_parse_word_document(self, mock_docx, parser):
        """Test parsing a Word document."""
        mock_para1 = Mock()
        mock_para1.text = "Paragraph 1"
        mock_para1.runs = []  # No runs with page breaks
        mock_para1._element = Mock()
        mock_para1._element.getnext.return_value = None

        mock_para2 = Mock()
        mock_para2.text = "Paragraph 2"
        mock_para2.runs = []
        mock_para2._element = Mock()
        mock_para2._element.getnext.return_value = None

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []  # Add empty tables list
        mock_doc.part = Mock()
        mock_doc.part.rels = Mock()
        mock_doc.part.rels.values.return_value = []  # No images
        mock_docx.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = await parser.parse(tmp_path)

        assert "Paragraph 1" in result.content
        assert "Paragraph 2" in result.content
        assert result.metadata.file_type == "docx"

        tmp_path.unlink()


class TestPowerPointParser:
    """Tests for PowerPoint presentation parser."""

    @pytest.fixture
    def parser(self):
        """Create a PowerPointParser instance."""
        strategy = TextOnlyStrategy()
        return PowerPointParser(parsing_strategy=strategy)

    @pytest.mark.asyncio
    @patch("doc_indexer.parsers.powerpoint_parser.Presentation")
    async def test_parse_powerpoint(self, mock_presentation, parser):
        """Test parsing a PowerPoint presentation."""
        # Create mock text frames and shapes
        mock_text_frame = Mock()
        mock_text_frame.text = "Slide content"
        mock_paragraph = Mock()
        mock_paragraph.level = 0
        mock_paragraph.text = "Slide content"
        mock_text_frame.paragraphs = [mock_paragraph]

        mock_shape = Mock()
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_text_frame

        mock_slide = Mock()
        mock_shapes = Mock()
        mock_shapes.__iter__ = Mock(return_value=iter([mock_shape]))
        mock_shapes.title = None
        mock_slide.shapes = mock_shapes
        mock_slide.has_notes_slide = False

        mock_prs = Mock()
        mock_prs.slides = [mock_slide]
        mock_presentation.return_value = mock_prs

        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = await parser.parse(tmp_path)

        assert "Slide content" in result.content
        assert result.metadata.file_type == "pptx"

        tmp_path.unlink()
