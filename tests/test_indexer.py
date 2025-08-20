"""
Tests for the main document indexer.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from doc_indexer.indexer import DocumentIndexer
from doc_indexer.models import Document, DocumentMetadata, SearchResult


class TestDocumentIndexer:
    """Tests for DocumentIndexer class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        with patch("doc_indexer.indexer.VectorStore") as mock:
            yield mock

    @pytest.fixture
    def mock_parser(self):
        """Create a mock document parser."""
        with patch("doc_indexer.indexer.DocumentParser") as mock:
            yield mock

    @pytest.fixture
    def indexer(self, mock_vector_store, mock_parser):
        """Create a DocumentIndexer with mocked dependencies."""
        mock_vs_instance = Mock()
        mock_vector_store.return_value = mock_vs_instance

        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance

        indexer = DocumentIndexer(persist_directory="./test_db")
        indexer.vector_store = mock_vs_instance
        indexer.parser = mock_parser_instance

        return indexer

    def test_indexer_initialization(self, mock_vector_store, mock_parser):
        """Test indexer initialization."""
        indexer = DocumentIndexer(persist_directory="./test_db")

        assert indexer is not None
        assert indexer.persist_directory == "./test_db"
        mock_vector_store.assert_called_once_with(persist_directory="./test_db")
        mock_parser.assert_called_once()

    def test_index_file_success(self, indexer):
        """Test successfully indexing a file."""
        test_file = Path("/test/document.pdf")

        mock_document = Document(
            content="Test content",
            metadata=DocumentMetadata(
                filename="document.pdf", file_type="pdf", file_path="/test/document.pdf"
            ),
        )

        indexer.parser.is_supported.return_value = True
        indexer.parser.parse.return_value = mock_document

        result = indexer.index_file(test_file)

        assert result is True
        indexer.parser.is_supported.assert_called_once_with(test_file)
        indexer.parser.parse.assert_called_once_with(test_file)
        indexer.vector_store.add_document.assert_called_once()

        # Check that indexed_at was set
        added_doc = indexer.vector_store.add_document.call_args[0][0]
        assert added_doc.metadata.indexed_at is not None

    def test_index_file_unsupported(self, indexer):
        """Test indexing an unsupported file type."""
        test_file = Path("/test/document.txt")

        indexer.parser.is_supported.return_value = False

        result = indexer.index_file(test_file)

        assert result is False
        indexer.parser.is_supported.assert_called_once_with(test_file)
        indexer.parser.parse.assert_not_called()
        indexer.vector_store.add_document.assert_not_called()

    def test_index_file_parse_error(self, indexer, capsys):
        """Test handling parse error."""
        test_file = Path("/test/document.pdf")

        indexer.parser.is_supported.return_value = True
        indexer.parser.parse.side_effect = Exception("Parse error")

        result = indexer.index_file(test_file)

        assert result is False
        captured = capsys.readouterr()
        assert "Error indexing" in captured.out
        assert "Parse error" in captured.out

    @patch("doc_indexer.indexer.tqdm")
    def test_index_directory_success(self, mock_tqdm, indexer):
        """Test indexing a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            pdf_file = tmpdir_path / "doc1.pdf"
            pdf_file.touch()
            docx_file = tmpdir_path / "doc2.docx"
            docx_file.touch()

            # Setup mock parser
            indexer.parser.parsers = {".pdf": Mock(), ".docx": Mock()}

            mock_document = Document(
                content="Test content",
                metadata=DocumentMetadata(
                    filename="test.pdf", file_type="pdf", file_path="/test.pdf"
                ),
            )
            indexer.parser.parse.return_value = mock_document

            # Mock tqdm context manager
            mock_progress = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress

            result = indexer.index_directory(tmpdir_path)

            assert result == 2
            assert indexer.parser.parse.call_count == 2
            indexer.vector_store.add_documents.assert_called()

    def test_index_directory_not_exists(self, indexer):
        """Test indexing non-existent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            indexer.index_directory(Path("/nonexistent"))

    def test_index_directory_not_dir(self, indexer):
        """Test indexing a file instead of directory."""
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValueError, match="Path is not a directory"):
                indexer.index_directory(Path(tmp.name))

    @patch("doc_indexer.indexer.tqdm")
    def test_index_directory_no_supported_files(self, mock_tqdm, indexer, capsys):
        """Test indexing directory with no supported files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create only unsupported files
            txt_file = tmpdir_path / "doc.txt"
            txt_file.touch()

            indexer.parser.parsers = {".pdf": Mock(), ".docx": Mock()}

            result = indexer.index_directory(tmpdir_path)

            assert result == 0
            captured = capsys.readouterr()
            assert "No supported documents found" in captured.out

    def test_search(self, indexer):
        """Test search functionality."""
        mock_results = [
            SearchResult(
                content="Result 1",
                metadata=DocumentMetadata(
                    filename="doc1.pdf", file_type="pdf", file_path="/doc1.pdf"
                ),
                score=0.9,
            )
        ]

        indexer.vector_store.search.return_value = mock_results

        results = indexer.search("test query", n_results=5)

        assert results == mock_results
        indexer.vector_store.search.assert_called_once_with("test query", n_results=5)

    def test_clear_index(self, indexer):
        """Test clearing the index."""
        indexer.clear_index()

        indexer.vector_store.delete_collection.assert_called_once()

    def test_get_stats(self, indexer):
        """Test getting statistics."""
        indexer.vector_store.get_document_count.return_value = 42

        stats = indexer.get_stats()

        assert stats["total_documents"] == 42
        assert stats["collection_name"] == "documents"
        assert stats["persist_directory"] == "./test_db"
        indexer.vector_store.get_document_count.assert_called_once()
