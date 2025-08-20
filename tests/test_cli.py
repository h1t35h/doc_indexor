"""
Tests for CLI interface.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from doc_indexer.cli import main


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.pdf").touch()
            Path(tmpdir, "test2.docx").touch()
            Path(tmpdir, "test3.pptx").touch()
            Path(tmpdir, "test4.txt").touch()  # Unsupported file

            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            Path(subdir, "nested.pdf").touch()

            yield tmpdir

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Document indexer CLI" in result.output
        assert "index" in result.output
        assert "search" in result.output
        assert "stats" in result.output

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_index_command_with_directory(self, mock_indexer_class, runner, temp_dir):
        """Test indexing a directory."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 3
        mock_indexer.get_stats.return_value = {"total_documents": 3}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["index", temp_dir])

        assert result.exit_code == 0
        assert "Successfully indexed 3 documents" in result.output
        mock_indexer.index_directory.assert_called_once_with(Path(temp_dir))

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_index_command_with_persist_dir(self, mock_indexer_class, runner, temp_dir):
        """Test indexing with custom persist directory."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 2
        mock_indexer.get_stats.return_value = {"total_documents": 2}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(
            main, ["index", temp_dir, "--persist-dir", "./custom_db"]
        )

        assert result.exit_code == 0
        mock_indexer_class.assert_called_once_with(persist_directory="./custom_db")

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_index_command_with_clear_flag(self, mock_indexer_class, runner, temp_dir):
        """Test indexing with clear flag."""
        mock_indexer = Mock()
        mock_indexer.index_directory.return_value = 2
        mock_indexer.get_stats.return_value = {"total_documents": 2}
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["index", temp_dir, "--clear"], input="y\n")

        assert result.exit_code == 0
        assert "Are you sure you want to clear" in result.output
        mock_indexer.clear_index.assert_called_once()

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_index_command_clear_cancelled(self, mock_indexer_class, runner, temp_dir):
        """Test cancelling clear operation."""
        mock_indexer = Mock()
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["index", temp_dir, "--clear"], input="n\n")

        assert result.exit_code == 0
        assert "Operation cancelled" in result.output
        mock_indexer.clear_index.assert_not_called()

    def test_index_nonexistent_directory(self, runner):
        """Test indexing nonexistent directory."""
        result = runner.invoke(main, ["index", "/nonexistent/directory"])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_search_command(self, mock_indexer_class, runner):
        """Test search command."""
        mock_indexer = Mock()
        mock_result = Mock()
        mock_result.content = "Found content"
        mock_result.metadata = Mock(filename="test.pdf", file_path="/path/to/test.pdf")
        mock_result.score = 0.95

        mock_indexer.search.return_value = [mock_result]
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["search", "test query"])

        assert result.exit_code == 0
        assert "Found 1 results" in result.output
        assert "test.pdf" in result.output
        assert "0.95" in result.output
        mock_indexer.search.assert_called_once_with("test query", n_results=5)

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_search_command_with_limit(self, mock_indexer_class, runner):
        """Test search command with custom result limit."""
        mock_indexer = Mock()
        mock_indexer.search.return_value = []
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["search", "test query", "--limit", "10"])

        assert result.exit_code == 0
        mock_indexer.search.assert_called_once_with("test query", n_results=10)

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_search_no_results(self, mock_indexer_class, runner):
        """Test search with no results."""
        mock_indexer = Mock()
        mock_indexer.search.return_value = []
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["search", "nonexistent query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_stats_command(self, mock_indexer_class, runner):
        """Test stats command."""
        mock_indexer = Mock()
        mock_indexer.get_stats.return_value = {
            "total_documents": 42,
            "collection_name": "documents",
            "persist_directory": "./chroma_db",
        }
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["stats"])

        assert result.exit_code == 0
        assert "Total documents: 42" in result.output
        assert "Collection: documents" in result.output
        assert "Storage: ./chroma_db" in result.output

    @patch("doc_indexer.cli.DocumentIndexer")
    def test_stats_empty_index(self, mock_indexer_class, runner):
        """Test stats command with empty index."""
        mock_indexer = Mock()
        mock_indexer.get_stats.return_value = {
            "total_documents": 0,
            "collection_name": "documents",
            "persist_directory": "./chroma_db",
        }
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(main, ["stats"])

        assert result.exit_code == 0
        assert "Total documents: 0" in result.output
        assert "Index is empty" in result.output
