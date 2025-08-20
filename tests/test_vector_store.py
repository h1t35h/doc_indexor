"""
Tests for ChromaDB vector store integration.
"""

from unittest.mock import Mock, patch

import pytest
from doc_indexer.models import Document, DocumentMetadata
from doc_indexer.vector_store import VectorStore


class TestVectorStore:
    """Tests for ChromaDB vector store."""

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client."""
        with patch("doc_indexer.vector_store.chromadb.PersistentClient") as mock:
            yield mock

    @pytest.fixture
    def vector_store(self, mock_chroma_client):
        """Create a VectorStore instance with mocked client."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = VectorStore(persist_directory="./test_chroma_db")
        store.collection = mock_collection
        return store

    def test_vector_store_initialization(self, mock_chroma_client):
        """Test vector store initialization."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = VectorStore(persist_directory="./test_db")

        assert store is not None
        assert store.collection is not None
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="documents"
        )

    def test_add_document(self, vector_store):
        """Test adding a document to the vector store."""
        document = Document(
            content="Test document content",
            metadata=DocumentMetadata(
                filename="test.pdf", file_type="pdf", file_path="/path/to/test.pdf"
            ),
        )

        vector_store.add_document(document)

        vector_store.collection.add.assert_called_once()
        call_args = vector_store.collection.add.call_args

        assert call_args.kwargs["documents"] == ["Test document content"]
        assert "ids" in call_args.kwargs
        assert "metadatas" in call_args.kwargs
        assert call_args.kwargs["metadatas"][0]["filename"] == "test.pdf"

    def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents."""
        documents = [
            Document(
                content=f"Document {i} content",
                metadata=DocumentMetadata(
                    filename=f"doc{i}.pdf",
                    file_type="pdf",
                    file_path=f"/path/to/doc{i}.pdf",
                ),
            )
            for i in range(3)
        ]

        vector_store.add_documents(documents)

        vector_store.collection.add.assert_called_once()
        call_args = vector_store.collection.add.call_args

        assert len(call_args.kwargs["documents"]) == 3
        assert len(call_args.kwargs["ids"]) == 3
        assert len(call_args.kwargs["metadatas"]) == 3

    def test_search_documents(self, vector_store):
        """Test searching documents."""
        mock_results = {
            "documents": [["Result 1 content", "Result 2 content"]],
            "metadatas": [
                [
                    {"filename": "doc1.pdf", "file_type": "pdf"},
                    {"filename": "doc2.pdf", "file_type": "pdf"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        vector_store.collection.query.return_value = mock_results

        results = vector_store.search("test query", n_results=2)

        assert len(results) == 2
        assert results[0].content == "Result 1 content"
        assert results[0].metadata.filename == "doc1.pdf"
        assert results[0].score == 0.1

        vector_store.collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=2
        )

    def test_delete_collection(self, vector_store, mock_chroma_client):
        """Test deleting the collection."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance

        vector_store.client = mock_client_instance
        vector_store.delete_collection()

        mock_client_instance.delete_collection.assert_called_once_with(name="documents")

    def test_get_document_count(self, vector_store):
        """Test getting document count."""
        vector_store.collection.count.return_value = 42

        count = vector_store.get_document_count()

        assert count == 42
        vector_store.collection.count.assert_called_once()

    def test_search_with_empty_query(self, vector_store):
        """Test that empty query raises an error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            vector_store.search("")

    def test_search_with_invalid_n_results(self, vector_store):
        """Test that invalid n_results raises an error."""
        with pytest.raises(ValueError, match="n_results must be positive"):
            vector_store.search("test query", n_results=0)

        with pytest.raises(ValueError, match="n_results must be positive"):
            vector_store.search("test query", n_results=-1)
