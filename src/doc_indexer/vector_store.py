"""
ChromaDB vector store for document indexing and search.
"""

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

from .models import Document, DocumentMetadata, SearchResult


class VectorStore:
    """Vector store using ChromaDB."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize vector store with ChromaDB."""
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(name="documents")

    def add_document(self, document: Document) -> None:
        """Add a single document to the vector store."""
        self.add_documents([document])

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the vector store."""
        if not documents:
            return

        ids = [doc.doc_id or f"doc_{i}" for i, doc in enumerate(documents)]
        contents = [doc.content for doc in documents]
        metadatas: List[Dict[str, Any]] = [
            {k: v for k, v in doc.metadata.to_dict().items()}  # type: ignore
            for doc in documents
        ]

        self.collection.add(documents=contents, ids=ids, metadatas=metadatas)  # type: ignore

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Search for documents similar to the query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if n_results <= 0:
            raise ValueError("n_results must be positive")

        results = self.collection.query(query_texts=[query], n_results=n_results)

        search_results = []

        if results["documents"] and results["documents"][0]:
            for i, content in enumerate(results["documents"][0]):
                metadata_dict = (
                    results["metadatas"][0][i] if results["metadatas"] else {}
                )

                metadata = DocumentMetadata(
                    filename=str(metadata_dict.get("filename", "unknown")),
                    file_type=str(metadata_dict.get("file_type", "unknown")),
                    file_path=str(metadata_dict.get("file_path", "")),
                )

                score = results["distances"][0][i] if results["distances"] else 0.0

                search_results.append(
                    SearchResult(content=content, metadata=metadata, score=score)
                )

        return search_results

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(name="documents")

    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        return self.collection.count()
