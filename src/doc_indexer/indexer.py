"""
Main document indexer that coordinates parsing and vector storage.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .models import SearchResult
from .parsers import DocumentParser
from .vector_store import VectorStore


class DocumentIndexer:
    """Main document indexer class."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize document indexer."""
        self.parser = DocumentParser()
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.persist_directory = persist_directory

    def index_file(self, file_path: Path) -> bool:
        """Index a single file."""
        try:
            if not self.parser.is_supported(file_path):
                return False

            document = self.parser.parse(file_path)
            document.metadata.indexed_at = datetime.now()

            self.vector_store.add_document(document)
            return True

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return False

    def index_directory(self, directory_path: Path) -> int:
        """Index all supported documents in a directory."""
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        # Find all supported files
        supported_extensions = list(self.parser.parsers.keys())
        files_to_index: List[Path] = []

        for extension in supported_extensions:
            files_to_index.extend(directory_path.rglob(f"*{extension}"))

        # Remove duplicates and sort
        files_to_index = sorted(set(files_to_index))

        if not files_to_index:
            print(f"No supported documents found in {directory_path}")
            return 0

        # Index files with progress bar
        indexed_count = 0
        documents_batch = []

        with tqdm(total=len(files_to_index), desc="Indexing documents") as pbar:
            for file_path in files_to_index:
                try:
                    document = self.parser.parse(file_path)
                    document.metadata.indexed_at = datetime.now()
                    documents_batch.append(document)

                    # Batch add documents for better performance
                    if len(documents_batch) >= 10:
                        self.vector_store.add_documents(documents_batch)
                        indexed_count += len(documents_batch)
                        documents_batch = []

                    pbar.update(1)
                    pbar.set_postfix({"indexed": indexed_count})

                except Exception as e:
                    print(f"\nError indexing {file_path}: {e}")
                    pbar.update(1)

        # Add remaining documents
        if documents_batch:
            self.vector_store.add_documents(documents_batch)
            indexed_count += len(documents_batch)

        return indexed_count

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Search indexed documents."""
        return self.vector_store.search(query, n_results=n_results)

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.delete_collection()

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "total_documents": self.vector_store.get_document_count(),
            "collection_name": "documents",
            "persist_directory": self.persist_directory,
        }
