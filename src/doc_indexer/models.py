"""
Data models for document indexer.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class DocumentMetadata:
    """Metadata for a document."""

    filename: str
    file_type: str
    file_path: str
    indexed_at: Optional[datetime] = None
    file_size: Optional[int] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert metadata to dictionary."""
        data: Dict[str, str] = {
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
        }
        if self.indexed_at:
            data["indexed_at"] = self.indexed_at.isoformat()
        if self.file_size:
            data["file_size"] = str(self.file_size)
        return data


@dataclass
class Document:
    """Represents a parsed document."""

    content: str
    metadata: DocumentMetadata
    doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate document ID if not provided."""
        if not self.doc_id:
            import hashlib

            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.doc_id = f"{self.metadata.filename}_{content_hash}"


@dataclass
class SearchResult:
    """Represents a search result."""

    content: str
    metadata: DocumentMetadata
    score: float
    doc_id: Optional[str] = None
