"""
Base classes and protocols for document parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from PIL import Image

from doc_indexer.models import Document, DocumentMetadata


@dataclass
class PageContent:
    """Represents content extracted from a single page."""

    page_number: int
    text: Optional[str] = None
    image: Optional[Image.Image] = None
    tables: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize tables list if not provided."""
        if self.tables is None:
            self.tables = []


class FileParser(Protocol):
    """Protocol for file parsers."""

    async def parse(self, file_path: Path) -> Document:
        """Parse a file and return a Document."""
        ...


class ParsingStrategy(Protocol):
    """Protocol for parsing strategies."""

    async def process_pages(self, pages: List[PageContent]) -> str:
        """Process extracted pages and return combined content."""
        ...


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, parsing_strategy: ParsingStrategy) -> None:
        """Initialize parser with a parsing strategy."""
        self.strategy = parsing_strategy

    @abstractmethod
    async def extract_pages(self, file_path: Path) -> List[PageContent]:
        """Extract pages from document file."""
        ...

    async def parse(self, file_path: Path) -> Document:
        """Parse document using the configured strategy."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        pages = await self.extract_pages(file_path)
        content = await self.strategy.process_pages(pages)

        metadata = DocumentMetadata(
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            file_path=str(file_path.absolute()),
            file_size=file_path.stat().st_size,
        )

        return Document(content=content, metadata=metadata)
