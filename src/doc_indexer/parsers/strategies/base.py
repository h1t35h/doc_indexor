"""
Base strategy for document parsing.
"""

from typing import List, Protocol

from doc_indexer.parsers.base import PageContent


class ParsingStrategy(Protocol):
    """Protocol for parsing strategies."""

    async def process_pages(self, pages: List[PageContent]) -> str:
        """
        Process extracted pages and return combined content.

        Args:
            pages: List of page content objects

        Returns:
            Combined content as string
        """
        ...
