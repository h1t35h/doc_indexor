"""
LLM-enhanced parsing strategy for comprehensive content extraction.
"""

import asyncio
from typing import List

from doc_indexer.parsers.base import PageContent
from doc_indexer.parsers.config import ParserConfig
from doc_indexer.parsers.llm_providers.base import LLMProvider
from doc_indexer.utils.security import PromptSanitizer


class LLMEnhancedStrategy:
    """Strategy for parsing with LLM enhancement."""

    def __init__(self, llm_provider: LLMProvider, config: ParserConfig) -> None:
        """
        Initialize LLM-enhanced strategy.

        Args:
            llm_provider: LLM provider instance
            config: Parser configuration
        """
        self.llm = llm_provider
        self.config = config

    async def process_pages(self, pages: List[PageContent]) -> str:
        """Process pages using LLM for enhanced extraction."""
        if not pages:
            return ""

        results = []
        batch_size = self.config.max_pages_per_batch
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)

        return self._combine_results(results)

    async def _process_batch(self, pages: List[PageContent]) -> List[str]:
        """Process a batch of pages concurrently."""
        tasks = []

        for page in pages:
            if self.config.parsing_mode == "llm_only":
                if page.image:
                    tasks.append(self._process_page_with_llm(page))
                else:
                    tasks.append(self._process_text_only(page))
            elif self.config.parsing_mode == "hybrid":
                tasks.append(self._process_page_hybrid(page))
            else:
                tasks.append(self._process_text_only(page))

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_page_with_llm(self, page: PageContent) -> str:
        """
        Process a page entirely with LLM.

        Args:
            page: Page to process

        Returns:
            Processed content
        """
        try:
            if page.image:
                prompt = self._build_extraction_prompt(page)
                result = await self.llm.analyze_image(page.image, prompt)
                return f"Page {page.page_number}:\n{result}"
            else:
                return f"Page {page.page_number}: [No image content]"
        except Exception as e:
            # Fallback to text extraction on error
            print(f"LLM processing error for page {page.page_number}: {e}")
            return await self._process_text_only(page)

    async def _process_page_hybrid(self, page: PageContent) -> str:
        """
        Process a page using hybrid approach (text + LLM).

        Args:
            page: Page to process

        Returns:
            Processed content
        """
        results = []

        # Process image with LLM if available
        if page.image:
            try:
                prompt = self._build_extraction_prompt(page)
                image_result = await self.llm.analyze_image(page.image, prompt)
                results.append(f"Visual content:\n{image_result}")
            except Exception as e:
                print(f"LLM image processing error: {e}")

        # Enhance text with LLM if available
        if page.text and page.text.strip():
            try:
                # Sanitize text and prompt
                sanitized_text = PromptSanitizer.sanitize_text(page.text)
                text_prompt = PromptSanitizer.sanitize_prompt(
                    "Extract and structure all information including tables, lists, and key points:"
                )
                text_result = await self.llm.analyze_text(sanitized_text, text_prompt)
                results.append(f"Text content:\n{text_result}")
            except Exception:
                # Fallback to original text (sanitized)
                sanitized_text = PromptSanitizer.sanitize_text(page.text)
                results.append(f"Text content:\n{sanitized_text}")

        # Format tables if present
        if page.tables:
            table_text = self._format_tables(page.tables)
            results.append(f"Tables:\n{table_text}")

        if results:
            return f"Page {page.page_number}:\n" + "\n".join(results)
        else:
            return f"Page {page.page_number}: [No content extracted]"

    async def _process_text_only(self, page: PageContent) -> str:
        """
        Process page with text extraction only (fallback).

        Args:
            page: Page to process

        Returns:
            Text content
        """
        parts = []

        if page.text and page.text.strip():
            parts.append(page.text)

        if page.tables:
            parts.append(self._format_tables(page.tables))

        if parts:
            return f"Page {page.page_number}:\n" + "\n".join(parts)
        else:
            return f"Page {page.page_number}: [No text content]"

    def _build_extraction_prompt(self, page: PageContent) -> str:
        """
        Build prompt for LLM extraction.

        Args:
            page: Page being processed

        Returns:
            Extraction prompt
        """
        base_prompt = """Analyze this document page and extract ALL information:

1. **Text Content**: Extract all visible text, maintaining structure and formatting
2. **Tables**: Extract table data in a structured format with headers and rows
3. **Charts/Graphs**: Describe the chart type, axes, data points, and trends
4. **Images**: Describe any images, diagrams, or illustrations
5. **Lists**: Extract bulleted or numbered lists maintaining hierarchy
6. **Headers/Footers**: Extract page numbers, headers, and footers
7. **Special Elements**: Mathematical formulas, code snippets, citations

Preserve the original document structure and context. Be comprehensive and accurate."""

        # Sanitize the prompt to prevent injection
        return PromptSanitizer.sanitize_prompt(base_prompt, max_length=1500)

    def _format_tables(self, tables: List[dict]) -> str:
        """
        Format tables as text.

        Args:
            tables: List of table dictionaries

        Returns:
            Formatted tables
        """
        if not tables:
            return ""

        formatted = []
        for i, table in enumerate(tables, 1):
            lines = [f"Table {i}:"]

            if "header" in table and table["header"]:
                lines.append(" | ".join(str(h) for h in table["header"]))
                lines.append("-" * 40)

            if "rows" in table and table["rows"]:
                for row in table["rows"]:
                    lines.append(" | ".join(str(cell) for cell in row))

            formatted.append("\n".join(lines))

        return "\n\n".join(formatted)

    def _combine_results(self, results: List[str]) -> str:
        """Combine results from multiple pages."""
        valid_results = [
            result for result in results if isinstance(result, str) and result.strip()
        ]

        errors = [r for r in results if isinstance(r, Exception)]
        for error in errors:
            print(f"Page processing error: {error}")

        return "\n\n".join(valid_results)
