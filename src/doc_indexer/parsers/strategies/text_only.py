"""
Text-only parsing strategy (traditional extraction).
"""

from typing import List

from doc_indexer.parsers.base import PageContent


class TextOnlyStrategy:
    """Strategy for extracting text without LLM enhancement."""

    async def process_pages(self, pages: List[PageContent]) -> str:
        """Process pages by extracting text and tables only."""
        if not pages:
            return ""

        content_parts = []

        for page in pages:
            page_content = []

            if len(pages) > 1:
                page_content.append(f"\n--- Page {page.page_number} ---\n")

            if page.text and page.text.strip():
                page_content.append(page.text)

            if page.tables:
                for table in page.tables:
                    table_text = self._format_table(table)
                    if table_text:
                        page_content.append(table_text)

            if page_content:
                content_parts.append("\n".join(page_content))

        return "\n\n".join(content_parts)

    def _format_table(self, table: dict) -> str:
        """Format table data as text."""
        if not table:
            return ""

        lines = []

        if "header" in table and table["header"]:
            lines.append(" | ".join(str(h) for h in table["header"]))
            lines.append("-" * 40)

        if "rows" in table and table["rows"]:
            for row in table["rows"]:
                lines.append(" | ".join(str(cell) for cell in row))

        return "\n".join(lines) if lines else ""
