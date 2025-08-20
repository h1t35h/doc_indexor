"""
PDF document parser with LLM support.
"""

from pathlib import Path
from typing import List, Optional

from PIL import Image
from pypdf import PdfReader

try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from doc_indexer.parsers.base import BaseParser, PageContent


class PDFParser(BaseParser):
    """Parser for PDF documents with image extraction support."""

    def __init__(self, *args, extract_images: bool = True, **kwargs) -> None:
        """
        Initialize PDF parser.

        Args:
            extract_images: Whether to extract images from PDF pages
            *args, **kwargs: Arguments passed to BaseParser
        """
        super().__init__(*args, **kwargs)
        self.extract_images = extract_images and PDF2IMAGE_AVAILABLE

    async def extract_pages(self, file_path: Path) -> List[PageContent]:
        """Extract pages from PDF document."""
        pages = []

        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text() or ""
                tables = self._extract_tables_from_page(page)

                page_content = PageContent(
                    page_number=page_num, text=text, image=None, tables=tables
                )
                pages.append(page_content)

        if self.extract_images:
            try:
                images = self._extract_images_from_pdf(file_path)
                for i, image in enumerate(images):
                    if i < len(pages):
                        pages[i].image = image
            except Exception as e:
                print(f"Warning: Could not extract images from PDF: {e}")

        return pages

    def _extract_images_from_pdf(self, file_path: Path) -> List[Optional[Image.Image]]:
        """Extract images from PDF pages."""
        if not PDF2IMAGE_AVAILABLE:
            return []

        try:
            images = convert_from_path(str(file_path), dpi=150, fmt="png")

            resized_images = []
            for image in images:
                if image.width > 1920 or image.height > 1080:
                    image.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                resized_images.append(image)

            return resized_images
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            return []

    def _extract_tables_from_page(self, page) -> List[dict]:
        """Extract tables from a PDF page (placeholder implementation)."""
        return []
