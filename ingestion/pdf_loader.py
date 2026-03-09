"""
PDF loading and basic extraction using PyMuPDF.

Handles reading PDF files and extracting text and images.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
import uuid

import fitz  # PyMuPDF

from utils.logger import setup_logger
from models.document import DocumentPage

logger = setup_logger(__name__)


class PDFLoader:
    """Loads and extracts basic content from PDF files."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize the PDF loader.

        Args:
            max_workers: Number of concurrent workers for page extraction.
        """
        self.max_workers = max_workers

    def load_pdf(self, pdf_path: str) -> Tuple[str, fitz.Document]:
        """
        Load a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Tuple of document ID and fitz Document object.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            RuntimeError: If PDF cannot be opened.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            doc_id = str(uuid.uuid4())
            logger.info(f"Loaded PDF: {path.name} ({doc.page_count} pages)")
            return doc_id, doc
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {str(e)}")
            raise RuntimeError(f"Cannot open PDF file: {str(e)}")

    def extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a single page.

        Args:
            page: PyMuPDF page object.

        Returns:
            Extracted text content.
        """
        try:
            text = page.get_text()
            return text if text else ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page: {str(e)}")
            return ""

    def extract_images_from_page(
        self, page: fitz.Page, max_width: int = 1024, max_height: int = 1024
    ) -> List[bytes]:
        """
        Extract images from a single page.

        Args:
            page: PyMuPDF page object.
            max_width: Maximum image width for resizing.
            max_height: Maximum image height for resizing.

        Returns:
            List of image bytes.
        """
        images = []
        try:
            image_list = page.get_images()
            for image_index in image_list:
                try:
                    xref = image_index[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    # Convert to RGB if necessary
                    if pix.n - pix.alpha < 4:  # Gray or RGB
                        pix_rgb = pix
                    else:  # CMYK
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)

                    # Resize if necessary
                    if pix_rgb.width > max_width or pix_rgb.height > max_height:
                        scale = min(
                            max_width / pix_rgb.width,
                            max_height / pix_rgb.height,
                        )
                        pix_rgb = pix_rgb.scale(scale, scale)

                    # Get image bytes in PNG format
                    img_bytes = pix_rgb.tobytes("png")
                    images.append(img_bytes)

                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {image_index}: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.warning(f"Failed to extract images from page: {str(e)}")

        return images

    async def extract_pages_async(
        self, pdf_path: str
    ) -> List[DocumentPage]:
        """
        Extract all pages from PDF asynchronously.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of DocumentPage objects.
        """
        doc_id, doc = self.load_pdf(pdf_path)

        # Extract pages in batches
        pages = []
        tasks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Run page extraction in executor to avoid blocking
            task = asyncio.get_event_loop().run_in_executor(
                None,
                self._extract_page_sync,
                page,
                page_num + 1,
            )
            tasks.append(task)

        # Wait for all tasks in batches
        results = await asyncio.gather(*tasks)
        pages = [p for p in results if p is not None]

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages

    def _extract_page_sync(self, page: fitz.Page, page_num: int) -> DocumentPage:
        """
        Synchronously extract content from a single page.

        Args:
            page: PyMuPDF page object.
            page_num: 1-indexed page number.

        Returns:
            DocumentPage object.
        """
        try:
            text = self.extract_text_from_page(page)
            images = self.extract_images_from_page(page)

            return DocumentPage(
                page_number=page_num,
                text=text,
                images=images,
            )
        except Exception as e:
            logger.error(f"Failed to extract page {page_num}: {str(e)}")
            return None

    def extract_pages_sync(self, pdf_path: str) -> List[DocumentPage]:
        """
        Extract all pages from PDF synchronously.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of DocumentPage objects.
        """
        doc_id, doc = self.load_pdf(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            doc_page = self._extract_page_sync(page, page_num + 1)
            if doc_page:
                pages.append(doc_page)

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages
