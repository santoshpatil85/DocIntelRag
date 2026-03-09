"""
Azure Document Intelligence integration for advanced layout and OCR extraction.

Uses the Document Intelligence "prebuilt-layout" model to extract document
structure, tables, paragraphs, and other layout elements.
"""

import asyncio
from typing import Dict, List, Any, Optional
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from config.settings import settings
from utils.logger import setup_logger
from models.document import Document, DocumentPage, BoundingBox

logger = setup_logger(__name__)


class DocumentIntelligenceExtractor:
    """
    Extracts document structure using Azure Document Intelligence.

    Handles OCR, table detection, layout analysis, and other advanced
    document understanding features.
    """

    def __init__(self):
        """Initialize the Document Intelligence client."""
        try:
            self.client = DocumentIntelligenceClient(
                endpoint=settings.azure.document_intelligence_endpoint,
                credential=AzureKeyCredential(
                    settings.azure.document_intelligence_key
                ),
            )
            logger.info("Initialized Document Intelligence client")
        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence: {str(e)}")
            raise

    async def analyze_document(
        self, pdf_path: str, document_id: str
    ) -> Optional[AnalyzeResult]:
        """
        Analyze a PDF document using Document Intelligence.

        Args:
            pdf_path: Path to the PDF file.
            document_id: Unique document identifier.

        Returns:
            AnalyzeResult object containing layout and content analysis.

        Raises:
            RuntimeError: If analysis fails.
        """
        try:
            logger.info(f"Analyzing document with Document Intelligence: {pdf_path}")

            # Read file as bytes
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            # Call the prebuilt-layout model
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                analyze_request=pdf_data,
                content_type="application/octet-stream",
            )

            # Wait for analysis to complete
            result: AnalyzeResult = poller.result()

            logger.info(
                f"Document analysis completed: {len(result.pages)} pages analyzed"
            )
            return result

        except Exception as e:
            logger.error(f"Document Intelligence analysis failed: {str(e)}")
            raise RuntimeError(f"Could not analyze document: {str(e)}")

    def extract_layout_elements(
        self, result: AnalyzeResult
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract layout elements from analysis result.

        Args:
            result: AnalyzeResult from Document Intelligence.

        Returns:
            Dictionary mapping page numbers to layout elements.
        """
        layout_elements = {}

        try:
            for page_idx, page in enumerate(result.pages):
                page_num = page_idx + 1
                elements = []

                # Extract paragraphs
                for para in getattr(page, "paragraphs", []):
                    bbox = self._extract_bounding_box(para.bounding_regions)
                    elements.append(
                        {
                            "type": "paragraph",
                            "content": para.content,
                            "role": getattr(para, "role", None),
                            "bounding_box": bbox,
                            "confidence": getattr(para, "confidence", None),
                        }
                    )

                # Extract tables
                for table_idx, table in enumerate(getattr(result, "tables", [])):
                    if getattr(table, "bounding_regions", None):
                        for bbox_region in table.bounding_regions:
                            if bbox_region.page_number == page_num:
                                bbox = self._extract_bounding_box([bbox_region])
                                elements.append(
                                    {
                                        "type": "table",
                                        "table_id": table_idx,
                                        "bounding_box": bbox,
                                    }
                                )

                # Extract page metadata
                elements.append(
                    {
                        "type": "page_metadata",
                        "width": page.width,
                        "height": page.height,
                        "unit": page.unit,
                    }
                )

                layout_elements[page_num] = elements

            logger.info(
                f"Extracted layout elements from {len(layout_elements)} pages"
            )
            return layout_elements

        except Exception as e:
            logger.error(f"Failed to extract layout elements: {str(e)}")
            return {}

    def extract_tables(self, result: AnalyzeResult) -> Dict[int, List[Dict]]:
        """
        Extract and structure tables from analysis result.

        Args:
            result: AnalyzeResult from Document Intelligence.

        Returns:
            Dictionary mapping page numbers to structured tables.
        """
        tables_by_page = {}

        try:
            for table in getattr(result, "tables", []):
                # Find page number
                page_num = 1
                if hasattr(table, "bounding_regions") and table.bounding_regions:
                    page_num = table.bounding_regions[0].page_number

                if page_num not in tables_by_page:
                    tables_by_page[page_num] = []

                # Extract table structure
                rows = table.row_count
                cols = table.column_count
                cells = table.cells

                # Build table grid
                table_data = []
                for row in range(rows):
                    table_data.append([None] * cols)

                for cell in cells:
                    content = cell.content
                    row_idx = cell.row_index
                    col_idx = cell.column_index
                    row_span = getattr(cell, "row_span", 1)
                    col_span = getattr(cell, "column_span", 1)

                    # Fill cell and spans
                    for r in range(row_idx, min(row_idx + row_span, rows)):
                        for c in range(col_idx, min(col_idx + col_span, cols)):
                            table_data[r][c] = content

                tables_by_page[page_num].append(
                    {
                        "rows": rows,
                        "columns": cols,
                        "data": table_data,
                        "raw_content": [cell.content for cell in cells],
                    }
                )

            logger.info(f"Extracted tables from {len(tables_by_page)} pages")
            return tables_by_page

        except Exception as e:
            logger.error(f"Failed to extract tables: {str(e)}")
            return {}

    def _extract_bounding_box(
        self, bounding_regions: List[Any]
    ) -> Optional[BoundingBox]:
        """
        Extract bounding box from layout regions.

        Args:
            bounding_regions: List of bounding region objects.

        Returns:
            BoundingBox object or None if no regions found.
        """
        if not bounding_regions:
            return None

        try:
            region = bounding_regions[0]
            polygon = region.polygon

            if not polygon:
                return None

            # Find bounding box from polygon points
            xs = [p.x for p in polygon]
            ys = [p.y for p in polygon]

            return BoundingBox(
                x=min(xs),
                y=min(ys),
                width=max(xs) - min(xs),
                height=max(ys) - min(ys),
            )
        except Exception as e:
            logger.warning(f"Failed to extract bounding box: {str(e)}")
            return None

    async def create_document_from_result(
        self,
        result: AnalyzeResult,
        filename: str,
        document_id: str,
        pages: List[DocumentPage],
    ) -> Document:
        """
        Create a Document object enriched with Document Intelligence results.

        Args:
            result: AnalyzeResult from Document Intelligence.
            filename: Original filename.
            document_id: Unique document identifier.
            pages: List of DocumentPage objects from basic PDF extraction.

        Returns:
            Enriched Document object.
        """
        try:
            # Extract layout and table information
            layout_elements = self.extract_layout_elements(result)
            tables = self.extract_tables(result)

            # Enrich pages with layout information
            for page in pages:
                page_num = page.page_number
                if page_num in layout_elements:
                    page.layout_elements = layout_elements[page_num]
                if page_num in tables:
                    page.tables = tables[page_num]

            # Create document
            document = Document(
                document_id=document_id,
                filename=filename,
                pages=pages,
                metadata={
                    "source": "Document Intelligence",
                    "total_pages": len(result.pages),
                },
            )

            logger.info(
                f"Created enriched document: {document_id} ({document.total_pages} pages)"
            )
            return document

        except Exception as e:
            logger.error(f"Failed to create document from result: {str(e)}")
            raise
