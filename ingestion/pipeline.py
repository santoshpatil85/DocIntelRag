"""
Ingestion pipeline orchestrator.

Coordinates PDF loading, layout extraction, and initial processing.
"""

from pathlib import Path
from typing import Optional

from ingestion.pdf_loader import PDFLoader
from ingestion.document_intelligence_extractor import DocumentIntelligenceExtractor
from processing.chunking import HierarchicalChunker
from utils.logger import setup_logger
from models.document import Document

logger = setup_logger(__name__)


class PDFIngestionPipeline:
    """
    End-to-end pipeline for PDF ingestion and processing.

    Coordinates all ingestion steps from PDF loading to chunking.
    """

    def __init__(self):
        """Initialize pipeline components."""
        self.pdf_loader = PDFLoader()
        self.doc_intelligence = DocumentIntelligenceExtractor()
        self.chunker = HierarchicalChunker()

    async def process_pdf(
        self,
        pdf_path: str,
        document_id: str,
    ) -> tuple[Optional[Document], list]:
        """
        Process a PDF file through the complete ingestion pipeline.

        Args:
            pdf_path: Path to the PDF file.
            document_id: Unique identifier for the document.

        Returns:
            Tuple of (Document, chunks) or (None, []) on failure.
        """
        try:
            logger.info(f"Starting PDF ingestion for {pdf_path}")

            # Step 1: Load PDF and extract basic content
            pages = self.pdf_loader.extract_pages_sync(pdf_path)

            if not pages:
                logger.warning(f"No pages extracted from {pdf_path}")
                return None, []

            # Step 2: Use Document Intelligence for advanced extraction
            result = await self.doc_intelligence.analyze_document(
                pdf_path,
                document_id=document_id,
            )

            if result is None:
                logger.warning("Document Intelligence analysis failed")
                return None, []

            # Step 3: Create enriched document
            document = await self.doc_intelligence.create_document_from_result(
                result,
                filename=Path(pdf_path).name,
                document_id=document_id,
                pages=pages,
            )

            # Step 4: Chunk the document
            chunks = self.chunker.chunk_document(document)

            logger.info(
                f"PDF ingestion complete: {len(pages)} pages, "
                f"{len(chunks)} chunks"
            )

            return document, chunks

        except Exception as e:
            logger.error(f"PDF ingestion failed: {str(e)}")
            return None, []
