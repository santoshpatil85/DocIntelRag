"""
Hierarchical document chunking with multimodal support.

Creates balanced chunks from text, tables, and charts with proper metadata.
"""

import uuid
from typing import List, Optional, Dict, Any

from config.settings import settings
from utils.logger import setup_logger
from models.document import Chunk, ChunkType, Document, BoundingBox
from processing.text_processor import TextProcessor

logger = setup_logger(__name__)


class ChunkingStrategy:
    """Base class for chunking strategies."""

    def chunk(self, content: str) -> List[str]:
        """
        Chunk content into logical units.

        Args:
            content: Content to chunk.

        Returns:
            List of chunks.
        """
        raise NotImplementedError


class SlidingWindowChunking(ChunkingStrategy):
    """Chunks text using sliding window approach."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        """
        Initialize sliding window chunker.

        Args:
            chunk_size: Target size for each chunk (words).
            overlap: Overlap between consecutive chunks (words).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: str) -> List[str]:
        """
        Chunk text using sliding window.

        Args:
            content: Content to chunk.

        Returns:
            List of text chunks.
        """
        if not content:
            return []

        # Split into sentences first
        sentences = TextProcessor.split_sentences(content)

        if not sentences:
            return [content] if content.strip() else []

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_size = len(words)

            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Keep last overlap% of sentences for next chunk
                overlap_sentences = int(len(current_chunk) * (self.overlap / self.chunk_size))
                overlap_sentences = max(1, overlap_sentences)

                current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.debug(f"Created {len(chunks)} chunks from content")
        return chunks


class HierarchicalChunker:
    """
    Creates multimodal chunks from documents with hierarchical structure.

    Handles text, tables, and charts with appropriate metadata.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize hierarchical chunker.

        Args:
            chunk_size: Size for text chunks.
            chunk_overlap: Overlap for text chunks.
        """
        self.chunk_size = chunk_size or settings.processing.chunk_size
        self.chunk_overlap = chunk_overlap or settings.processing.chunk_overlap
        self.text_chunker = SlidingWindowChunking(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

    def chunk_document(
        self,
        document: Document,
    ) -> List[Chunk]:
        """
        Create chunks from a complete document.

        Args:
            document: Document to chunk.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        chunk_counter = 0

        try:
            for page in document.pages:
                page_num = page.page_number

                # Chunk text content
                if page.text:
                    text_chunks = self.text_chunker.chunk(page.text)
                    for text_content in text_chunks:
                        chunk_counter += 1
                        chunk = Chunk(
                            chunk_id=f"{document.document_id}_text_{chunk_counter}",
                            document_id=document.document_id,
                            content=text_content,
                            chunk_type=ChunkType.TEXT,
                            page_number=page_num,
                            metadata={
                                "document_name": document.filename,
                                "content_type": "paragraph",
                            },
                        )
                        chunks.append(chunk)

                # Create chunks for tables
                if page.tables:
                    for table_idx, table in enumerate(page.tables):
                        chunk_counter += 1
                        # Create both raw table and summary
                        table_content = self._table_to_text(table)
                        chunk = Chunk(
                            chunk_id=f"{document.document_id}_table_{chunk_counter}",
                            document_id=document.document_id,
                            content=table_content,
                            chunk_type=ChunkType.TABLE,
                            page_number=page_num,
                            metadata={
                                "document_name": document.filename,
                                "table_index": table_idx,
                                "rows": table.get("rows", 0),
                                "columns": table.get("columns", 0),
                            },
                        )
                        chunks.append(chunk)

            logger.info(
                f"Created {len(chunks)} chunks from document {document.document_id}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk document: {str(e)}")
            return chunks

    def add_chart_chunks(
        self,
        document: Document,
        chart_analyses: Dict[str, str],
    ) -> List[Chunk]:
        """
        Add chart analysis chunks to existing chunks.

        Args:
            document: Document object.
            chart_analyses: Dictionary mapping chart IDs to analysis text.

        Returns:
            List of new chart chunks.
        """
        chunks = []
        chunk_counter = 0

        try:
            for chart_id, analysis in chart_analyses.items():
                chunk_counter += 1
                chunk = Chunk(
                    chunk_id=f"{document.document_id}_chart_{chunk_counter}",
                    document_id=document.document_id,
                    content=analysis,
                    chunk_type=ChunkType.CHART,
                    page_number=1,  # Default to first page
                    metadata={
                        "document_name": document.filename,
                        "chart_id": chart_id,
                        "source": "GPT-4o Vision Analysis",
                    },
                )
                chunks.append(chunk)

            logger.info(f"Created {len(chunks)} chart chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to create chart chunks: {str(e)}")
            return []

    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """
        Convert table structure to readable text.

        Args:
            table: Table dictionary from document.

        Returns:
            Text representation of the table.
        """
        try:
            parts = []

            # Add summary if available
            if "summary" in table:
                parts.append(f"Table Summary: {table['summary']}")

            # Format table data
            rows = table.get("rows", 0)
            cols = table.get("columns", 0)
            data = table.get("data", [])

            if data:
                # Add header row
                if len(data) > 0:
                    headers = data[0]
                    parts.append("Headers: " + " | ".join(str(h) for h in headers))

                    # Add sample data rows
                    sample_rows = data[1:4]  # Take first 3 data rows
                    for row in sample_rows:
                        parts.append(" | ".join(str(cell) for cell in row))

                if len(data) > 4:
                    parts.append(f"... ({len(data) - 4} more rows)")

            return "\n".join(parts) if parts else "Table - No data extracted"

        except Exception as e:
            logger.warning(f"Failed to convert table to text: {str(e)}")
            return "Table - Unable to extract content"
