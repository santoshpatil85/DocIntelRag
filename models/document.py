"""
Data models for document representation in the RAG system.

Defines structured representations for documents, chunks, and
metadata throughout the processing pipeline.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class ChunkType(str, Enum):
    """Types of document chunks."""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"


@dataclass
class BoundingBox:
    """Represents a 2D bounding box region."""

    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BoundingBox":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Chunk:
    """
    Represents a multimodal chunk extracted from a document.

    A chunk is a logical unit of content (text, table, chart) with
    associated metadata and embeddings.
    """

    chunk_id: str
    document_id: str
    content: str
    chunk_type: ChunkType
    page_number: int
    section_title: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Initialize default metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary representation.

        Returns:
            Dictionary with all chunk data.
        """
        data = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "metadata": self.metadata,
        }
        if self.bounding_box:
            data["bounding_box"] = self.bounding_box.to_dict()
        return data

    def to_json(self) -> str:
        """Serialize chunk to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary representation."""
        chunk_type = ChunkType(data.pop("chunk_type"))
        bounding_box = None
        if "bounding_box" in data and data["bounding_box"]:
            bounding_box = BoundingBox.from_dict(data.pop("bounding_box"))
        embedding = data.pop("embedding", None)

        chunk = cls(
            chunk_type=chunk_type,
            bounding_box=bounding_box,
            **data,
        )
        chunk.embedding = embedding
        return chunk


@dataclass
class DocumentPage:
    """Represents a single page from a document."""

    page_number: int
    text: str
    images: List[bytes] = None
    tables: List[Dict[str, Any]] = None
    layout_elements: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default collections if not provided."""
        if self.images is None:
            self.images = []
        if self.tables is None:
            self.tables = []
        if self.layout_elements is None:
            self.layout_elements = []


@dataclass
class Document:
    """
    Represents a complete document with metadata and pages.

    This is the result of PDF ingestion and layout extraction.
    """

    document_id: str
    filename: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any] = None
    total_pages: int = 0

    def __post_init__(self):
        """Initialize metadata and page count."""
        if self.metadata is None:
            self.metadata = {}
        self.total_pages = len(self.pages)

    def get_page(self, page_number: int) -> Optional[DocumentPage]:
        """
        Retrieve a specific page by number.

        Args:
            page_number: 1-indexed page number.

        Returns:
            DocumentPage or None if not found.
        """
        if 0 <= page_number < len(self.pages):
            return self.pages[page_number]
        return None


@dataclass
class ChartAnalysis:
    """Structured analysis of a chart or visual element."""

    title: str
    description: str
    axes: Dict[str, str]
    trends: List[str]
    legend: Optional[Dict[str, str]] = None
    data_insights: Optional[List[str]] = None

    def to_text(self) -> str:
        """Convert analysis to natural language text."""
        parts = [
            f"Chart: {self.title}",
            f"Description: {self.description}",
        ]

        if self.axes:
            axes_str = ", ".join(f"{k}: {v}" for k, v in self.axes.items())
            parts.append(f"Axes: {axes_str}")

        if self.trends:
            parts.append(f"Trends: {', '.join(self.trends)}")

        if self.legend:
            legend_str = ", ".join(f"{k}: {v}" for k, v in self.legend.items())
            parts.append(f"Legend: {legend_str}")

        if self.data_insights:
            parts.append(f"Insights: {', '.join(self.data_insights)}")

        return "\n".join(parts)


@dataclass
class TableAnalysis:
    """Structured analysis of a table."""

    table_summary: str
    row_count: int
    column_count: int
    headers: List[str]
    key_metrics: Optional[List[str]] = None

    def to_text(self) -> str:
        """Convert analysis to natural language text."""
        parts = [
            f"Table Summary: {self.table_summary}",
            f"Dimensions: {self.row_count} rows × {self.column_count} columns",
            f"Headers: {', '.join(self.headers)}",
        ]

        if self.key_metrics:
            parts.append(f"Key Metrics: {', '.join(self.key_metrics)}")

        return "\n".join(parts)
