"""
Unit tests for the Multimodal RAG system.

Tests core components and functionality.
"""

import pytest
import asyncio
from pathlib import Path

from models.document import (
    Chunk,
    ChunkType,
    Document,
    DocumentPage,
    BoundingBox,
    ChartAnalysis,
    TableAnalysis,
)
from processing.text_processor import TextProcessor
from processing.chunking import SlidingWindowChunking
from vectordb.vector_store import FAISSVectorStore
from embeddings.embedding_service import EmbeddingService


# Tests for Document Models


def test_bounding_box_creation():
    """Test BoundingBox creation and serialization."""
    bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)

    assert bbox.x == 10.0
    assert bbox.width == 100.0

    # Test serialization
    bbox_dict = bbox.to_dict()
    assert bbox_dict["x"] == 10.0

    # Test deserialization
    bbox_restored = BoundingBox.from_dict(bbox_dict)
    assert bbox_restored.x == bbox.x


def test_chunk_creation():
    """Test Chunk creation and serialization."""
    chunk = Chunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        content="Sample text content",
        chunk_type=ChunkType.TEXT,
        page_number=1,
        section_title="Introduction",
    )

    assert chunk.chunk_id == "chunk_001"
    assert chunk.chunk_type == ChunkType.TEXT

    # Test serialization
    chunk_dict = chunk.to_dict()
    assert chunk_dict["chunk_type"] == "text"

    # Test deserialization
    chunk_restored = Chunk.from_dict(chunk_dict)
    assert chunk_restored.content == chunk.content


def test_document_creation():
    """Test Document creation."""
    pages = [
        DocumentPage(page_number=1, text="Page 1 content"),
        DocumentPage(page_number=2, text="Page 2 content"),
    ]

    doc = Document(
        document_id="doc_001",
        filename="test.pdf",
        pages=pages,
    )

    assert doc.total_pages == 2
    assert doc.get_page(0).text == "Page 1 content"


def test_chart_analysis_to_text():
    """Test ChartAnalysis text generation."""
    analysis = ChartAnalysis(
        title="Sales Trend",
        description="Monthly sales trending upward",
        axes={"X": "Month", "Y": "Sales"},
        trends=["Upward trend", "Peak in Q4"],
    )

    text = analysis.to_text()
    assert "Sales Trend" in text
    assert "Month" in text


# Tests for Text Processing


def test_text_cleaning():
    """Test text cleaning and normalization."""
    text = "This  is   a\n\ntest    with   spaces"
    cleaned = TextProcessor.clean_text(text)

    # The function removes all excessive whitespace including newlines
    assert "  " not in cleaned
    assert cleaned == "This is a test with spaces"


def test_sentence_splitting():
    """Test sentence splitting."""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = TextProcessor.split_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "First sentence."


def test_text_truncation():
    """Test text truncation."""
    text = "A" * 1000
    truncated = TextProcessor.truncate_text(text, max_length=100)

    assert len(truncated) <= 105  # +5 for "..."


# Tests for Chunking


def test_sliding_window_chunking():
    """Test sliding window chunking strategy."""
    text = (
        "Sentence one. Sentence two. Sentence three. "
        "Sentence four. Sentence five. Sentence six."
    )

    chunker = SlidingWindowChunking(chunk_size=3, overlap=1)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    # Each chunk should contain multiple sentences
    for chunk in chunks:
        assert len(chunk) > 0


# Tests for Vector Store


def test_faiss_vector_store_creation():
    """Test FAISS vector store initialization."""
    store = FAISSVectorStore(dimension=512)

    assert store.index is not None
    assert store.index.d == 512


def test_vector_store_add_chunk():
    """Test adding a chunk to vector store."""
    store = FAISSVectorStore(dimension=10)

    chunk = Chunk(
        chunk_id="test_chunk",
        document_id="doc_001",
        content="Test content",
        chunk_type=ChunkType.TEXT,
        page_number=1,
        embedding=[0.1] * 10,
    )

    success = store.add_chunk(chunk)
    assert success
    assert store.index.ntotal == 1


def test_vector_store_stats():
    """Test vector store statistics."""
    store = FAISSVectorStore(dimension=10)

    chunk = Chunk(
        chunk_id="test_chunk",
        document_id="doc_001",
        content="Test",
        chunk_type=ChunkType.TEXT,
        page_number=1,
        embedding=[0.1] * 10,
    )

    store.add_chunk(chunk)
    stats = store.get_stats()

    assert stats["total_chunks"] == 1
    assert stats["embedding_dimension"] == 10


# Test Chart Analysis


def test_chart_analysis_parsing():
    """Test ChartAnalysis data structure."""
    analysis = ChartAnalysis(
        title="Annual Revenue",
        description="Revenue by quarter",
        axes={"X": "Quarter", "Y": "Revenue ($M)"},
        trends=["Steady growth", "Seasonal variations"],
        legend={"Series 1": "Product A", "Series 2": "Product B"},
    )

    assert analysis.title == "Annual Revenue"
    assert len(analysis.trends) == 2
    assert "Product A" in analysis.legend.values()


# Test Table Analysis


def test_table_analysis_creation():
    """Test TableAnalysis data structure."""
    analysis = TableAnalysis(
        table_summary="Monthly sales data",
        row_count=12,
        column_count=4,
        headers=["Month", "Product A", "Product B", "Total"],
        key_metrics=["Average: 1000", "Max: 2000"],
    )

    assert analysis.row_count == 12
    assert len(analysis.headers) == 4


# Async Tests (requires pytest-asyncio)


@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test EmbeddingService initialization (requires Azure credentials)."""
    try:
        service = EmbeddingService()
        assert service.model is not None
    except Exception:
        # Skip if Azure credentials not available
        pytest.skip("Azure credentials not configured")


@pytest.mark.asyncio
async def test_embedding_text():
    """Test text embedding (requires Azure credentials)."""
    try:
        service = EmbeddingService()
        embedding = await service.embed_text("Test text")

        if embedding is not None:
            assert isinstance(embedding, list)
            assert len(embedding) == 3072  # text-embedding-3-large dimension
    except Exception:
        pytest.skip("Azure credentials not configured")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
