"""
Request and response schemas for the API.

Uses Pydantic for validation and serialization.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadPDFRequest(BaseModel):
    """Request model for PDF upload."""

    filename: str = Field(..., description="Name of the PDF file")


class UploadPDFResponse(BaseModel):
    """Response model for PDF upload."""

    success: bool
    document_id: str
    filename: str
    total_pages: int
    chunks_created: int
    message: str


class AskQuestionRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., description="The question to ask")
    document_id: Optional[str] = Field(
        None, description="Specific document to query (optional)"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )


class SourceReference(BaseModel):
    """Reference to a source in the document."""

    document: str
    page: int
    type: str


class AskQuestionResponse(BaseModel):
    """Response model for question answering."""

    question: str
    answer: str
    sources: List[SourceReference]
    chunks_used: int


class VectorStoreStats(BaseModel):
    """Statistics about the vector store."""

    total_chunks: int
    embedding_dimension: int
    metadata_stored: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_store_stats: VectorStoreStats


class DeleteDocumentRequest(BaseModel):
    """Request model for document deletion."""

    document_id: str = Field(..., description="ID of document to delete")


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""

    success: bool
    document_id: str
    chunks_deleted: int
    message: str
