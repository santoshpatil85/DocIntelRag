"""
FastAPI routes for the Multimodal RAG system.

Exposes endpoints for PDF ingestion, chunking, embedding, and Q&A.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from config.settings import settings
from utils.logger import setup_logger
from models.document import Document
from ingestion.pdf_loader import PDFLoader
from ingestion.document_intelligence_extractor import DocumentIntelligenceExtractor
from ingestion.image_extractor import ImageExtractor
from processing.chunking import HierarchicalChunker
from processing.table_parser import TableParser
from processing.chart_analyzer import ChartAnalyzer
from embeddings.embedding_service import EmbeddingService
from vectordb.vector_store import FAISSVectorStore
from rag.retriever import LayoutAwareRetriever
from rag.qa_engine import RAGQAEngine
from api.schemas import (
    UploadPDFRequest,
    UploadPDFResponse,
    AskQuestionRequest,
    AskQuestionResponse,
    SourceReference,
    HealthResponse,
    VectorStoreStats,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
)

logger = setup_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Multimodal RAG System",
        description="PDF analysis with multimodal RAG capabilities",
        version="1.0.0",
    )

    # Initialize services
    pdf_loader = PDFLoader()
    doc_intelligence = DocumentIntelligenceExtractor()
    image_extractor = ImageExtractor()
    chunker = HierarchicalChunker()
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore()
    retriever = LayoutAwareRetriever(vector_store, embedding_service)
    qa_engine = RAGQAEngine(retriever)

    # Store services in app state
    app.state.pdf_loader = pdf_loader
    app.state.doc_intelligence = doc_intelligence
    app.state.image_extractor = image_extractor
    app.state.chunker = chunker
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.qa_engine = qa_engine

    @app.get("/health")
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        stats = vector_store.get_stats()
        return HealthResponse(
            status="healthy",
            vector_store_stats=VectorStoreStats(
                total_chunks=stats["total_chunks"],
                embedding_dimension=stats["embedding_dimension"],
                metadata_stored=stats["metadata_stored"],
            ),
        )

    @app.post("/upload_pdf", response_model=UploadPDFResponse)
    async def upload_pdf(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
    ) -> UploadPDFResponse:
        """
        Upload and process a PDF file.

        Ingests PDF, extracts layout, chunks content, and embeds chunks.
        """
        temp_path = None

        try:
            # Save uploaded file
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)

            logger.info(f"Processing PDF: {file.filename}")

            # Extract basic content
            pages = pdf_loader.extract_pages_sync(temp_path)

            # Use Document Intelligence for advanced extraction
            result = await doc_intelligence.analyze_document(
                temp_path, document_id="temp"
            )

            # Create enriched document
            document = await doc_intelligence.create_document_from_result(
                result,
                filename=file.filename,
                document_id="temp",
                pages=pages,
            )

            # Process tables
            for page in document.pages:
                if page.tables:
                    for table in page.tables:
                        table_analysis = TableParser.extract_table_metadata(
                            table.get("data", [])
                        )
                        if table_analysis:
                            table["summary"] = table_analysis.to_text()

            # Chunk document
            chunks = app.state.chunker.chunk_document(document)

            if not chunks:
                logger.warning("No chunks created from document")
                return UploadPDFResponse(
                    success=False,
                    document_id=document.document_id,
                    filename=file.filename,
                    total_pages=document.total_pages,
                    chunks_created=0,
                    message="Failed to create chunks from document",
                )

            # Generate embeddings
            chunks = await app.state.embedding_service.embed_chunks(chunks)

            # Add to vector store
            added = app.state.vector_store.add_chunks_batch(chunks)

            # Save vector store
            app.state.vector_store.save_to_disk()

            logger.info(
                f"Successfully processed {file.filename}: "
                f"{document.total_pages} pages, {added} chunks"
            )

            return UploadPDFResponse(
                success=True,
                document_id=document.document_id,
                filename=file.filename,
                total_pages=document.total_pages,
                chunks_created=added,
                message=f"Successfully processed PDF with {added} chunks",
            )

        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF: {str(e)}",
            )
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.post("/ask", response_model=AskQuestionResponse)
    async def ask_question(
        request: AskQuestionRequest,
    ) -> AskQuestionResponse:
        """
        Ask a question about the uploaded documents.

        Uses RAG to retrieve relevant chunks and generate an answer.
        """
        try:
            logger.info(f"Processing question: {request.question}")

            # Get answer from QA engine
            result = await app.state.qa_engine.answer_question(
                question=request.question,
                top_k=request.top_k,
                include_sources=True,
            )

            # Convert sources to API format
            sources = [
                SourceReference(**source) for source in result["sources"]
            ]

            return AskQuestionResponse(
                question=request.question,
                answer=result["answer"],
                sources=sources,
                chunks_used=result["chunks_used"],
            )

        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process question: {str(e)}",
            )

    @app.delete(
        "/delete_document/{document_id}",
        response_model=DeleteDocumentResponse,
    )
    async def delete_document(
        document_id: str,
    ) -> DeleteDocumentResponse:
        """
        Delete a document and its associated chunks from the vector store.

        Args:
            document_id: ID of the document to delete.

        Returns:
            Deletion result.
        """
        try:
            deleted = app.state.vector_store.delete_by_document(document_id)
            app.state.vector_store.save_to_disk()

            return DeleteDocumentResponse(
                success=True,
                document_id=document_id,
                chunks_deleted=deleted,
                message=f"Deleted {deleted} chunks from document {document_id}",
            )

        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document: {str(e)}",
            )

    @app.get("/stats")
    async def get_stats() -> Dict:
        """Get statistics about the system."""
        return {
            "vector_store": app.state.vector_store.get_stats(),
            "services": {
                "pdf_loader": "PDFLoader",
                "doc_intelligence": "DocumentIntelligenceExtractor",
                "embedding_service": "EmbeddingService",
                "vector_store": "FAISSVectorStore",
            },
        }

    return app
