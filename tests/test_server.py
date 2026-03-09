"""
Test entry point using mock services.

Run this to start the API with mock services (no Azure credentials needed).
"""

import os
from pathlib import Path

# Set mock mode before importing anything else
os.environ["MOCK_MODE"] = "true"
os.environ["SKIP_VALIDATION"] = "true"

from config.settings import settings
from utils.logger import setup_logger
from tests.mock_services import (
    MockEmbeddingService,
    MockVectorStore,
    MockRetriever,
    MockQAEngine,
)

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    SourceReference,
    HealthResponse,
    VectorStoreStats,
)

logger = setup_logger(__name__)


def create_test_app() -> FastAPI:
    """Create FastAPI application with mock services."""
    app = FastAPI(
        title="Multimodal RAG System (Mock Mode)",
        description="PDF analysis RAG with mock services for testing",
        version="1.0.0-mock",
    )

    # Initialize mock services
    embedding_service = MockEmbeddingService()
    vector_store = MockVectorStore()
    retriever = MockRetriever()
    qa_engine = MockQAEngine(retriever)

    # Store in app state
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.qa_engine = qa_engine

    @app.get("/health")
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        stats = vector_store.get_stats()
        return HealthResponse(
            status="healthy (mock mode)",
            vector_store_stats=VectorStoreStats(
                total_chunks=stats["total_chunks"],
                embedding_dimension=stats["embedding_dimension"],
                metadata_stored=stats["metadata_stored"],
            ),
        )

    @app.post("/upload_pdf")
    async def upload_pdf(file: UploadFile = File(...)):
        """Mock PDF upload endpoint."""
        logger.info(f"Mock: Processing PDF: {file.filename}")

        return {
            "success": True,
            "document_id": "mock_doc_" + file.filename.replace(".", "_"),
            "filename": file.filename,
            "total_pages": 10,
            "chunks_created": 45,
            "message": f"Mock: Would process PDF with 45 chunks",
        }

    @app.post("/ask", response_model=AskQuestionResponse)
    async def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
        """Ask question using mock services."""
        try:
            logger.info(f"Mock: Processing question: {request.question}")

            result = await app.state.qa_engine.answer_question(
                question=request.question,
                top_k=request.top_k,
                include_sources=True,
            )

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
            logger.error(f"Mock: Error answering question: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error: {str(e)}",
            )

    @app.delete("/delete_document/{document_id}")
    async def delete_document(document_id: str):
        """Mock delete document endpoint."""
        logger.info(f"Mock: Deleting document {document_id}")

        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": 0,
            "message": f"Mock: Would delete chunks for {document_id}",
        }

    @app.get("/stats")
    async def get_stats():
        """Get system statistics."""
        return {
            "mode": "MOCK (no Azure services)",
            "vector_store": vector_store.get_stats(),
            "services": {
                "embedding_service": "MockEmbeddingService",
                "vector_store": "MockVectorStore",
                "retriever": "MockRetriever",
                "qa_engine": "MockQAEngine",
            },
        }

    return app


def main():
    """Start the test server with mock services."""
    logger.info("Starting Multimodal RAG Service in MOCK MODE")
    logger.info(f"API will run on http://{settings.api.host}:{settings.api.port}")
    logger.info("Interactive docs at http://localhost:8000/docs")
    logger.info("\n⚠️  WARNING: This is mock mode - no actual Azure services are used")
    logger.info("Real Azure credentials are NOT required\n")

    # Create app with mock services
    app = create_test_app()

    # Start server
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.api.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
