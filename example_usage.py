"""
Example usage and integration test for the Multimodal RAG system.

Demonstrates how to use the system programmatically.
"""

import asyncio
import os
from pathlib import Path

from config.settings import settings
from ingestion.pipeline import PDFIngestionPipeline
from embeddings.embedding_service import EmbeddingService
from vectordb.vector_store import FAISSVectorStore
from rag.retriever import LayoutAwareRetriever
from rag.qa_engine import RAGQAEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def main():
    """
    Example usage of the Multimodal RAG system.

    This demonstrates the complete flow from PDF ingestion to Q&A.
    """

    # Initialize components
    logger.info("Initializing RAG system components...")

    pipeline = PDFIngestionPipeline()
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore()
    retriever = LayoutAwareRetriever(vector_store, embedding_service)
    qa_engine = RAGQAEngine(retriever)

    # Example PDF path (change to your PDF)
    pdf_path = "example_document.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return

    # Step 1: Ingest PDF
    logger.info(f"Ingesting PDF: {pdf_path}")
    document, chunks = await pipeline.process_pdf(
        pdf_path,
        document_id="example_001",
    )

    if document is None:
        logger.error("Failed to ingest PDF")
        return

    logger.info(f"Ingested document: {document.filename}")
    logger.info(f"Created {len(chunks)} chunks")

    # Step 2: Generate embeddings
    logger.info("Generating embeddings...")
    chunks = await embedding_service.embed_chunks(chunks)

    successful_embeddings = sum(1 for c in chunks if c.embedding is not None)
    logger.info(f"Generated {successful_embeddings} embeddings")

    # Step 3: Add to vector store
    logger.info("Adding chunks to vector store...")
    added = vector_store.add_chunks_batch(chunks)
    logger.info(f"Added {added} chunks to vector store")

    # Step 4: Save vector store
    logger.info("Saving vector store...")
    vector_store.save_to_disk()

    # Step 5: Ask questions
    logger.info("Answering questions...")

    questions = [
        "What are the main topics covered in this document?",
        "Can you summarize the key findings?",
        "What tables or data are presented?",
    ]

    for question in questions:
        logger.info(f"Q: {question}")

        result = await qa_engine.answer_question(
            question=question,
            top_k=5,
            include_sources=True,
        )

        logger.info(f"A: {result['answer']}")
        logger.info(f"Sources: {result['sources']}")
        logger.info(f"Chunks used: {result['chunks_used']}\n")


if __name__ == "__main__":
    # Validate configuration
    if not settings.validate():
        logger.error("Configuration validation failed")
        exit(1)

    # Run example
    asyncio.run(main())
