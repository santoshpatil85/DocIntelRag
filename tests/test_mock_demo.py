"""
Example test/demo using mock services.

Demonstrates how to test the RAG system without Azure credentials.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mock_services import (
    MockEmbeddingService,
    MockVectorStore,
    MockRetriever,
    MockQAEngine,
    create_mock_services,
)
from models.document import Chunk, ChunkType
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def demo_mock_embedding():
    """Demo: Mock embedding service."""
    print("\n" + "="*60)
    print("DEMO 1: Mock Embedding Service")
    print("="*60)

    embedding_service = MockEmbeddingService()

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a powerful programming language",
        "Machine learning enables computers to learn from data",
    ]

    print(f"\nEmbedding {len(texts)} texts...")
    embeddings = await embedding_service.embed_texts_batch(texts)

    for text, emb in zip(texts, embeddings):
        print(f"\nText: {text[:50]}...")
        print(f"Embedding dimension: {len(emb)}")
        print(f"First 5 values: {[f'{x:.4f}' for x in emb[:5]]}")


async def demo_mock_vector_store():
    """Demo: Mock vector store."""
    print("\n" + "="*60)
    print("DEMO 2: Mock Vector Store")
    print("="*60)

    # Create mock embedding service and vector store
    embedding_service = MockEmbeddingService()
    vector_store = MockVectorStore()

    # Create and embed sample chunks
    chunks = [
        Chunk(
            chunk_id=f"chunk_{i}",
            document_id="demo_doc",
            content=f"Sample content about machine learning concept {i}",
            chunk_type=ChunkType.TEXT,
            page_number=i + 1,
        )
        for i in range(5)
    ]

    print(f"\nCreated {len(chunks)} sample chunks")

    # Embed chunks
    chunks = await embedding_service.embed_chunks(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks")

    # Add to vector store
    added = vector_store.add_chunks_batch(chunks)
    print(f"Added {added} chunks to vector store")

    # Get statistics
    stats = vector_store.get_stats()
    print(f"\nVector store stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")

    # Search
    if chunks[0].embedding:
        results = vector_store.search(chunks[0].embedding, top_k=3)
        print(f"\nSearch results (top 3):")
        for chunk, score in results:
            print(f"  {chunk.chunk_id}: similarity={score:.4f}")


async def demo_mock_retriever():
    """Demo: Mock retriever."""
    print("\n" + "="*60)
    print("DEMO 3: Mock Retriever")
    print("="*60)

    retriever = MockRetriever()

    query = "What is machine learning?"
    print(f"\nQuery: {query}")

    chunks = await retriever.retrieve(query, top_k=3)
    print(f"Retrieved {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Page {chunk.page_number}: {chunk.content[:60]}...")

    context = retriever.build_context(chunks)
    print(f"\nFormatted context:\n{context}")


async def demo_mock_qa_engine():
    """Demo: Mock QA engine."""
    print("\n" + "="*60)
    print("DEMO 4: Mock QA Engine")
    print("="*60)

    qa_engine = MockQAEngine()

    questions = [
        "What is the main topic of this document?",
        "What are the key findings?",
        "How is table 1 structured?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = await qa_engine.answer_question(question, top_k=3)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['sources'])} document sections")


async def demo_complete_pipeline():
    """Demo: Complete mock RAG pipeline."""
    print("\n" + "="*60)
    print("DEMO 5: Complete Mock RAG Pipeline")
    print("="*60)

    # Create all mock services
    services = create_mock_services()
    print("\n✓ Created mock services")

    # Create sample chunks
    sample_chunks = [
        Chunk(
            chunk_id=f"doc_chunk_{i}",
            document_id="sample_doc_001",
            content=f"This is paragraph {i} about data analysis and insights. "
            f"It contains important information about the topic.",
            chunk_type=ChunkType.TEXT,
            page_number=i + 1,
        )
        for i in range(5)
    ]
    print(f"✓ Created {len(sample_chunks)} sample chunks")

    # Embed chunks
    embedding_service = services["embedding_service"]
    sample_chunks = await embedding_service.embed_chunks(sample_chunks)
    print("✓ Generated embeddings")

    # Add to vector store
    vector_store = services["vector_store"]
    added = vector_store.add_chunks_batch(sample_chunks)
    print(f"✓ Stored {added} chunks in vector store")

    # Get QA engine
    qa_engine = services["qa_engine"]

    # Ask questions
    test_questions = [
        "What is the main content?",
        "Summarize the document",
    ]

    print("\nRunning mock Q&A:")
    for question in test_questions:
        response = await qa_engine.answer_question(question, top_k=3)
        print(f"\n  Q: {question}")
        print(f"  A: {response['answer'][:100]}...")
        print(f"  Sources: {response['chunks_used']} chunks")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("MOCK SERVICES DEMONSTRATION")
    print("Testing without Azure credentials")
    print("="*60)

    try:
        await demo_mock_embedding()
        await demo_mock_vector_store()
        await demo_mock_retriever()
        await demo_mock_qa_engine()
        await demo_complete_pipeline()

        print("\n" + "="*60)
        print("✓ All demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
