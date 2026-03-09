#!/usr/bin/env python3
"""
Quick validation script for mock services.
Run this to verify all components are working.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.mock_services import create_mock_services
from models.document import Chunk, ChunkType
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def validate_embedding_service():
    """Validate embedding service works."""
    logger.info("Testing embedding service...")
    services = create_mock_services()
    
    embedding_service = services["embedding_service"]
    text = "This is a test document about machine learning."
    
    embedding = await embedding_service.embed_text(text)
    assert len(embedding) == 3072, f"Expected 3072 dims, got {len(embedding)}"
    assert all(isinstance(x, float) for x in embedding), "All values should be floats"
    
    logger.info(f"✓ Embedding service OK ({len(embedding)} dimensions)")
    return True


async def validate_vector_store():
    """Validate vector store works."""
    logger.info("Testing vector store...")
    services = create_mock_services()
    
    embedding_service = services["embedding_service"]
    vector_store = services["vector_store"]
    
    # Create test chunk
    text = "Machine learning is a subset of artificial intelligence."
    embedding = await embedding_service.embed_text(text)
    
    chunk = Chunk(
        chunk_id="test_1",
        document_id="doc_1",
        content=text,
        chunk_type=ChunkType.TEXT,
        page_number=1,
        embedding=embedding
    )
    
    # Add to store
    vector_store.add_chunk(chunk)
    
    # Search
    results = vector_store.search(embedding, top_k=1)
    assert len(results) > 0, "Should find at least one result"
    assert results[0]["chunk_id"] == "test_1", "Should find our chunk"
    
    # Get stats
    stats = vector_store.get_stats()
    assert stats["total_chunks"] > 0, "Should have chunks"
    
    logger.info(f"✓ Vector store OK ({stats['total_chunks']} chunks)")
    return True


async def validate_retriever():
    """Validate retriever works."""
    logger.info("Testing retriever...")
    services = create_mock_services()
    
    retriever = services["retriever"]
    
    # Retrieve chunks
    chunks = await retriever.retrieve("What is machine learning?", top_k=5)
    assert isinstance(chunks, list), "Should return a list"
    
    # Build context
    context = retriever.build_context(chunks)
    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"
    
    logger.info(f"✓ Retriever OK ({len(chunks)} chunks retrieved)")
    return True


async def validate_qa_engine():
    """Validate QA engine works."""
    logger.info("Testing QA engine...")
    services = create_mock_services()
    
    qa_engine = services["qa_engine"]
    
    # Ask a question
    result = await qa_engine.answer_question(
        "What is machine learning?",
        top_k=5,
        include_sources=True
    )
    
    assert "answer" in result, "Should have answer"
    assert "sources" in result, "Should have sources"
    assert "chunks_used" in result, "Should have chunks_used"
    assert len(result["answer"]) > 0, "Answer should not be empty"
    
    logger.info(f"✓ QA engine OK (answer: {len(result['answer'])} chars)")
    return True


async def validate_complete_pipeline():
    """Validate complete pipeline works."""
    logger.info("Testing complete pipeline...")
    services = create_mock_services()
    
    embedding_service = services["embedding_service"]
    vector_store = services["vector_store"]
    retriever = services["retriever"]
    qa_engine = services["qa_engine"]
    
    # Create multiple chunks
    texts = [
        "Deep learning uses neural networks with multiple layers.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Reinforcement learning learns from rewards and penalties.",
        "Transfer learning applies knowledge from one task to another.",
    ]
    
    logger.info(f"  Adding {len(texts)} chunks...")
    for i, text in enumerate(texts):
        embedding = await embedding_service.embed_text(text)
        chunk = Chunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            content=text,
            chunk_type=ChunkType.TEXT,
            page_number=i + 1,
            embedding=embedding
        )
        vector_store.add_chunk(chunk)
    
    # Search
    logger.info("  Searching...")
    chunks = await retriever.retrieve("Tell me about deep learning")
    
    # Generate answer
    logger.info("  Generating answer...")
    result = await qa_engine.answer_question(
        "What is deep learning?",
        top_k=3,
        include_sources=True
    )
    
    assert result["chunks_used"] > 0, "Should use chunks"
    assert len(result["sources"]) > 0, "Should have sources"
    
    logger.info(f"✓ Complete pipeline OK ({result['chunks_used']} chunks used)")
    return True


async def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("MOCK SERVICES VALIDATION")
    print("="*60 + "\n")
    
    tests = [
        ("Embedding Service", validate_embedding_service),
        ("Vector Store", validate_vector_store),
        ("Retriever", validate_retriever),
        ("QA Engine", validate_qa_engine),
        ("Complete Pipeline", validate_complete_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, "PASS"))
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
            results.append((test_name, "FAIL"))
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    # Summary
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        logger.info("✓ All validation tests passed!")
        print("System is ready for testing.\n")
        print("Quick start commands:")
        print("  1. Start mock server:  python tests/test_server.py")
        print("  2. Run demo script:    python tests/test_mock_demo.py")
        print("  3. Run unit tests:     pytest tests/ -v")
        print()
        return 0
    else:
        logger.error(f"✗ {total - passed} validation tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
