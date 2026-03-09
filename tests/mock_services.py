"""
Mock implementations for testing without Azure services.

Provides stub classes that simulate Azure services for development and testing.
"""

from typing import List, Optional, Dict, Any
import asyncio
import uuid

from utils.logger import setup_logger
from models.document import Chunk, ChunkType

logger = setup_logger(__name__)


class MockEmbeddingService:
    """Mock embedding service that returns dummy vectors."""

    def __init__(self, batch_size: int = 20):
        """Initialize mock embedding service."""
        self.model = "text-embedding-3-large"
        self.batch_size = batch_size
        self.embedding_dim = 3072
        logger.info("Initialized MockEmbeddingService")

    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate mock embedding (deterministic based on text length).

        Args:
            text: Text to embed.

        Returns:
            Mock embedding vector.
        """
        # Generate deterministic vector based on text
        seed_val = hash(text) % 10000
        import random
        random.seed(seed_val)
        embedding = [
            random.gauss(0, 1) for _ in range(self.embedding_dim)
        ]
        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]
        
        logger.debug(f"Generated mock embedding for text length {len(text)}")
        return embedding

    async def embed_texts_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate mock embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            emb = await self.embed_text(text)
            embeddings.append(emb)
        logger.info(f"Generated {len(embeddings)} mock embeddings")
        return embeddings

    async def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts_batch(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    async def embed_chunk(self, chunk: Chunk) -> Chunk:
        """Generate embedding for single chunk."""
        chunk.embedding = await self.embed_text(chunk.content)
        return chunk

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class MockRetriever:
    """Mock retriever that returns dummy results."""

    def __init__(self):
        """Initialize mock retriever."""
        self.sample_chunks = []
        logger.info("Initialized MockRetriever")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Mock retrieve - returns sample chunks based on query.

        Args:
            query: User query.
            top_k: Number of results.

        Returns:
            List of mock chunks.
        """
        # Create dummy chunks if none exist
        if not self.sample_chunks:
            self.sample_chunks = [
                Chunk(
                    chunk_id=f"mock_chunk_{i}",
                    document_id="mock_doc_001",
                    content=f"Mock content for chunk {i}. "
                             f"This is sample text about {query.lower() if len(query) < 50 else 'document analysis'}.",
                    chunk_type=ChunkType.TEXT,
                    page_number=i + 1,
                    metadata={"source": "mock"},
                )
                for i in range(10)
            ]

        logger.debug(f"Mock retrieve for query: {query}")
        return self.sample_chunks[:top_k]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[tuple]:
        """Mock retrieve with similarity scores."""
        chunks = await self.retrieve(query, top_k)
        # Return chunks with mock similarity scores
        return [(chunk, 0.85 - i * 0.05) for i, chunk in enumerate(chunks)]

    def build_context(self, chunks: List[Chunk]) -> str:
        """Build context string from chunks."""
        parts = [f"Context from {len(chunks)} chunks:"]
        for chunk in chunks:
            parts.append(f"[Page {chunk.page_number}] {chunk.content[:100]}...")
        return "\n".join(parts)


class MockQAEngine:
    """Mock QA engine that returns dummy answers."""

    def __init__(self, retriever=None):
        """Initialize mock QA engine."""
        self.retriever = retriever or MockRetriever()
        logger.info("Initialized MockQAEngine")

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Mock answer question.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            include_sources: Whether to include sources.

        Returns:
            Mock answer response.
        """
        chunks = await self.retriever.retrieve(question, top_k)

        # Generate mock answer based on question
        answer = (
            f"Based on the document analysis, regarding '{question}': "
            f"This is a mock response. In production, this would be "
            f"generated by GPT-4o using the retrieved context from "
            f"{len(chunks)} relevant document sections."
        )

        sources = [
            {
                "document": "mock_document.pdf",
                "page": chunk.page_number,
                "type": chunk.chunk_type.value,
            }
            for chunk in chunks
        ]

        logger.info(f"Mock answer generated for: {question}")

        return {
            "answer": answer,
            "sources": sources if include_sources else [],
            "chunks_used": len(chunks),
        }

    async def ask_followup(
        self,
        original_question: str,
        followup_question: str,
        previous_chunks: List[Chunk],
    ) -> str:
        """Answer a followup question."""
        return (
            f"Follow-up answer to '{followup_question}' based on "
            f"original question '{original_question}': "
            f"This is a mock follow-up response."
        )


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, dimension: int = 3072):
        """Initialize mock vector store."""
        self.dimension = dimension
        self.chunks = []
        self.total_stored = 0
        logger.info(f"Initialized MockVectorStore with dimension {dimension}")

    def add_chunk(self, chunk: Chunk) -> bool:
        """Mock add chunk."""
        if chunk.embedding is None:
            return False
        self.chunks.append(chunk)
        self.total_stored += 1
        return True

    def add_chunks_batch(self, chunks: List[Chunk]) -> int:
        """Mock add chunks batch."""
        added = 0
        for chunk in chunks:
            if self.add_chunk(chunk):
                added += 1
        logger.info(f"Mock: Added {added} chunks to store")
        return added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Mock search."""
        # Return mock results
        results = []
        for i, chunk in enumerate(self.chunks[:top_k]):
            score = 0.9 - (i * 0.1)
            results.append({
                "chunk_id": chunk.chunk_id,
                "chunk": chunk,
                "score": score
            })
        return results

    def search_with_filter(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Mock filtered search."""
        all_results = self.search(query_embedding, top_k * 2)

        filtered = [
            result
            for result in all_results
            if (chunk_type_filter is None or result["chunk"].chunk_type.value == chunk_type_filter)
            and (page_filter is None or result["chunk"].page_number == page_filter)
        ]

        return filtered[:top_k]

    def save_to_disk(self) -> bool:
        """Mock save."""
        logger.info("Mock: Saved vector store (no-op)")
        return True

    def get_stats(self) -> Dict[str, any]:
        """Get mock statistics."""
        return {
            "total_chunks": self.total_stored,
            "embedding_dimension": self.dimension,
            "metadata_stored": self.total_stored,
        }

    def delete_by_document(self, document_id: str) -> int:
        """Mock delete by document."""
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        deleted = original_count - len(self.chunks)
        logger.info(f"Mock: Deleted {deleted} chunks for document {document_id}")
        return deleted

    def clear(self) -> None:
        """Clear store."""
        self.chunks = []
        self.total_stored = 0


def create_mock_services():
    """
    Create a set of mock services for testing.

    Returns:
        Dictionary of mock service instances.
    """
    embedding_service = MockEmbeddingService()
    vector_store = MockVectorStore()
    retriever = MockRetriever()
    qa_engine = MockQAEngine(retriever)

    logger.info("Created all mock services")

    return {
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "retriever": retriever,
        "qa_engine": qa_engine,
    }
