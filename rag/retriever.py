"""
Retrieval system with layout-aware search.

Performs semantic search with content-type prioritization.
"""

import re
from typing import List, Optional, Dict, Any

from vectordb.vector_store import FAISSVectorStore
from embeddings.embedding_service import EmbeddingService
from utils.logger import setup_logger
from models.document import Chunk, ChunkType

logger = setup_logger(__name__)


class LayoutAwareRetriever:
    """
    Retrieves relevant document chunks using semantic search.

    Enhances retrieval by analyzing query and prioritizing chunk types.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: FAISSVectorStore instance.
            embedding_service: EmbeddingService instance.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def _analyze_query(self, query: str) -> Optional[str]:
        """
        Analyze query to determine preferred chunk type.

        Args:
            query: User query string.

        Returns:
            Preferred chunk type ('table', 'chart', 'text') or None.
        """
        query_lower = query.lower()

        # Keywords for different chunk types
        table_keywords = [
            "table",
            "data",
            "numbers",
            "values",
            "statistics",
            "metrics",
            "rows",
            "columns",
        ]
        chart_keywords = [
            "chart",
            "graph",
            "visual",
            "trend",
            "diagram",
            "picture",
            "image",
            "plot",
        ]

        # Count keyword matches
        table_matches = sum(1 for kw in table_keywords if kw in query_lower)
        chart_matches = sum(1 for kw in chart_keywords if kw in query_lower)

        if table_matches > chart_matches and table_matches > 0:
            return ChunkType.TABLE.value
        elif chart_matches > table_matches and chart_matches > 0:
            return ChunkType.CHART.value

        return None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of relevant Chunk objects.
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)

            if query_embedding is None:
                logger.warning("Failed to generate query embedding")
                return []

            # Analyze query for preferred chunk type
            preferred_type = self._analyze_query(query)

            # Retrieve with filter if available
            if preferred_type:
                results = self.vector_store.search_with_filter(
                    query_embedding,
                    top_k=top_k,
                    chunk_type_filter=preferred_type,
                )
            else:
                results = self.vector_store.search(query_embedding, top_k=top_k)

            # Extract chunks from results
            chunks = [chunk for chunk, score in results]

            logger.info(
                f"Retrieved {len(chunks)} chunks for query "
                f"(preferred type: {preferred_type})"
            )

            return chunks

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[tuple[Chunk, float]]:
        """
        Retrieve chunks with similarity scores.

        Args:
            query: User query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of tuples (Chunk, similarity_score).
        """
        try:
            query_embedding = await self.embedding_service.embed_text(query)

            if query_embedding is None:
                return []

            preferred_type = self._analyze_query(query)

            if preferred_type:
                results = self.vector_store.search_with_filter(
                    query_embedding,
                    top_k=top_k,
                    chunk_type_filter=preferred_type,
                )
            else:
                results = self.vector_store.search(query_embedding, top_k=top_k)

            return results

        except Exception as e:
            logger.error(f"Retrieval with scores failed: {str(e)}")
            return []

    def _format_chunk_for_context(self, chunk: Chunk) -> str:
        """
        Format chunk for inclusion in context.

        Args:
            chunk: Chunk to format.

        Returns:
            Formatted chunk text.
        """
        parts = [f"[Page {chunk.page_number}]"]

        if chunk.section_title:
            parts.append(f"Section: {chunk.section_title}")

        parts.append(f"Type: {chunk.chunk_type.value.upper()}")
        parts.append(f"Content: {chunk.content}")

        return "\n".join(parts)

    def build_context(self, chunks: List[Chunk]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of relevant chunks.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return "No relevant information found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"\n--- Source {i} ---")
            context_parts.append(self._format_chunk_for_context(chunk))

        return "\n".join(context_parts)
