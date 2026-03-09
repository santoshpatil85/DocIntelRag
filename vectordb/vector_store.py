"""
FAISS-based vector store for semantic search.

Manages storage and retrieval of document embeddings.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import pickle

import faiss
import numpy as np

from config.settings import settings
from utils.logger import setup_logger
from models.document import Chunk

logger = setup_logger(__name__)


class FAISSVectorStore:
    """
    Vector store implementation using Facebook's FAISS library.

    Supports fast similarity search over high-dimensional embeddings.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (default from settings).
            index_path: Path to save/load FAISS index.
            metadata_path: Path to save/load metadata.
        """
        self.dimension = dimension or settings.vectordb.dimension
        self.index_path = index_path or settings.vectordb.faiss_index_path
        self.metadata_path = metadata_path or settings.vectordb.metadata_path
        self.chunk_metadata = []  # List of chunk metadata
        self.index = None

        self._initialize_index()
        self._load_from_disk()

    def _initialize_index(self) -> None:
        """Initialize or create FAISS index."""
        try:
            # Create index using L2 (Euclidean) distance
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(
                f"Created FAISS index with dimension {self.dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise

    def add_chunk(
        self,
        chunk: Chunk,
    ) -> bool:
        """
        Add a single chunk with its embedding to the store.

        Args:
            chunk: Chunk object with embedding.

        Returns:
            True if successful, False otherwise.
        """
        if chunk.embedding is None:
            logger.warning(f"Chunk {chunk.chunk_id} has no embedding")
            return False

        try:
            # Convert embedding to numpy array
            embedding_array = np.array([chunk.embedding], dtype=np.float32)

            # Add to FAISS index
            self.index.add(embedding_array)

            # Store metadata
            chunk_dict = chunk.to_dict()
            chunk_dict.pop("embedding", None)  # Don't store embedding in metadata
            self.chunk_metadata.append(chunk_dict)

            return True

        except Exception as e:
            logger.error(f"Failed to add chunk {chunk.chunk_id}: {str(e)}")
            return False

    def add_chunks_batch(
        self,
        chunks: List[Chunk],
    ) -> int:
        """
        Add multiple chunks to the store.

        Args:
            chunks: List of Chunk objects with embeddings.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        try:
            # Filter chunks with embeddings
            chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

            if not chunks_with_embeddings:
                logger.warning("No chunks with valid embeddings provided")
                return 0

            # Convert all embeddings to numpy array
            embeddings = np.array(
                [c.embedding for c in chunks_with_embeddings],
                dtype=np.float32,
            )

            # Add to FAISS index
            self.index.add(embeddings)

            # Store metadata
            for chunk in chunks_with_embeddings:
                chunk_dict = chunk.to_dict()
                chunk_dict.pop("embedding", None)
                self.chunk_metadata.append(chunk_dict)

            logger.info(f"Added {len(chunks_with_embeddings)} chunks to vector store")
            return len(chunks_with_embeddings)

        except Exception as e:
            logger.error(f"Failed to add chunks batch: {str(e)}")
            return 0

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks using embedding similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of tuples (Chunk, similarity_score).
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        try:
            # Convert query to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)

            # Search FAISS index
            distances, indices = self.index.search(query_array, top_k)

            # Build results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):
                    # FAISS returns L2 distance, higher = less similar
                    # Convert to similarity score (lower distance = higher similarity)
                    similarity = 1.0 / (1.0 + distance)
                    chunk_dict = self.chunk_metadata[idx]
                    chunk = Chunk.from_dict(chunk_dict)
                    results.append((chunk, similarity))

            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def search_with_filter(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        chunk_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search with metadata filtering.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.
            chunk_type_filter: Filter by chunk type (e.g., "table", "chart").
            page_filter: Filter by page number.

        Returns:
            List of tuples (Chunk, similarity_score).
        """
        # Get all results (larger retrieval)
        all_results = self.search(query_embedding, top_k=top_k * 3)

        # Filter results
        filtered_results = []
        for chunk, score in all_results:
            if chunk_type_filter and chunk.chunk_type.value != chunk_type_filter:
                continue
            if page_filter and chunk.page_number != page_filter:
                continue
            filtered_results.append((chunk, score))

        return filtered_results[:top_k]

    def save_to_disk(self) -> bool:
        """
        Save index and metadata to disk.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save metadata
            with open(self.metadata_path, "w") as f:
                json.dump(self.chunk_metadata, f, indent=2)

            logger.info(
                f"Saved vector store to {self.index_path} and {self.metadata_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            return False

    def _load_from_disk(self) -> None:
        """Load index and metadata from disk if they exist."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(
                self.metadata_path
            ):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)

                # Load metadata
                with open(self.metadata_path, "r") as f:
                    self.chunk_metadata = json.load(f)

                logger.info(
                    f"Loaded vector store with {self.index.ntotal} chunks"
                )
            else:
                logger.info("No existing vector store found, starting fresh")

        except Exception as e:
            logger.warning(f"Failed to load vector store from disk: {str(e)}")
            self._initialize_index()

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics.
        """
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.dimension,
            "metadata_stored": len(self.chunk_metadata),
        }

    def clear(self) -> None:
        """Clear all data from the vector store."""
        try:
            self._initialize_index()
            self.chunk_metadata = []
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")

    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Note: FAISS doesn't support deletion, so we filter-rebuild the index.

        Args:
            document_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        try:
            # Filter out chunks from this document
            original_count = len(self.chunk_metadata)
            self.chunk_metadata = [
                m for m in self.chunk_metadata
                if m.get("document_id") != document_id
            ]

            deleted_count = original_count - len(self.chunk_metadata)

            # Rebuild index
            if self.chunk_metadata and self.chunk_metadata[0].get("embedding"):
                embeddings = np.array(
                    [m.get("embedding") for m in self.chunk_metadata],
                    dtype=np.float32,
                )
                self._initialize_index()
                self.index.add(embeddings)

            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete by document: {str(e)}")
            return 0
