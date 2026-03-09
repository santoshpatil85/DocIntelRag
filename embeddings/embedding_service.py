"""
Embedding generation using Azure OpenAI.

Converts text chunks to vector embeddings for semantic search.
"""

import asyncio
from typing import List, Optional
import time

from openai import AsyncAzureOpenAI

from config.settings import settings
from utils.logger import setup_logger, log_function_result
from models.document import Chunk

logger = setup_logger(__name__)


class EmbeddingService:
    """
    Generates embeddings for text using Azure OpenAI.

    Uses the text-embedding-3-large model for high-quality embeddings.
    """

    def __init__(self, batch_size: int = 20):
        """
        Initialize embedding service.

        Args:
            batch_size: Number of texts to embed in each batch.
        """
        try:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure.openai_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=settings.azure.openai_endpoint,
            )
            self.model = settings.azure.embedding_model
            self.batch_size = batch_size
            logger.info(
                f"Initialized EmbeddingService with model: {self.model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {str(e)}")
            raise

    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding, or None if failed.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
            )

            if response.data:
                return response.data[0].embedding
            else:
                logger.warning("No embedding returned from API")
                return None

        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            return None

    async def embed_texts_batch(
        self,
        texts: List[str],
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.

        Uses batching to optimize API calls while respecting rate limits.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings (None for failed items).
        """
        if not texts:
            return []

        embeddings = []
        start_time = time.time()

        try:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Call embedding API with batch
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )

                # Extract embeddings in order
                batch_embeddings = {}
                for data in response.data:
                    batch_embeddings[data.index] = data.embedding

                # Maintain order
                for j in range(len(batch)):
                    embeddings.append(batch_embeddings.get(j))

                logger.debug(
                    f"Embedded batch {i // self.batch_size + 1}/"
                    f"{(len(texts) + self.batch_size - 1) // self.batch_size}"
                )

                # Small delay between batches to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)

            duration = time.time() - start_time
            log_function_result(
                logger,
                "embed_texts_batch",
                duration,
                f"{len(embeddings)} embeddings generated",
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed batch: {str(e)}")
            # Return None for each item on failure
            return [None] * len(texts)

    async def embed_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        """
        Generate embeddings for a list of chunks.

        Modifies chunks in-place to add embeddings.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of chunks with embeddings (in-place modified).
        """
        if not chunks:
            return []

        try:
            # Extract texts from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = await self.embed_texts_batch(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            successful = sum(1 for e in embeddings if e is not None)
            logger.info(
                f"Embedded {successful}/{len(chunks)} chunks successfully"
            )

            return chunks

        except Exception as e:
            logger.error(f"Failed to embed chunks: {str(e)}")
            return chunks

    async def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: Chunk object.

        Returns:
            Chunk with embedding added.
        """
        try:
            embedding = await self.embed_text(chunk.content)
            chunk.embedding = embedding
            return chunk
        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk.chunk_id}: {str(e)}")
            return chunk

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension (typically 3072 for text-embedding-3-large).
        """
        return settings.vectordb.dimension
