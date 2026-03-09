"""
QA engine for answering questions over documents.

Uses RAG (Retrieval-Augmented Generation) with GPT-4o.
"""

from typing import List, Optional, Dict, Any
import re

from openai import AsyncAzureOpenAI

from config.settings import settings
from utils.logger import setup_logger
from models.document import Chunk
from rag.retriever import LayoutAwareRetriever

logger = setup_logger(__name__)


class RAGQAEngine:
    """
    Question-Answering engine using Retrieval-Augmented Generation.

    Retrieves relevant document chunks and uses GPT-4o to generate answers.
    """

    def __init__(
        self,
        retriever: LayoutAwareRetriever,
    ):
        """
        Initialize QA engine.

        Args:
            retriever: LayoutAwareRetriever instance.
        """
        try:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure.openai_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=settings.azure.openai_endpoint,
            )
            self.model = settings.azure.openai_model
            self.retriever = retriever
            logger.info(f"Initialized RAG QA Engine with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize QA engine: {str(e)}")
            raise

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve for context.
            include_sources: Whether to include source information.

        Returns:
            Dictionary with answer and metadata.
        """
        try:
            logger.info(f"Answering question: {question}")

            # Retrieve relevant chunks
            chunks = await self.retriever.retrieve(question, top_k=top_k)

            if not chunks:
                logger.warning("No relevant chunks found")
                return {
                    "answer": (
                        "I could not find relevant information to answer "
                        "your question."
                    ),
                    "sources": [],
                    "chunks_used": 0,
                }

            # Build context from chunks
            context = self.retriever.build_context(chunks)

            # Generate answer using GPT-4o
            answer = await self._generate_answer(question, context)

            # Extract source information
            sources = self._extract_sources(chunks)

            return {
                "answer": answer,
                "sources": sources if include_sources else [],
                "chunks_used": len(chunks),
            }

        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "chunks_used": 0,
                "error": str(e),
            }

    async def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using GPT-4o with retrieved context.

        Args:
            question: User question.
            context: Retrieved context.

        Returns:
            Generated answer.
        """
        try:
            system_prompt = """You are a helpful assistant specialized in 
analyzing documents and answering questions based on provided content.

Guidelines:
1. Answer only using the provided context
2. If information is not in the context, say so explicitly
3. Cite relevant page numbers and sources
4. Break down complex answers into clear sections
5. Explain tables, charts, and visual elements when referenced
6. Be concise but complete"""

            user_prompt = f"""Based on the following document content, 
please answer this question: {question}

DOCUMENT CONTEXT:
{context}

Please provide a clear, well-structured answer citing the relevant sources."""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return "Failed to generate an answer."

    def _extract_sources(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Extract source information from chunks.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            List of source dictionaries.
        """
        sources = []
        seen_pages = set()

        for chunk in chunks:
            source_key = (chunk.document_id, chunk.page_number)

            if source_key not in seen_pages:
                sources.append(
                    {
                        "document": chunk.metadata.get("document_name", "Unknown"),
                        "page": chunk.page_number,
                        "type": chunk.chunk_type.value,
                    }
                )
                seen_pages.add(source_key)

        return sources

    async def ask_followup(
        self,
        original_question: str,
        followup_question: str,
        previous_chunks: List[Chunk],
    ) -> str:
        """
        Answer a followup question with context from previous retrieval.

        Args:
            original_question: Original question.
            followup_question: Followup question.
            previous_chunks: Chunks from original retrieval.

        Returns:
            Answer string.
        """
        try:
            context = self.retriever.build_context(previous_chunks)

            system_prompt = """You are a helpful assistant in a document 
analysis conversation. The user has asked a followup question about 
their previous inquiry.

Use the provided context to answer the followup question while maintaining
consistency with your previous responses."""

            user_prompt = f"""Original question: {original_question}
Followup question: {followup_question}

DOCUMENT CONTEXT:
{context}

Please answer the followup question based on the same document context."""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to answer followup: {str(e)}")
            return "Failed to generate a response."
