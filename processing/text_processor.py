"""
Text processing and normalization.

Handles text cleaning, normalization, and preparation for embeddings.
"""

import re
from typing import List

from utils.logger import setup_logger

logger = setup_logger(__name__)


class TextProcessor:
    """Processes and normalizes text content."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Removes extra whitespace, special characters, and normalizes spacing.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        # Normalize line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences.

        Uses simple heuristic-based sentence splitting.

        Args:
            text: Text to split into sentences.

        Returns:
            List of sentences.
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text.

        Simple implementation using word frequency and length.
        In production, use NLP libraries like spaCy or NLTK.

        Args:
            text: Text to extract keywords from.
            top_k: Number of keywords to extract.

        Returns:
            List of keywords.
        """
        # Simple tokenization
        words = re.findall(r"\b\w{4,}\b", text.lower())

        # Filter common words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "was",
            "are", "been", "be", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
        }

        filtered_words = [w for w in words if w not in stopwords]

        # Get frequency-based top words
        from collections import Counter
        counter = Counter(filtered_words)
        top_words = [word for word, _ in counter.most_common(top_k)]

        return top_words

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """
        Truncate text to maximum length while preserving words.

        Args:
            text: Text to truncate.
            max_length: Maximum length in characters.

        Returns:
            Truncated text.
        """
        if len(text) <= max_length:
            return text

        # Find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > 0:
            truncated = text[:last_space]

        return truncated + "..."
