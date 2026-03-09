"""
Configuration management for the Multimodal RAG system.

Loads environment variables and provides centralized configuration
for all services.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AzureConfig:
    """Azure service credentials and endpoints."""

    document_intelligence_endpoint: str
    document_intelligence_key: str
    openai_endpoint: str
    openai_api_key: str
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    faiss_index_path: str = "./data/faiss_index.bin"
    metadata_path: str = "./data/metadata.json"
    dimension: int = 3072  # Dimension for text-embedding-3-large


@dataclass
class ProcessingConfig:
    """Document processing configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    max_image_width: int = 1024
    max_image_height: int = 1024


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class Settings:
    """Centralized settings class for the entire application."""

    def __init__(self):
        """Initialize all configuration sections."""
        self.azure = AzureConfig(
            document_intelligence_endpoint=os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""
            ),
            document_intelligence_key=os.getenv(
                "AZURE_DOCUMENT_INTELLIGENCE_KEY", ""
            ),
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        )

        self.vectordb = VectorDBConfig(
            faiss_index_path=os.getenv(
                "FAISS_INDEX_PATH", "./data/faiss_index.bin"
            ),
            metadata_path=os.getenv("METADATA_PATH", "./data/metadata.json"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "3072")),
        )

        self.processing = ProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            max_image_width=int(os.getenv("MAX_IMAGE_WIDTH", "1024")),
            max_image_height=int(os.getenv("MAX_IMAGE_HEIGHT", "1024")),
        )

        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> bool:
        """
        Validate that all required configuration is present.

        Returns:
            bool: True if all required configs are set, False otherwise.
        """
        # Skip validation in mock/test mode
        skip_validation = os.getenv("SKIP_VALIDATION", "false").lower() == "true"
        if skip_validation:
            return True

        required_fields = [
            self.azure.document_intelligence_endpoint,
            self.azure.document_intelligence_key,
            self.azure.openai_endpoint,
            self.azure.openai_api_key,
        ]
        return all(required_fields)


# Global settings instance
settings = Settings()
