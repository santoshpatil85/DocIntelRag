"""
Main entry point for the Multimodal RAG system.

Starts the FastAPI server and manages the application lifecycle.
"""

import asyncio
import logging
from pathlib import Path

import uvicorn

from config.settings import settings
from utils.logger import setup_logger
from api.routes import create_app

logger = setup_logger(__name__)


def main():
    """Main entry point for the application."""
    # Validate configuration
    if not settings.validate():
        logger.error(
            "Configuration validation failed. "
            "Please set required environment variables."
        )
        return

    # Create data directories
    Path(settings.vectordb.faiss_index_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    # Create FastAPI app
    app = create_app()

    # Start server
    logger.info(
        f"Starting Multimodal RAG service on "
        f"{settings.api.host}:{settings.api.port}"
    )

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.api.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
