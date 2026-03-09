"""
Comprehensive README for the Multimodal RAG System.

This document describes the architecture, setup, and usage of the system.
"""

# Multimodal RAG System for PDF Analysis

A production-ready Python service for analyzing complex PDF documents using Retrieval-Augmented Generation (RAG) with multimodal support (text, tables, charts).

## Features

- **PDF Ingestion**: Load and extract content from PDF files
- **OCR & Layout Extraction**: Azure Document Intelligence for advanced document understanding
- **Table Understanding**: Automatic table detection and summarization
- **Chart/Image Analysis**: GPT-4o Vision for chart interpretation
- **Hierarchical Chunking**: Intelligent document segmentation with content-type awareness
- **Semantic Search**: FAISS-based vector similarity search
- **Layout-Aware Retrieval**: Query-aware chunk prioritization
- **Question Answering**: RAG-based Q&A using GPT-4o
- **RESTful API**: FastAPI endpoints for easy integration

## Architecture

```
pdf_ingestion
├── pdf_loader.py           # PyMuPDF-based PDF extraction
├── document_intelligence_extractor.py  # Azure Document Intelligence
└── image_extractor.py      # Image region detection

document_processing
├── text_processor.py       # Text cleaning and normalization
├── table_parser.py         # Table analysis and summarization
├── chart_analyzer.py       # GPT-4o Vision chart analysis
└── chunking.py             # Hierarchical document chunking

embeddings
└── embedding_service.py    # Azure OpenAI embedding generation

vectordb
└── vector_store.py         # FAISS-based vector database

rag
├── retriever.py            # Layout-aware semantic retrieval
└── qa_engine.py            # RAG-based question answering

api
├── routes.py               # FastAPI endpoint handlers
└── schemas.py              # Pydantic request/response models

config
└── settings.py             # Configuration management

utils
└── logger.py               # Structured logging setup
```

## Setup

### Prerequisites

- Python 3.10+
- Azure subscription with:
  - Azure AI Document Intelligence resource
  - Azure OpenAI resource (with GPT-4o and text-embedding-3-large models)

### Installation

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd DocIntelRag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

### Azure Configuration

Update `.env` with your credentials:

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
```

## Usage

### Start the Server

```bash
python main.py
```

Server runs on `http://localhost:8000`

API documentation at `http://localhost:8000/docs`

### Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Upload PDF
```bash
POST /upload_pdf
Content-Type: multipart/form-data

Form data:
- file: <pdf_file>
```

Response:
```json
{
  "success": true,
  "document_id": "uuid",
  "filename": "document.pdf",
  "total_pages": 10,
  "chunks_created": 45,
  "message": "Successfully processed PDF with 45 chunks"
}
```

#### 3. Ask Question
```bash
POST /ask
Content-Type: application/json

{
  "question": "What are the key metrics in table 1?",
  "document_id": "uuid",  # Optional
  "top_k": 5
}
```

Response:
```json
{
  "question": "What are the key metrics in table 1?",
  "answer": "According to the table on page 3...",
  "sources": [
    {
      "document": "document.pdf",
      "page": 3,
      "type": "table"
    }
  ],
  "chunks_used": 2
}
```

#### 4. Get Statistics
```bash
GET /stats
```

#### 5. Delete Document
```bash
DELETE /delete_document/{document_id}
```

## Processing Pipeline

### Step 1: PDF Ingestion
- Load PDF using PyMuPDF
- Extract text and images from each page

### Step 2: Layout Extraction
- Send PDF to Azure Document Intelligence
- Extract document structure (paragraphs, tables, figures)
- Get layout bounding boxes and OCR results

### Step 3: Content Processing
- **Text**: Clean, normalize, extract keywords
- **Tables**: Convert to DataFrames, generate summaries
- **Charts**: Analyze with GPT-4o Vision, extract insights

### Step 4: Hierarchical Chunking
- Create text chunks with sliding window (512 tokens, 64 overlap)
- Create table chunks from summaries
- Create chart chunks from GPT-4o analysis
- Attach metadata: page number, section title, bounding box

### Step 5: Embedding Generation
- Embed all chunks using text-embedding-3-large
- Dimension: 3072

### Step 6: Vector Storage
- Store embeddings in FAISS index
- Save metadata alongside

### Step 7: Retrieval
- Analyze user query for content type preference
- Retrieve top-k most similar chunks
- Prioritize tables/charts if relevant to query

### Step 8: Question Answering
- Build context from retrieved chunks
- Pass to GPT-4o with system prompt
- Generate answer with citations

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| CHUNK_SIZE | 512 | Target size for text chunks (words) |
| CHUNK_OVERLAP | 64 | Overlap between chunks (words) |
| EMBEDDING_DIMENSION | 3072 | Dimension for embeddings |
| API_HOST | 0.0.0.0 | API server host |
| API_PORT | 8000 | API server port |
| LOG_LEVEL | INFO | Logging level |

## Code Examples

### Python Client

```python
import asyncio
import aiohttp
import aiofiles

async def upload_and_query():
    async with aiohttp.ClientSession() as session:
        # Upload PDF
        with open('document.pdf', 'rb') as f:
            form_data = aiohttp.FormData()
            form_data.add_field('file', f, filename='document.pdf')
            
            async with session.post(
                'http://localhost:8000/upload_pdf',
                data=form_data
            ) as resp:
                result = await resp.json()
                doc_id = result['document_id']
        
        # Ask question
        async with session.post(
            'http://localhost:8000/ask',
            json={
                'question': 'Summarize the main findings',
                'document_id': doc_id,
                'top_k': 5
            }
        ) as resp:
            answer = await resp.json()
            print(answer['answer'])

asyncio.run(upload_and_query())
```

### CURL Examples

```bash
# Upload PDF
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload_pdf

# Ask question (requires jq)
DOC_ID=$(curl -s -X POST -F "file=@document.pdf" \
  http://localhost:8000/upload_pdf | jq -r '.document_id')

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the summary?",
    "document_id": "'$DOC_ID'",
    "top_k": 5
  }'
```

## Performance Optimization

### Implemented Optimizations

- **Async Processing**: Concurrent page extraction and embedding generation
- **Batch Embeddings**: Process multiple texts in single API call
- **Sliding Window Chunking**: Preserve context with overlapping chunks
- **Lazy Loading**: Load FAISS index and metadata on startup

### Future Improvements

- Parallel page processing with thread pools
- Hybrid search combining vector + keyword search
- Metadata-based filtering and faceted search
- Query expansion and multi-step retrieval
- Embedding caching for frequently asked questions

## Logging

Structured logging configured in `utils/logger.py`:

```python
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Processing started", extra={"doc_id": "123"})
```

## Error Handling

All components include try-catch blocks with logging:

```python
try:
    # Operation
except SpecificException as e:
    logger.error(f"Operation failed: {str(e)}")
    return None
```

## Type Hints

Full type annotation throughout for IDE assistance:

```python
async def embed_chunks(
    self,
    chunks: List[Chunk],
) -> List[Chunk]:
    """Generate embeddings for chunks."""
    # Implementation
```

## Testing

Basic test structure:

```bash
pytest tests/
```

Example test:
```python
import pytest
from vectordb.vector_store import FAISSVectorStore

@pytest.mark.asyncio
async def test_vector_store():
    store = FAISSVectorStore()
    assert store.index is not None
```

## Production Deployment

### Considerations

- Use environment variables for secrets (Azure keys)
- Configure log aggregation (ELK, Datadog)
- Set up monitoring for API latency
- Use HTTPS in production
- Rate limit the /ask endpoint
- Implement authentication/authorization

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PORT=8000
EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t docintelsrag .
docker run -p 8000:8000 --env-file .env docintelsrag
```

## Troubleshooting

### Issue: "Document Intelligence connection failed"
- Verify endpoint URL and key in .env
- Check Azure resource exists and is in correct region

### Issue: "No embeddings returned"
- Verify Azure OpenAI resource has gpt-4o and embeddings models deployed
- Check API version compatibility

### Issue: "FAISS index not found"
- First upload will create index automatically
- Check write permissions on data directory

## Contributing

Guidelines for extending the system:

1. Add new chunk types in `models/document.py`
2. Implement new processors in `processing/`
3. Add new API endpoints in `api/routes.py`
4. Add Pydantic schemas in `api/schemas.py`
5. Update tests

## License

MIT License - See LICENSE file

## Support

For issues, questions, or suggestions:
1. Check existing issues on GitHub
2. Create detailed bug report
3. Include: error message, steps to reproduce, environment version
Project for Multimodal Retrieval Augmented Generation (RAG) system for analyzing complex PDFs
