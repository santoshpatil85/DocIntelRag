"""
Testing Guide for Multimodal RAG System

Comprehensive guide for testing without Azure credentials.
"""

# Testing Guide for Multimodal RAG System

## Quick Start with Mock Services

The system includes complete mock implementations for all Azure services, allowing you to test and develop without Azure credentials.

## Setup

### 1. Environment Configuration

Mock `.env` file has been created with dummy credentials:

```bash
cat /workspaces/DocIntelRag/.env
```

**Key settings for testing:**
```env
MOCK_MODE=true          # Enable mock mode
SKIP_VALIDATION=true    # Skip credential validation
LOG_LEVEL=DEBUG         # Verbose logging for debugging
```

## Running Tests

### Option 1: Mock API Server (Recommended)

Start the API server with mock services:

```bash
python tests/test_server.py
```

**Output:**
```
2026-03-09 21:15:30 - tests.test_server - INFO - Starting Multimodal RAG Service in MOCK MODE
2026-03-09 21:15:30 - tests.test_server - INFO - API will run on http://0.0.0.0:8000
2026-03-09 21:15:30 - tests.test_server - INFO - Interactive docs at http://localhost:8000/docs

⚠️  WARNING: This is mock mode - no actual Azure services are used
Real Azure credentials are NOT required
```

**Test with curl:**

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "top_k": 5
  }'

# Upload mock PDF
curl -X POST -F "file=@sample.pdf" http://localhost:8000/upload_pdf

# View API documentation
open http://localhost:8000/docs
```

### Option 2: Run Mock Demo

Run the interactive mock services demo:

```bash
python tests/test_mock_demo.py
```

**Sample output:**
```
============================================================
MOCK SERVICES DEMONSTRATION
Testing without Azure credentials
============================================================

============================================================
DEMO 1: Mock Embedding Service
============================================================

Embedding 3 texts...

Text: The quick brown fox jumps over the lazy dog...
Embedding dimension: 3072
First 5 values: ['-0.0234', '0.1456', '-0.0892', '0.2134', '0.0456']

... (more demos)
```

### Option 3: Run Unit Tests

Run the test suite:

```bash
pytest tests/test_mock_demo.py -v
```

## Mock Services Overview

### MockEmbeddingService

Generates deterministic embeddings based on text content.

```python
from tests.mock_services import MockEmbeddingService

embedding_service = MockEmbeddingService()
embedding = await embedding_service.embed_text("Hello world")
# Returns: [3072-dimensional vector]
```

**Features:**
- Async embedding generation
- Batch processing
- 3072-dimensional vectors (matches text-embedding-3-large)
- Deterministic (same text = same embedding)

### MockVectorStore

In-memory FAISS-like vector store simulation.

```python
from tests.mock_services import MockVectorStore
from models.document import Chunk, ChunkType

store = MockVectorStore()

chunk = Chunk(
    chunk_id="test_1",
    document_id="doc_1",
    content="Test content",
    chunk_type=ChunkType.TEXT,
    page_number=1,
    embedding=[0.1] * 3072
)

store.add_chunk(chunk)
results = store.search([0.1] * 3072, top_k=5)
```

**Features:**
- Add individual or batch chunks
- Semantic search with similarity scores
- Optional metadata filtering
- Statistics tracking
- Document deletion

### MockRetriever

Query-aware chunk retriever.

```python
from tests.mock_services import MockRetriever

retriever = MockRetriever()
chunks = await retriever.retrieve("What about charts?", top_k=5)
context = retriever.build_context(chunks)
```

**Features:**
- Query analysis
- Chunk retrieval
- Context building
- Formatted output

### MockQAEngine

Question-answering with mock LLM responses.

```python
from tests.mock_services import MockQAEngine

qa_engine = MockQAEngine()
result = await qa_engine.answer_question(
    "What are the findings?",
    top_k=5,
    include_sources=True
)

print(result["answer"])      # String
print(result["sources"])     # List of source dicts
print(result["chunks_used"]) # Number of chunks
```

**Features:**
- Question answering
- Source citation
- Follow-up questions
- Realistic response format

## Real vs Mock Comparison

| Feature | Real | Mock |
|---------|------|------|
| Azure Services | Required ✓ | Not used ✗ |
| Credentials | Real keys | Dummy values |
| Performance | Network latency | Instant |
| Cost | $$ per request | Free |
| Vector Quality | 3072-dim semantic | Deterministic |
| Answers | GPT-4o generated | Static templates |
| Testing | Production-like | Functional |

## Example: Complete Mock Flow

```python
import asyncio
from tests.mock_services import create_mock_services
from models.document import Chunk, ChunkType

async def test_complete_flow():
    # 1. Create mock services
    services = create_mock_services()
    
    # 2. Create sample document chunks
    chunks = [
        Chunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            content=f"Content about topic {i}",
            chunk_type=ChunkType.TEXT,
            page_number=i + 1,
        )
        for i in range(5)
    ]
    
    # 3. Generate embeddings
    embedding_service = services["embedding_service"]
    chunks = await embedding_service.embed_chunks(chunks)
    
    # 4. Store in vector database
    vector_store = services["vector_store"]
    vector_store.add_chunks_batch(chunks)
    
    # 5. Ask questions
    qa_engine = services["qa_engine"]
    result = await qa_engine.answer_question("What is topic 1?")
    
    # 6. Review results
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Chunks used: {result['chunks_used']}")

# Run the test
asyncio.run(test_complete_flow())
```

## Testing API Endpoints

### Test with Python

```python
import httpx

async with httpx.AsyncClient() as client:
    # Health check
    response = await client.get("http://localhost:8000/health")
    print(response.json())
    
    # Ask question
    response = await client.post(
        "http://localhost:8000/ask",
        json={"question": "Hello?", "top_k": 5}
    )
    print(response.json())
```

### Test with JavaScript/Node

```javascript
// Ask a question
const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: 'What is the main topic?',
        top_k: 5
    })
});

const result = await response.json();
console.log(result.answer);
console.log(result.sources);
```

## Debugging

### Enable Verbose Logging

Edit `.env`:
```env
LOG_LEVEL=DEBUG
```

Then run server:
```bash
python tests/test_server.py
```

### Inspect Mock Objects

```python
from tests.mock_services import MockVectorStore

store = MockVectorStore()
stats = store.get_stats()

print(f"Chunks stored: {stats['total_chunks']}")
print(f"Dimension: {stats['embedding_dimension']}")
print(f"Metadata items: {stats['metadata_stored']}")
```

### Check Configuration

```python
from config.settings import settings
import os

print(f"Mock Mode: {os.getenv('MOCK_MODE')}")
print(f"Skip Validation: {os.getenv('SKIP_VALIDATION')}")
print(f"API Host: {settings.api.host}")
print(f"API Port: {settings.api.port}")
print(f"Chunk Size: {settings.processing.chunk_size}")
```

## Running Tests with Pytest

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test

```bash
pytest tests/test_mock_demo.py::demo_mock_embedding -v
```

### Run with coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Run and show output

```bash
pytest tests/test_mock_demo.py -v -s
```

## Integration Testing

Use mock services to test your API integrations:

```python
import pytest
from tests.mock_services import create_mock_services
from models.document import Chunk, ChunkType

@pytest.mark.asyncio
async def test_qa_workflow():
    services = create_mock_services()
    
    # Simulate user flow
    result = await services["qa_engine"].answer_question(
        "What's the summary?"
    )
    
    assert "answer" in result
    assert len(result["sources"]) > 0
    assert result["chunks_used"] > 0
```

## Transitioning to Real Azure

When ready to use real Azure services:

1. **Get credentials:**
   ```bash
   # Azure Portal → Your Resources → Keys & Endpoints
   ```

2. **Update .env:**
   ```bash
   cp .env.example .env
   # Fill in real Azure credentials
   SKIP_VALIDATION=false
   ```

3. **Run with real services:**
   ```bash
   python main.py
   ```

## Troubleshooting

### Mock mode not working
```python
# Check environment variables
import os
print(os.getenv("MOCK_MODE"))      # Should be "true"
print(os.getenv("SKIP_VALIDATION")) # Should be "true"
```

### Embeddings not consistent
```python
# Use same MockEmbeddingService instance
from tests.mock_services import MockEmbeddingService

service = MockEmbeddingService()
emb1 = await service.embed_text("Hello")
emb2 = await service.embed_text("Hello")
assert emb1 == emb2  # Should be identical
```

### Vector store empty
```python
# Remember to use same vector store instance
store = MockVectorStore()
store.add_chunk(chunk)
results = store.search(query_embedding)
# 'results' should have items from same store instance
```

## Performance Benchmarks

Mock mode performance (on typical machine):

| Operation | Time |
|-----------|------|
| Embed text | <1ms |
| Embed 100 texts | <10ms |
| Add chunk to store | <1ms |
| Search (top 5) | <1ms |
| Generate answer | <5ms |

## Next Steps

1. **Explore the API:** Visit http://localhost:8000/docs
2. **Run demo:** `python tests/test_mock_demo.py`
3. **Write tests:** Add to `tests/` directory
4. **Configure real Azure:** When credentials available
5. **Deploy:** Use Docker or cloud deployment

## Resources

- [Mock Services Code](mock_services.py)
- [Test Server](test_server.py)
- [Demo Script](test_mock_demo.py)
- [Main Documentation](../README.md)
- [API Endpoints](../api/routes.py)
