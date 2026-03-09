"""
Architecture documentation for the Multimodal RAG system.

Provides detailed technical specifications and design decisions.
"""

# System Architecture

## Component Overview

### 1. Ingestion Layer
- **PDFLoader**: Extracts text and images from PDFs using PyMuPDF
- **DocumentIntelligenceExtractor**: Leverages Azure Document Intelligence for advanced OCR and layout analysis
- **ImageExtractor**: Isolates and processes images/charts from document

### 2. Processing Layer
- **TextProcessor**: Cleans, normalizes, and analyzes text content
- **TableParser**: Converts tables to DataFrames and generates natural language summaries
- **ChartAnalyzer**: Uses GPT-4o Vision to interpret visual elements
- **HierarchicalChunker**: Creates balanced chunks from multimodal content

### 3. Embedding Layer
- **EmbeddingService**: Generates embeddings using Azure OpenAI's text-embedding-3-large

### 4. Vector Storage Layer
- **FAISSVectorStore**: Implements semantic search using FAISS IndexFlatL2
  - In-memory vector index for fast similarity search
  - Persistent storage to disk (JSON metadata + binary index)

### 5. Retrieval Layer
- **LayoutAwareRetriever**: Performs semantic search with content-type prioritization
  - Analyzes query keywords to detect preferred chunk types
  - Retrieves most similar chunks with optional filtering

### 6. QA Layer
- **RAGQAEngine**: Orchestrates RAG pipeline
  - Retrieves relevant chunks
  - Constructs context with proper formatting
  - Calls GPT-4o for answer generation
  - Extracts and returns source citations

### 7. API Layer
- **FastAPI Application**: RESTful interface
  - Multipart file upload for PDFs
  - JSON request/response with Pydantic validation
  - Async/await pattern for non-blocking I/O

## Data Flow

```
User Input (PDF)
    ↓
[PDF Loading]
    ├─ Extract pages
    ├─ Extract text per page
    └─ Extract images
    ↓
[Document Intelligence]
    ├─ Layout analysis
    ├─ Table detection
    ├─ OCR results
    └─ Bounding boxes
    ↓
[Content Processing]
    ├─ Text cleaning → TextProcessor
    ├─ Table parsing → TableParser
    └─ Image analysis → ChartAnalyzer (GPT-4o Vision)
    ↓
[Hierarchical Chunking]
    ├─ Text chunks (512 tokens, 64 overlap)
    ├─ Table chunks (with summaries)
    └─ Chart chunks (with GPT analysis)
    ↓
[Embedding Generation]
    └─ embed_text_batch() via Azure OpenAI
    ↓
[Vector Storage]
    ├─ FAISS IndexFlatL2
    └─ Metadata JSON
    ↓
[Query]
    ↓
[Retrieval]
    ├─ Generate query embedding
    ├─ Vector similarity search
    └─ Apply metadata filters
    ↓
[Context Building]
    └─ Format chunks for LLM
    ↓
[QA Generation]
    └─ GPT-4o with retrieved context
    ↓
[Response with Citations]
```

## Key Design Decisions

### 1. Chunk Strategy
- **Hierarchical**: Different handling for text, tables, charts
- **Sliding Window**: Overlap preserves context between chunks
- **Content-Aware**: Metadata enables filtering and prioritization

### 2. Vector Search
- **L2 Distance**: Suitable for normalized embeddings
- **FAISS IndexFlatL2**: Simple, fast, deterministic (no approximation)
- **Query-Aware Filtering**: Prioritize relevant content types

### 3. Async Processing
- **Non-blocking I/O**: PDFLoader, EmbeddingService support async
- **Batch Operations**: Embedding generation batches texts for efficiency
- **Concurrent Tasks**: Gallery of image analysis can run in parallel

### 4. Metadata Storage
- **JSON Serialization**: Human-readable, easy to debug
- **Persistent Storage**: FAISS index + metadata saved to disk
- **Reconstruction**: Metadata enables rebuilding chunks

## Scalability Considerations

### Current Limitations
- FAISS IndexFlatL2 is in-memory (suitable for ~1M chunks)
- Single-machine deployment (no distributed processing)
- Synchronous table parsing and chart analysis

### Recommended Improvements
- Use FAISS IndexIVF for larger datasets (> 1M vectors)
- Implement distributed embeddings via message queue (e.g., Celery)
- Multi-threaded page extraction with ProcessPoolExecutor
- Batch chart analysis with async gathering

### Production Scaling
1. **Horizontal**: Deploy multiple API instances behind load balancer
2. **Vertical**: Increase chunk batch size, embedding workers
3. **Caching**: Cache embeddings for repeated queries
4. **Hybrid Search**: Combine vector + keyword search (BM25)

## Error Handling Strategy

### Component-Level
- Try-catch in every operation
- Graceful degradation (return None, empty list)
- Detailed logging for debugging

### API-Level
- HTTP 500 for server errors
- Validation errors return 422
- Health checks detect service issues

### Retry Logic
- Document Intelligence: Relies on Azure SDK retry mechanism
- Embeddings: Batch processing with partial failures handled
- Vector Store: Read-only operations safe, writes are transactional

## Security Considerations

### Credentials Management
- Azure keys in environment variables (never hardcoded)
- Use azure-identity for managed identity in Azure
- HTTPS for API endpoints in production

### Data Privacy
- PDF files processed in-memory (not persisted)
- Metadata stored locally (deploy securely)
- No logging of PII or document content

### Input Validation
- All Pydantic schemas validate input
- File size limits recommended
- Query length capped to prevent abuse

## Testing Strategy

### Unit Tests
- Data models: Serialization, deserialization
- Text processing: Cleaning, chunking
- Vector store: Add, search, metadata

### Integration Tests
- Full pipeline: PDF → embeddings → Q&A
- Error cases: Invalid PDFs, missing credentials
- Performance benchmarks and load testing

### E2E Testing
- Sample PDF upload and query
- Multi-page document handling
- Complex tables and charts

## Monitoring & Observability

### Metrics to Track
- API latency (by endpoint)
- Embeddings generated per minute
- Vector store size
- Failed operations (with reasons)

### Logging Levels
- DEBUG: Function calls, parameter values
- INFO: Major operations (PDF loaded, chunks created)
- WARNING: Non-critical failures (missing image OCR)
- ERROR: Critical failures (API errors)

### Debugging Aids
- Structured logging with context (doc_id, chunk_count)
- Function call timing for performance analysis
- Full stack traces on exceptions

## Future Enhancements

### Phase 2: Advanced Retrieval
- Hybrid search (vector + BM25 keyword)
- Multi-hop reasoning queries
- Query expansion and refinement
- Metadata-based faceted search

### Phase 3: Performance
- GPU-accelerated embeddings
- FAISS GPU indexes
- Caching and CDN for common queries
- Quantization for faster search

### Phase 4: Advanced NLP
- Named entity recognition
- Relation extraction
- Document summarization
- Multi-language support
