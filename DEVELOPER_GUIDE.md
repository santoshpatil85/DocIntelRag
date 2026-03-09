"""
Developer Guide for extending the Multimodal RAG system.

Instructions for adding features, customizing components, and best practices.
"""

# Developer Guide

## Adding New Features

### 1. Add a New Chunk Type

#### Step 1: Update the Enum
File: `models/document.py`
```python
class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"
    CODE = "code"  # NEW
```

#### Step 2: Add Processing Logic
File: `processing/chunking.py`
```python
def chunk_document(self, document: Document) -> List[Chunk]:
    # ... existing code ...
    
    # Add code chunk processing
    if hasattr(page, 'code_blocks'):
        for code_block in page.code_blocks:
            chunk = Chunk(
                chunk_id=f"{document.document_id}_code_{chunk_counter}",
                document_id=document.document_id,
                content=code_block,
                chunk_type=ChunkType.CODE,
                page_number=page_num,
                metadata={"language": "python"}
            )
            chunks.append(chunk)
```

#### Step 3: Update Retrieval (if needed)
File: `rag/retriever.py`
```python
def _analyze_query(self, query: str) -> Optional[str]:
    code_keywords = ["code", "function", "method", "class", "implementation"]
    # Add code keyword detection
    if sum(1 for kw in code_keywords if kw in query_lower) > 0:
        return ChunkType.CODE.value
```

### 2. Add a New Data Source

#### Example: CSV File Support

File: `ingestion/csv_loader.py` (NEW)
```python
import pandas as pd
from models.document import Document, DocumentPage
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CSVLoader:
    """Loads CSV files and creates document representation."""
    
    async def load_csv(self, csv_path: str, document_id: str) -> Document:
        """Load and parse CSV file."""
        try:
            df = pd.read_csv(csv_path)
            
            # Create document with CSV data as table chunk
            page = DocumentPage(
                page_number=1,
                text=f"CSV Data: {Path(csv_path).name}",
                tables=[{
                    "rows": len(df),
                    "columns": len(df.columns),
                    "data": df.values.tolist()
                }]
            )
            
            return Document(
                document_id=document_id,
                filename=Path(csv_path).name,
                pages=[page],
                metadata={"source": "csv"}
            )
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            return None
```

Then update pipeline: `ingestion/pipeline.py`
```python
from ingestion.csv_loader import CSVLoader

class PDFIngestionPipeline:
    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.csv_loader = CSVLoader()  # NEW
        # ...
```

### 3. Add a Custom Retriever Mode

File: `rag/retriever.py`
```python
class HybridRetriever(LayoutAwareRetriever):
    """Combines vector search with keyword matching."""
    
    async def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,  # Weight for vector vs keyword
    ) -> List[Chunk]:
        """
        Hybrid search combining semantic + keyword similarity.
        
        Args:
            query: User query
            top_k: Number of results
            alpha: Weight for vector search (1-alpha for keyword)
        """
        # Vector search
        vector_results = await self.retrieve(query, top_k=top_k)
        
        # Keyword search (simple BM25 approximation)
        keywords = query.lower().split()
        keyword_results = []
        for chunk in all_chunks:  # Iterate all chunks
            score = sum(1 for kw in keywords if kw in chunk.content.lower())
            if score > 0:
                keyword_results.append((chunk, score / len(keywords)))
        
        # Combine results
        # ...implementation...
        
        return combined_results
```

### 4. Add a New LLM Model

Currently: GPT-4o
Example: Add Claude support

File: `rag/qa_engine.py`
```python
class HybridQAEngine:
    """Support multiple LLM backends."""
    
    def __init__(
        self,
        retriever: LayoutAwareRetriever,
        model_choice: str = "gpt4o",  # or "claude"
    ):
        if model_choice == "gpt4o":
            self.client = AsyncAzureOpenAI(...)
            self.model = "gpt-4o"
        elif model_choice == "claude":
            import anthropic
            self.client = anthropic.Anthropic(...)
            self.model = "claude-3-opus"
        # ...
    
    async def _generate_answer(self, question: str, context: str) -> str:
        if self.model.startswith("gpt"):
            # Existing GPT implementation
            return await self._generate_gpt_answer(question, context)
        else:
            return await self._generate_claude_answer(question, context)
```

## Performance Optimization

### 1. Batch Processing

Already implemented in:
- `EmbeddingService.embed_texts_batch()`
- `FAISSVectorStore.add_chunks_batch()`

Example of extending:

```python
# Chart analysis batching
async def analyze_charts_batch(
    self,
    charts: List[Tuple[bytes, str]],
    batch_size: int = 5,
) -> List[ChartAnalysis]:
    """Process charts in batches to manage API calls."""
    results = []
    for i in range(0, len(charts), batch_size):
        batch = charts[i:i+batch_size]
        batch_results = await asyncio.gather(
            *[self.analyze_chart(img, ctx) for img, ctx in batch]
        )
        results.extend(batch_results)
        await asyncio.sleep(1)  # Rate limiting
    return results
```

### 2. Caching

Add to `rag/retriever.py`:
```python
from functools import lru_cache
import hashlib

class CachingRetriever(LayoutAwareRetriever):
    def __init__(self, *args, cache_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size
    
    @lru_cache(maxsize=128)
    def _query_hash(self, query: str) -> str:
        """Hash query for caching."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve with caching."""
        query_hash = self._query_hash(query)
        
        if query_hash in self.cache:
            return self.cache[query_hash]
        
        chunks = await super().retrieve(query, top_k)
        self.cache[query_hash] = chunks
        
        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        return chunks
```

### 3. Parallel Page Processing

File: `ingestion/pdf_loader.py`
```python
from concurrent.futures import ProcessPoolExecutor

class HighPerformancePDFLoader(PDFLoader):
    def __init__(self, max_workers: int = 8):
        super().__init__()
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def extract_pages_parallel(self, pdf_path: str) -> List[DocumentPage]:
        """Extract pages with process pool."""
        doc_id, doc = self.load_pdf(pdf_path)
        
        # Use process pool for CPU-bound extraction
        pages = list(self.executor.map(
            self._extract_page_sync,
            [(doc[i], i+1) for i in range(len(doc))]
        ))
        
        doc.close()
        return pages
```

## Testing Best Practices

### Unit Test Examples

```python
# tests.py

@pytest.fixture
def sample_chunk():
    """Fixture for chunk testing."""
    return Chunk(
        chunk_id="test_001",
        document_id="doc_001",
        content="Test content",
        chunk_type=ChunkType.TEXT,
        page_number=1,
        embedding=[0.1] * 3072
    )

def test_chunk_serialization(sample_chunk):
    """Test chunk serialization round-trip."""
    chunk_dict = sample_chunk.to_dict()
    restored = Chunk.from_dict(chunk_dict)
    assert restored.chunk_id == sample_chunk.chunk_id

@pytest.mark.asyncio
async def test_qa_engine_with_mock_retriever(sample_chunk):
    """Test QA engine with mocked retriever."""
    
    class MockRetriever:
        async def retrieve(self, query: str, top_k: int):
            return [sample_chunk]
        
        def build_context(self, chunks):
            return "Mock context"
    
    qa_engine = RAGQAEngine(MockRetriever())
    # Test...
```

### Integration Tests

```python
@pytest.mark.integration
async def test_full_pipeline(sample_pdf_path):
    """Test complete PDF to Q&A flow."""
    pipeline = PDFIngestionPipeline()
    document, chunks = await pipeline.process_pdf(sample_pdf_path, "test_001")
    
    assert document is not None
    assert len(chunks) > 0
    assert all(c.embedding is None for c in chunks)  # Not embedded yet
```

## Code Style Guidelines

### Type Hints
```python
# Good
async def process_chunks(
    chunks: List[Chunk],
    batch_size: int = 32,
) -> Dict[str, int]:
    """Process chunks and return statistics."""
    pass

# Avoid
def process_chunks(chunks, batch_size=32):
    pass
```

### Docstrings
```python
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
    
    Raises:
        ValueError: If query_embedding dimension mis matches.
    """
```

### Error Handling
```python
def operation(self, data):
    """Perform operation with proper error handling."""
    try:
        result = self._do_operation(data)
        logger.info("Operation successful")
        return result
    except SpecificError as e:
        logger.warning(f"Expected error: {str(e)}")
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError(f"Operation failed: {str(e)}")
```

## Debugging Tips

### Enable Debug Logging
```bash
LOG_LEVEL=DEBUG python main.py
```

### Add Debug Prints
```python
from utils.logger import log_function_call, log_function_result

async def complex_operation(self, data):
    start = time.time()
    log_function_call(logger, "complex_operation", data_size=len(data))
    
    result = await self._process(data)
    
    duration = time.time() - start
    log_function_result(logger, "complex_operation", duration, f"Processed {len(result)} items")
    
    return result
```

### Inspect Vector Store
```python
# In Python shell
from vectordb.vector_store import FAISSVectorStore

store = FAISSVectorStore()
stats = store.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Metadata items: {stats['metadata_stored']}")

# Search test
test_embedding = [0.1] * 3072
results = store.search(test_embedding, top_k=5)
for chunk, score in results:
    print(f"Page {chunk.page_number}: similarity={score:.4f}")
```

## Deployment Customization

### Custom Docker Image
```dockerfile
# Dockerfile.custom
FROM docintelsrag:latest

# Add custom dependencies
RUN pip install custom-package

# Override configuration
ENV CHUNK_SIZE=1024
ENV LOG_LEVEL=DEBUG

# Copy custom files
COPY custom_processors/ /app/custom_processors/
```

### Environment-Specific Settings
```python
# config/settings.py (hybrid approach)

class Settings:
    def __init__(self, env: str = "production"):
        if env == "development":
            self.api.log_level = "DEBUG"
            self.processing.chunk_size = 256  # Smaller for testing
        elif env == "production":
            self.api.log_level = "INFO"
            self.processing.chunk_size = 512
```

## Contributing Checklist

Before submitting changes:

- [ ] Code follows style guidelines
- [ ] Type hints for all functions
- [ ] Docstrings for public methods
- [ ] Unit tests added/updated
- [ ] All tests pass: `pytest tests.py -v`
- [ ] No hardcoded secrets
- [ ] Logging at appropriate levels
- [ ] Error handling implemented
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
