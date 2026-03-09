"""
Getting Started Guide for the Multimodal RAG System.

Quick reference for setup and common operations.
"""

# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Azure Credentials
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

Required environment variables:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
```

### 3. Start the Server
```bash
python main.py
```

Server runs at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

## First Example

### Upload a PDF
```bash
curl -X POST -F "file=@sample.pdf" http://localhost:8000/upload_pdf
```

Response:
```json
{
  "success": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample.pdf",
  "total_pages": 10,
  "chunks_created": 45
}
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "top_k": 5
  }'
```

Response:
```json
{
  "question": "What are the main findings?",
  "answer": "Based on the document, the main findings include...",
  "sources": [
    {
      "document": "sample.pdf",
      "page": 3,
      "type": "text"
    }
  ],
  "chunks_used": 2
}
```

## Command Cheat Sheet

| Action | Command |
|--------|---------|
| Start server | `python main.py` |
| Run tests | `pytest tests.py -v` |
| Health check | `curl http://localhost:8000/health` |
| Get stats | `curl http://localhost:8000/stats` |
| Delete document | `curl -X DELETE http://localhost:8000/delete_document/{id}` |

## Project Structure

```
DocIntelRag/
├── config/                  # Settings and configuration
│   └── settings.py
├── ingestion/              # PDF loading and extraction
│   ├── pdf_loader.py
│   ├── document_intelligence_extractor.py
│   ├── image_extractor.py
│   └── pipeline.py
├── processing/             # Content analysis and chunking
│   ├── text_processor.py
│   ├── table_parser.py
│   ├── chart_analyzer.py
│   └── chunking.py
├── embeddings/             # Vector embeddings
│   └── embedding_service.py
├── vectordb/               # Vector storage
│   └── vector_store.py
├── rag/                    # Retrieval and Q&A
│   ├── retriever.py
│   └── qa_engine.py
├── api/                    # HTTP API
│   ├── routes.py
│   └── schemas.py
├── models/                 # Data models
│   └── document.py
├── utils/                  # Utilities
│   └── logger.py
├── main.py                 # Entry point
├── example_usage.py        # Usage example
├── tests.py               # Unit tests
├── requirements.txt        # Dependencies
├── .env.example           # Configuration template
├── Dockerfile             # Container setup
├── docker-compose.yml     # Docker compose
├── README.md              # Full documentation
└── ARCHITECTURE.md        # Design documentation
```

## Common Issues

### "Connection refused" error
- Verify `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` is correct
- Check your Azure credentials are valid
- Ensure your Azure resource is in the correct region

### "No embeddings returned"
- Verify Azure OpenAI endpoint and key in `.env`
- Confirm `gpt-4o` and `text-embedding-3-large` models are deployed
- Check API quota hasn't been exceeded

### "Permission denied" for data directory
```bash
mkdir -p data
chmod 755 data
```

### FAISS index not found (first time)
- This is normal on first run, it will be created automatically
- Just upload your first PDF

## Performance Tips

### For Large Documents
1. Increase `CHUNK_SIZE` in .env for faster processing
2. Reduce `top_k` in queries to speed up retrieval
3. Use `document_id` filter if querying specific documents

### For Many Documents
1. Use batch operations where available
2. Implement caching for repeated queries
3. Consider distributed deployment with load balancing

## Development Workflow

### Make Code Changes
```bash
# Edit any module
# Changes are reflected immediately in tests

# Run tests
pytest tests.py -v

# Or test with live API
python main.py  # In one terminal
# Make requests in another
```

### Add New Features
1. Extend classes in relevant module
2. Add type hints and docstrings
3. Write unit tests
4. Update API routes if needed
5. Update this guide if user-facing

## Deployment

### Local Testing
```bash
python main.py
# API at http://localhost:8000
```

### Docker
```bash
docker build -t docintelsrag .
docker run -p 8000:8000 --env-file .env docintelsrag
```

### Cloud Deployment (Azure)
```bash
az containerapp up --name docintelsrag \
  --resource-group mygroup \
  --environment myenv \
  --image docintelsrag:latest
```

## Next Steps

1. **Try the example**: Read [example_usage.py](example_usage.py)
2. **Understand the architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Explore API docs**: Visit http://localhost:8000/docs
4. **Run tests**: Execute `pytest tests.py -v`
5. **Customize**: Modify `config/settings.py` and components as needed

## Getting Help

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Code Documentation**: Inline docstrings in each module
- **Examples**: See `example_usage.py` and endpoint tests
- **Troubleshooting**: Check logs with `LOG_LEVEL=DEBUG`

## Production Checklist

- [ ] Set all environment variables
- [ ] Configure HTTPS/TLS
- [ ] Set up logging aggregation
- [ ] Enable authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Test error scenarios
- [ ] Document any customizations
- [ ] Set up backup procedures
- [ ] Configure scaling policies
