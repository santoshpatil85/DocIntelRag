# Quick Start Guide

Your Multimodal RAG system is ready to use! Choose from these options:

## 1️⃣ Start Mock API Server (Recommended for Testing)

```bash
python tests/test_server.py
```

Then visit these endpoints:
- **Interactive API docs:** http://localhost:8000/docs
- **Health check:** curl http://localhost:8000/health
- **Ask question:** curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is this about?"}'

## 2️⃣ Run Interactive Demo

See all mock services in action:

```bash
python tests/test_mock_demo.py
```

Output includes:
- Embedding generation & dimension validation
- Vector store operations (add, search, filter)
- Retrieved context formatting
- Q&A engine responses

## 3️⃣ Run Validation Tests

Verify everything works:

```bash
python validate.py
```

Expected output: `5/5 tests passed`

## 4️⃣ Run Full Pytest Suite

```bash
pytest tests/ -v
```

## 🔧 Configuration

The `.env` file contains mock credentials for testing:

```env
MOCK_MODE=true          # Use mock services (no Azure required)
SKIP_VALIDATION=true    # Skip credential validation
LOG_LEVEL=DEBUG         # Verbose logging
```

## 📚 What Each File Does

| File | Purpose |
|------|---------|
| `validate.py` | One-command verification of all services |
| `tests/mock_services.py` | Mock implementations (MockEmbeddingService, MockVectorStore, etc.) |
| `tests/test_mock_demo.py` | Interactive demonstration of each service |
| `tests/test_server.py` | Standalone FastAPI server with mock endpoints |
| `TESTING.md` | Comprehensive testing guide |

## 🚀 Next Steps

1. **Explore the API:** Start the server and visit http://localhost:8000/docs
2. **Run tests:** `python validate.py` to verify installation
3. **Test demo:** `python tests/test_mock_demo.py` for interactive walkthrough
4. **Review docs:** See [TESTING.md](TESTING.md) for detailed testing workflows

## 🔄 Switching to Real Azure

When ready to use real Azure services:

1. Update `.env` with real credentials:
   ```env
   SKIP_VALIDATION=false
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your-real-key
   AZURE_OPENAI_KEY=your-real-key
   # ... other real Azure settings
   ```

2. Start the main server:
   ```bash
   python main.py
   ```

## ❓ Common Questions

**Q: Do I need Azure credentials right now?**  
A: No! Mock mode works without any credentials. Credentials only needed when SKIP_VALIDATION=false.

**Q: Can I test the API?**  
A: Yes! Run `python tests/test_server.py` and visit http://localhost:8000/docs

**Q: How do I verify everything works?**  
A: Run `python validate.py` - it tests all components and takes ~5 seconds.

**Q: Can I integrate with my own code?**  
A: Yes! See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for integration examples.

## 📋 Project Structure

```
/workspaces/DocIntelRag/
├── main.py                 # Production server entrypoint
├── validate.py             # Quick validation script
├── .env                    # Configuration (mock credentials)
├── requirements.txt        # Python dependencies
├── config/                 # Configuration modules
├── ingestion/              # PDF processing
├── processing/             # Text & table processing
├── embeddings/             # Embedding services
├── vectordb/               # Vector store
├── rag/                    # Retrieval & QA
├── api/                    # FastAPI routes
├── models/                 # Data models
├── utils/                  # Utilities (logging, etc)
└── tests/                  # Mock services & tests
    ├── mock_services.py    # Mock implementations
    ├── test_mock_demo.py   # Interactive demo
    └── test_server.py      # Standalone test server
```

## 🎯 System Ready!

✓ All dependencies installed  
✓ Configuration validated  
✓ Mock services implemented  
✓ API endpoints defined  
✓ Testing infrastructure ready  

**You can start using the system right now!**
