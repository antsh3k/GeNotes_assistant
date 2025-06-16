# GeNotes Backend

A FastAPI-based backend for the GeNotes Genomic Guidelines Assistant.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app initialization and routing
│   ├── config.py           # Configuration and environment variables
│   ├── models/             # Pydantic models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── api/                # API routes
│   │   ├── __init__.py
│   │   ├── deps.py         # Dependencies
│   │   └── endpoints/      # Route handlers
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       ├── scrape.py
│   │       └── upload.py
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   ├── scraping.py
│   │   ├── embeddings.py
│   │   └── chat.py
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   └── scrape_service.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── logger.py
├── tests/                  # Test files
├── .env                   # Environment variables
├── requirements.txt       # Dependencies
└── run.py                # Application entry point
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd genotes/backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Copy the example environment file and update the values:
   ```bash
   cp .env.example .env
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

## API Endpoints

### Chat
- `POST /api/chat` - Send a chat message
- `POST /api/chat/session` - Chat with session management
- `GET /api/chat/ws` - WebSocket endpoint for real-time chat

### Scraping
- `POST /api/scrape` - Scrape content from a URL

### File Upload
- `POST /api/upload` - Upload and process files

### System
- `GET /health` - Health check
- `GET /` - API root

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
isort .
```

### Linting
```bash
flake8
mypy .
```

## Deployment

### Docker
```bash
docker build -t genotes-backend .
docker run -p 8000:8000 genotes-backend
```

### Production
For production deployments, it's recommended to use:
- Gunicorn with Uvicorn workers
- A reverse proxy like Nginx
- Process manager like systemd or Supervisor

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `DATABASE_URL` | Database connection URL | `sqlite:///./genotes.db` |
| `CHROMA_DB_PATH` | Path to Chroma DB | `./data/chromadb` |
| `COLLECTION_NAME` | Chroma collection name | `rag_data` |
| `DATASETS_PATH` | Path to store datasets | `./data/datasets` |
| `EMBEDDING_MODEL` | Embedding model name | `mxbai-embed-large` |
| `CHAT_MODEL` | Chat model name | `llama3.2:3b` |
| `MODEL_PROVIDER` | Model provider | `ollama` |
| `OLLAMA_API_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `SECRET_KEY` | Secret key for JWT | `your-secret-key-here` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiry | `30` |

## License

MIT
