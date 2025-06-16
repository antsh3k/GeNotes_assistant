# GeNotes Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-3.0%2B-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-0.5.4-7C3AED?logo=ollama)](https://ollama.ai/)

Quick, concise information to help healthcare professionals make the right genomic decisions at each stage of a clinical pathway.

> **Note**: This is a containerized application using FastAPI for the backend and Nginx for reverse proxy. It integrates with Ollama for local LLM processing and ChromaDB for vector storage.

## Features

- **Interactive Chat Interface**: Natural language interface for querying genomic guidelines
- **Document Management**: Upload and manage clinical documents and guidelines
- **Local LLM Processing**: Uses Ollama for privacy-focused local language model processing
- **Vector Store Integration**: ChromaDB for efficient storage and retrieval of genomic knowledge
- **Containerized Architecture**: Easy deployment with Docker and Docker Compose
- **WebSocket Support**: Real-time chat capabilities
- **Session Management**: In-memory session handling for chat conversations

## Prerequisites

- Docker 20.10.0+
- Docker Compose 2.0.0+
- 8GB+ RAM (16GB recommended for optimal LLM performance)
- 10GB+ free disk space (for vector store and models)
- Ollama installed locally (for local LLM processing)
- Python 3.11+ (for development)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/GeNotes_assistant.git
   cd GeNotes_assistant
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration if needed
   ```

3. **Build and start the application**
   ```bash
   docker-compose up --build -d
   ```

4. **Download Required Language Models**
   After the services are running, you'll need to download the required language models. Run the following command:
   ```bash
   chmod +x download_models.sh  # Only needed once
   ./download_models.sh
   ```
   
   This will download the following models:
   - `nomic-embed-text` (for embeddings)
   - `mxbai-embed-large` (for embeddings)
   - `llama3.2:3b` (for chat completions)
   
   > **Note**: The download may take 10-30 minutes depending on your internet connection. The models will be stored in a Docker volume for persistence.

5. **Verify the services**
   Check that all containers are running:
   ```bash
   docker-compose ps
   ```
   
   You should see all services with a status of "healthy" or "up".

6. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Ollama API: http://localhost:11434

## Managing Models

### Checking Installed Models
To see which models are currently available in the Ollama container:

```bash
docker-compose exec ollama ollama list
# Or using the API
curl http://localhost:11434/api/tags
```

### Downloading Additional Models
To download additional models:

```bash
docker-compose exec ollama ollama pull <model-name>
```

### Common Models
- Embedding models: `nomic-embed-text`, `mxbai-embed-large`
- Chat models: `llama3:3b`, `llama3:8b`, `mistral:7b`

### Verifying Model Installation
After downloading models, you can verify they're working with:

```bash
# For embedding models
curl http://localhost:11434/api/embeddings -H "Content-Type: application/json" -d '{"model": "nomic-embed-text", "prompt": "test"}'

# For chat models
curl http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{"model": "llama3:3b", "prompt": "Hello"}'
```

## Troubleshooting

### Model Download Issues
If the `download_models.sh` script fails:

1. Check your internet connection
2. Verify the Ollama service is running:
   ```bash
   docker-compose ps | grep ollama
   ```
3. Check the Ollama logs:
   ```bash
   docker-compose logs ollama
   ```
4. Try downloading models manually using the commands in the "Managing Models" section

### Disk Space
Models can take up significant disk space (several GB each). To check disk usage:

```bash
docker system df
```

To clean up unused models and free space:

```bash
docker-compose exec ollama ollama rm <model-name>
```

### Restarting Services
If you make changes to the environment or need to restart:

```bash
docker-compose down
docker-compose up -d
```

## Development

### Backend Development
   ```bash
   # Check container status
   docker-compose ps
   
   # View logs
   docker-compose logs -f
   ```

## Development Setup

### Backend Development

1. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the backend server**
   ```bash
   cd backend
   uvicorn 3_chatbot:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Development

1. **Install dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**
   ```bash
   npm start
   ```

## Project Structure

```
GeNotes_assistant/
├── nginx/                    # Nginx reverse proxy configuration
│   └── conf.d/              # Nginx server configurations
│       └── default.conf     # Main Nginx configuration
├── backend/                 # FastAPI backend application
│   ├── 3_chatbot.py        # Main FastAPI application
│   └── 2_chunking_embedding_ingestion.py  # Document processing
├── .dockerignore            # Files to ignore in Docker builds
├── .env.example             # Example environment variables
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile.backend       # Backend Dockerfile
├── Dockerfile.nginx         # Nginx Dockerfile
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Environment Variables

Create a `.env` file based on `.env.example` with the following variables:

```env
# Backend
CHAT_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
MODEL_PROVIDER=ollama
MODEL_TEMPERATURE=0.7
DATA_DIR=./data
COLLECTION_NAME=genomic_guidelines

# Ollama
OLLAMA_HOST=ollama:11434

# Application
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
```

## API Endpoints

Once the application is running, you can access the following endpoints:

### Chat API
- `POST /api/chat` - Send a chat message
- `GET /api/chat/session/{session_id}` - Get chat session history
- `WS /ws/chat` - WebSocket endpoint for real-time chat

### Document Management
- `POST /api/scrape` - Scrape and process a website
- `POST /api/upload` - Upload and process files
- `GET /api/collections` - List available collections

### System
- `GET /` - Health check
- `GET /status` - System status and statistics

### API Documentation
- `GET /docs` - Interactive Swagger UI
- `GET /redoc` - ReDoc documentation

## Development

### Backend Development

1. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the backend server**
   ```bash
   cd backend
   uvicorn 3_chatbot:app --reload --host 0.0.0.0 --port 8000
   ```

### Testing

```bash
# Run backend tests
cd backend
pytest

# Run linters
flake8 .
mypy .
black --check .
```

### Debugging

```bash
# View logs
docker-compose logs -f backend

# Access container shell
docker-compose exec backend /bin/bash
```

## Deployment

### Production Deployment

1. **Set up environment variables**
   ```bash
   cp .env.production .env
   # Update production-specific variables
   ```

2. **Build and deploy**
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

### Kubernetes (Optional)

For production deployments, you can use the provided Kubernetes manifests in the `k8s/` directory.

## Troubleshooting

### Common Issues

1. **Ollama connection issues**
   - Verify Ollama is running: `curl http://localhost:11434/api/version`
   - Check if the model is downloaded: `ollama list`

2. **Port conflicts**
   - Ensure ports 3000 (frontend), 8000 (backend), and 11434 (Ollama) are available
   - Check with: `lsof -i :<port>` or `netstat -tuln | grep <port>`

3. **Docker resource limits**
   - Increase Docker's memory allocation in Docker Desktop settings
   - Recommended: 8GB RAM, 4 CPU cores for LLM processing

4. **Container health issues**
   - Check container logs: `docker-compose logs -f <service>`
   - Verify container health: `docker ps --filter "health=unhealthy"`

5. **Vector store issues**
   - Clear the vector store directory if needed: `rm -rf data/chromadb/*`
   - Rebuild the vector store after clearing data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- [React](https://reactjs.org/) - A JavaScript library for building user interfaces
- [Ollama](https://ollama.ai/) - Local LLM server
- [Chroma DB](https://www.trychroma.com/) - Vector database for AI applications
