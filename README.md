# GeNotes Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-3.0%2B-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react)](https://reactjs.org/)

Quick, concise information to help healthcare professionals make the right genomic decisions at each stage of a clinical pathway.

## Features

- **Interactive Chat Interface**: Natural language interface for querying genomic guidelines
- **Document Management**: Upload and manage clinical documents and guidelines
- **Vector Store Integration**: Store and retrieve genomic knowledge efficiently
- **Secure & Scalable**: Containerized architecture with security best practices
- **Responsive Design**: Works on desktop and tablet devices

## Prerequisites

- Docker 20.10.0+
- Docker Compose 2.0.0+
- 8GB+ RAM (16GB recommended for optimal performance)
- 10GB+ free disk space

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/GeNotes_assistant.git
   cd GeNotes_assistant
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. **Build and start the application**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the application**
   - Frontend: https://localhost
   - Backend API: https://localhost/api
   - Ollama API: http://localhost:11434

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
   cd simple_streamlitt_app
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
├── .github/                  # GitHub workflows and issue templates
├── frontend/                  # React frontend application
│   ├── public/               # Static files
│   └── src/                  # React source code
│       ├── components/       # Reusable UI components
│       ├── pages/            # Page components
│       └── services/         # API services
├── nginx/                    # Nginx configuration
│   ├── conf.d/              # Server configurations
│   ├── ssl/                 # SSL certificates
│   └── nginx.conf           # Main Nginx config
├── simple_streamlitt_app/    # FastAPI backend
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality
│   ├── models/              # Database models
│   └── services/            # Business logic
├── .dockerignore            # Files to ignore in Docker builds
├── .env.example             # Example environment variables
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile.backend        # Backend Dockerfile
├── Dockerfile.frontend       # Frontend Dockerfile
└── README.md                # This file
```

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: https://localhost/api/docs
- ReDoc: https://localhost/api/redoc

## Testing

### Run tests

```bash
# Backend tests
cd simple_streamlitt_app
pytest

# Frontend tests
cd frontend
npm test
```

### Linting

```bash
# Backend
flake8 .
mypy .
black --check .

# Frontend
cd frontend
npm run lint
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

1. **Port conflicts**
   - Ensure ports 80, 443, 8000, and 3000 are not in use
   - Check with: `lsof -i :<port>`

2. **Docker resource limits**
   - Increase Docker's memory allocation in Docker Desktop settings
   - Recommended: 8GB RAM, 4 CPU cores

3. **SSL certificate errors**
   - For local development, accept the self-signed certificate
   - Or add the certificate to your trusted store

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
