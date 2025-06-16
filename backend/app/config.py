"""
Application configuration settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory - use absolute path from environment or default to /app in container
BASE_DIR = Path(os.getenv("APP_HOME", "/app"))

# Application settings
APP_NAME = "GeNotes API"
APP_VERSION = "0.1.0"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# API settings
API_PREFIX = "/api"
DOCS_URL = "/docs"
REDOC_URL = "/redoc"

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/genotes.db")

# Ensure data directories exist
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Vector store settings
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chromadb"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_data")

# Dataset settings
DATASETS_PATH = os.getenv("DATASETS_PATH", str(DATA_DIR / "datasets"))

# Create directories if they don't exist
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(DATASETS_PATH).mkdir(parents=True, exist_ok=True)

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# CORS settings
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", 
    "*"  # In production, replace with specific origins
).split(",")

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "app": {
            "handlers": ["console"],
            "level": "DEBUG" if DEBUG else "INFO",
            "propagate": False,
        },
    },
}
