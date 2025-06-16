"""
Main FastAPI application.
"""
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
    API_PREFIX,
    DOCS_URL,
    REDOC_URL,
    CORS_ORIGINS,
    LOGGING_CONFIG
)
from app.api.endpoints import chat, scrape, upload

# Configure logging
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    docs_url=DOCS_URL if DEBUG else None,
    redoc_url=REDOC_URL if DEBUG else None,
    debug=DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix=API_PREFIX, tags=["chat"])
app.include_router(scrape.router, prefix=API_PREFIX, tags=["scrape"])
app.include_router(upload.router, prefix=API_PREFIX, tags=["upload"])

# Health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION
    }

# Root endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def read_root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {APP_NAME} API",
        "version": APP_VERSION,
        "docs": f"{DOCS_URL}" if DOCS_URL else None
    }

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"},
    )

# Application lifespan events
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Debug mode: {'ON' if DEBUG else 'OFF'}")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")
