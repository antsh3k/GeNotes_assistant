"""
Pydantic models for request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    """Represents a chat message."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: List[Dict[str, str]] = []
    metadata: Dict[str, Any] = {}

class ScrapeRequest(BaseModel):
    """Request model for website scraping."""
    url: str
    source_type: str = "website"

class CollectionInfo(BaseModel):
    """Information about a vector collection."""
    name: str
    document_count: int
    embeddings: str
    is_active: bool
    created_at: str
    updated_at: str

class ChatSessionRequest(BaseModel):
    """Request model for chat session."""
    message: str
    session_id: Optional[str] = None
    stream: bool = False

class ChatSessionResponse(BaseModel):
    """Response model for chat session."""
    message: str
    session_id: str
    timestamp: str
    sources: List[Dict[str, str]] = []

class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str
    messages: List[Dict[str, str]]
    created_at: str
    updated_at: str
