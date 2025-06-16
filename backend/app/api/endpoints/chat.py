"""
Chat endpoints for the GeNotes Assistant API.
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
import re
from typing import List, Dict, Any, Optional

from app.core.chat import ChatManager
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionRequest,
    ChatSessionResponse,
    Message
)
from app.api.deps import get_chat_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for chat sessions (in production, use a database)
sessions: Dict[str, List[Dict[str, str]]] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> ChatResponse:
    """
    Handle chat messages and return responses using RAG.
    
    Args:
        chat_request: The chat request containing the message and history
        chat_manager: The chat manager instance with RAG capabilities
        
    Returns:
        ChatResponse containing the assistant's response and sources
    """
    try:
        # Generate response using ChatManager
        response = chat_manager.generate_response(
            query=chat_request.message,
            chat_history=chat_request.history
        )
        
        # Extract sources from the response content
        sources = []
        if "sources" in response:
            sources = response["sources"]
        
        # Extract sources from the response text if not already in sources
        response_text = response.get("response", "")
        if "Source:" in response_text and not sources:
            # Extract all source URLs from the response
            source_matches = re.findall(r'Source: (https?://[^\s]+)', response_text)
            if source_matches:
                sources = list(set(source_matches))  # Remove duplicates
        
        # Clean up the response text by removing source lines
        cleaned_response = re.sub(r'Source: .+\n?', '', response_text).strip()
        
        # Return the response with sources
        return ChatResponse(
            response=cleaned_response,
            sources=sources,
            metadata=response.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/chat/session", response_model=ChatSessionResponse)
async def chat_session(
    chat_request: ChatSessionRequest,
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> ChatSessionResponse:
    """
    Handle chat messages with session management.
    
    Args:
        chat_request: The chat session request
        chat_manager: The chat manager instance
        
    Returns:
        ChatSessionResponse with the assistant's response and session ID
    """
    try:
        session_id = chat_request.session_id or str(len(sessions) + 1)
        
        # Get or create session
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Add user message to session
        sessions[session_id].append({"role": "user", "content": chat_request.message})
        
        # Generate response
        response = chat_manager.generate_response(
            query=chat_request.message,
            chat_history=sessions[session_id][:-1]  # Exclude current message
        )
        
        # Add assistant response to session
        sessions[session_id].append({"role": "assistant", "content": response["response"]})
        
        return ChatSessionResponse(
            message=response["response"],
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            sources=response.get("sources", [])
        )
        
    except Exception as e:
        logger.error(f"Error in chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat session"
        )

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    chat_manager = get_chat_manager()
    session_id = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            session_id = data.get("session_id") or session_id or str(len(sessions) + 1)
            
            # Initialize session if it doesn't exist
            if session_id not in sessions:
                sessions[session_id] = []
            
            # Add user message to session
            sessions[session_id].append({"role": "user", "content": message})
            
            # Generate response
            response = chat_manager.generate_response(
                query=message,
                chat_history=sessions[session_id][:-1]  # Exclude current message
            )
            
            # Add assistant response to session
            sessions[session_id].append({"role": "assistant", "content": response["response"]})
            
            # Send response to client
            await websocket.send_json({
                "message": response["response"],
                "session_id": session_id,
                "sources": response.get("sources", []),
                "metadata": response.get("metadata", {})
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket chat: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        if websocket.client_state != "disconnected":
            await websocket.close()
