"""
Dependencies for API endpoints.
"""
from fastapi import Depends, HTTPException, status
from typing import Optional, List
from pathlib import Path
from langchain.agents import Tool
from langchain.schema import Document

from app.core.embeddings import VectorStore
from app.core.chat import ChatManager
from app.config import (
    EMBEDDING_MODEL,
    CHAT_MODEL,
    MODEL_PROVIDER,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DATASETS_PATH,
    OLLAMA_API_BASE_URL
)

# Initialize vector store
vector_store = VectorStore(
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_model=EMBEDDING_MODEL
)

def get_tools(vector_store) -> List[Tool]:
    """Initialize and return the tools for the agent."""
    
    def retrieve_docs(query: str) -> str:
        """Retrieve documents relevant to the query."""
        try:
            # Get the Chroma client
            chroma_client = vector_store.client
            
            # Perform similarity search
            docs = chroma_client.similarity_search(query, k=3)
            
            # Format results
            if not docs:
                return "No relevant documents found."
                
            result = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Document')
                content = doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                result.append(f"[{i}] {source}: {content}")
                
            return "\n\n".join(result)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    # Define tools
    tools = [
        Tool(
            name="retrieve_documents",
            func=retrieve_docs,
            description="Useful for retrieving relevant documents from the knowledge base. "
                      "Input should be a question or topic to search for."
        )
    ]
    
    return tools

# Initialize chat manager with RAG capabilities
chat_manager = ChatManager(
    model_name=CHAT_MODEL,
    embedding_model=EMBEDDING_MODEL,
    vector_store=vector_store
)

def get_vector_store() -> VectorStore:
    """Get the vector store instance."""
    return vector_store

def get_chat_manager() -> ChatManager:
    """Get the chat manager instance."""
    return chat_manager

def get_datasets_dir() -> Path:
    """Get the datasets directory path."""
    path = Path(DATASETS_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path
