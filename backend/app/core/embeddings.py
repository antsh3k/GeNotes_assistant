"""
Embedding and vector store functionality.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

class VectorStore:
    """Wrapper around Chroma vector store."""
    
    def __init__(self, persist_directory: str, collection_name: str, embedding_model: str):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = None
        
    @property
    def client(self):
        """Lazy load the Chroma client."""
        if self._client is None:
            self._client = self._get_client()
        return self._client
    
    def _get_client(self):
        """Initialize and return the Chroma client."""
        try:
            # Create parent directories if they don't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize embeddings with base_url from environment
            base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
            embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=base_url
            )
            
            # Initialize Chroma
            return Chroma(
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
                embedding_function=embeddings
            )
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add, each with 'page_content' and 'metadata'
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        try:
            # Extract texts and metadatas
            texts = [doc['page_content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Add to Chroma
            ids = self.client.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            # Persist the data by reinitializing the client with the same collection
            # This ensures the data is written to disk
            self._client = None
            self.client
            
            logger.info(f"Added {len(ids)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            results = self.client.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.client._client.get_collection(self.collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "embedding_model": self.embedding_model,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
