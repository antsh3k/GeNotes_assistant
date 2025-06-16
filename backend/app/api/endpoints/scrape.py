"""
Scraping endpoints.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.scraping import fetch_article, save_scraped_content
from app.models.schemas import ScrapeRequest
from app.api.deps import get_datasets_dir, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/scrape")
async def scrape_website(
    request: ScrapeRequest,
    datasets_dir: Path = Depends(get_datasets_dir),
    vector_store = Depends(get_vector_store)
):
    """
    Scrape and process a website URL, then add to vector store.
    
    Args:
        request: ScrapeRequest containing the URL to scrape
        datasets_dir: Directory to save scraped content
        vector_store: Vector store instance for document storage
        
    Returns:
        Dictionary with scraping and indexing results
    """
    try:
        if not request.url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL is required"
            )
        
        # Check if URL is valid
        if not request.url.startswith(('http://', 'https://')):
            request.url = 'https://' + request.url
        
        # Fetch and process the article
        result = fetch_article(request.url)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["content"]
            )
        
        # Save the scraped content
        saved_path = save_scraped_content(result, datasets_dir)
        
        if not saved_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save scraped content"
            )
        
        # Create document for vector store
        doc = Document(
            page_content=result["content"],
            metadata={
                "source": request.url,
                "title": result["title"],
                "type": "scraped_web",
                "saved_path": str(saved_path)
            }
        )
        
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents([doc])
        
        # Add to vector store
        try:
            # Get the Chroma client
            chroma_client = vector_store.client
            
            # Add documents to the vector store
            chroma_client.add_documents(chunks)
            
            # Persist the vector store
            chroma_client.persist()
            
            return {
                "status": "success",
                "message": f"Successfully scraped and indexed {request.url}",
                "title": result["title"],
                "content_length": len(result["content"]),
                "saved_path": str(saved_path),
                "documents_processed": len(chunks),
                "vector_store_size": chroma_client._collection.count()
            }
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return {
                "status": "partial_success",
                "message": f"Scraping succeeded but failed to index: {str(e)}",
                "title": result["title"],
                "content_length": len(result["content"]),
                "saved_path": str(saved_path),
                "documents_processed": 0
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scraping website: {str(e)}"
        )
