"""
File upload endpoints.
"""
import logging
import mimetypes
import json
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from pathlib import Path
from typing import List, Dict, Any, Generator

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)

from app.api.deps import get_datasets_dir, get_vector_store

# Configure mimetypes for additional file types
mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
mimetypes.add_type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx')
mimetypes.add_type('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx')

logger = logging.getLogger(__name__)

router = APIRouter()

def process_json_lines(file_path: Path) -> List[Dict[str, Any]]:
    """Process each JSON line and extract relevant information."""
    extracted = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not all(key in obj for key in ['url', 'title', 'content']):
                        logger.warning(f"Skipping line - missing required fields: {line[:100]}...")
                        continue
                    extracted.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON line: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise
    return extracted

def create_documents_from_json(data: List[Dict[str, Any]], text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Create LangChain documents from JSON data with proper metadata."""
    documents = []
    for item in data:
        try:
            # Split the content into chunks
            chunks = text_splitter.create_documents(
                [item['content']],
                metadatas=[{"source": item['url'], "title": item['title']}]
            )
            documents.extend(chunks)
        except Exception as e:
            logger.warning(f"Error processing item {item.get('url', 'unknown')}: {str(e)}")
            continue
    return documents

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    datasets_dir: Path = Depends(get_datasets_dir),
    vector_store = Depends(get_vector_store)
):
    """
    Upload, process JSONL files, and add them to the vector store.
    
    Args:
        files: List of JSONL files to upload (each line should be a JSON object with url, title, and content)
        datasets_dir: Directory to save uploaded files
        vector_store: Vector store instance for document storage
        
    Returns:
        Dictionary with upload status, file information, and vector store info
    """
    try:
        saved_files = []
        total_docs_processed = 0
        
        # Text splitter configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        for file in files:
            file_info: Dict[str, Any] = {
                "filename": file.filename,
                "status": "success",
                "documents_processed": 0
            }
            
            try:
                # Generate a safe filename
                filename = "".join(c if c.isalnum() or c in ' .-_' else '_' for c in file.filename)
                file_path = datasets_dir / filename
                
                # Save file to dataset directory
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Update file info
                file_info.update({
                    "saved_path": str(file_path),
                    "size": len(content),
                    "content_type": file.content_type
                })
                
                # Process the JSONL file
                try:
                    # Parse JSONL file
                    json_data = process_json_lines(file_path)
                    
                    if not json_data:
                        raise ValueError("No valid JSON objects found in the file")
                    
                    # Create documents with proper metadata
                    documents = create_documents_from_json(json_data, text_splitter)
                    
                    if documents:
                        # Add documents to the vector store with UUIDs
                        uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
                        
                        # Prepare documents for vector store
                        docs_to_add = []
                        for doc, doc_id in zip(documents, uuids):
                            docs_to_add.append({
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                                "id": doc_id
                            })
                        
                        # Add to vector store
                        vector_store.add_documents(docs_to_add)
                        
                        # Update counters
                        file_info["documents_processed"] = len(documents)
                        total_docs_processed += len(documents)
                        
                except Exception as e:
                    logger.error(f"Error processing JSONL file {file.filename}: {str(e)}")
                    file_info.update({
                        "status": "warning",
                        "warning": f"File saved but not processed: {str(e)}"
                    })
                
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                file_info.update({
                    "status": "error",
                    "error": str(e)
                })
            
            saved_files.append(file_info)
        
        # Get final vector store size
        vector_store_size = vector_store.client._collection.count()
        
        return {
            "status": "success",
            "files_processed": len(files),
            "documents_processed": total_docs_processed,
            "vector_store_size": vector_store_size,
            "saved_files": saved_files
        }
        
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )
