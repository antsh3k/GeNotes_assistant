"""
File upload endpoints.
"""
import logging
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from pathlib import Path
from typing import List, Dict, Any

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

def load_document(file_path: Path) -> List[Document]:
    """Load a document using the appropriate loader based on file extension."""
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_extension in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(str(file_path))
        elif file_extension in ['.pptx', '.ppt']:
            loader = UnstructuredPowerPointLoader(str(file_path))
        elif file_extension in ['.txt', '.md', '.csv', '.json']:
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    datasets_dir: Path = Depends(get_datasets_dir),
    vector_store = Depends(get_vector_store)
):
    """
    Upload, process files, and add them to the vector store.
    
    Args:
        files: List of files to upload
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
            length_function=len
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
                
                # Load and process the document
                try:
                    # Load document
                    docs = load_document(file_path)
                    
                    # Split into chunks
                    chunks = text_splitter.split_documents(docs)
                    
                    # Add metadata to each chunk
                    for chunk in chunks:
                        chunk.metadata.update({
                            "source": str(file_path.name),
                            "type": "uploaded_file",
                            "original_filename": file.filename,
                            "content_type": file.content_type
                        })
                    
                    # Add to vector store
                    if chunks:
                        # Get the Chroma client
                        chroma_client = vector_store.client
                        
                        # Add documents to the vector store
                        chroma_client.add_documents(chunks)
                        
                        # Update counters
                        file_info["documents_processed"] = len(chunks)
                        total_docs_processed += len(chunks)
                        
                        # Persist the vector store
                        chroma_client.persist()
                        
                except Exception as e:
                    logger.error(f"Error processing document {file.filename}: {str(e)}")
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
        chroma_client = vector_store.client
        vector_store_size = chroma_client._collection.count()
        
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
