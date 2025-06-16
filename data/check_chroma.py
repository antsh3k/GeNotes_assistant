"""
Script to check the contents of the ChromaDB collection.
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def check_chroma_collection():
    """Check the contents of the ChromaDB collection."""
    try:
        # Get the ChromaDB path from environment variables
        chroma_path = os.getenv("CHROMA_DB_PATH", "/data/chromadb")
        collection_name = os.getenv("COLLECTION_NAME", "rag_data")
        
        print(f"Connecting to ChromaDB at: {chroma_path}")
        print(f"Collection name: {collection_name}")
        
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
            base_url=os.getenv("OLLAMA_API_BASE_URL", "http://host.docker.internal:11434")
        )
        
        # Initialize Chroma client
        db = Chroma(
            persist_directory=chroma_path,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Get collection info
        collection = db._collection
        print(f"\nCollection info:")
        print(f"  Collection name: {collection.name}")
        print(f"  Number of documents: {collection.count()}")
        
        # Get a few sample documents if available
        results = collection.get(limit=2)
        if results and 'documents' in results and results['documents']:
            print("\nSample documents:")
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\nDocument {i+1}:")
                print(f"  Content: {doc[:200]}...")
                print(f"  Metadata: {metadata}")
        else:
            print("\nNo documents found in the collection.")
            
        # Perform a simple similarity search
        print("\nPerforming a test similarity search...")
        query = "BRCA1 and BRCA2"
        print(f"Query: {query}")
        
        docs = db.similarity_search(query, k=2)
        print(f"\nTop {len(docs)} results:")
        for i, doc in enumerate(docs):
            print(f"\nResult {i+1}:")
            print(f"  Content: {doc.page_content[:200]}...")
            print(f"  Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chroma_collection()
