import chromadb
import os
from pathlib import Path

# Get the ChromaDB path from environment or use default
chroma_path = os.getenv('CHROMA_DB_PATH', './chromadb')
chroma_path = Path(chroma_path).resolve()

print(f"Connecting to ChromaDB at: {chroma_path}")

# Create a Chroma client
client = chromadb.PersistentClient(path=str(chroma_path))

# List all collections
collections = client.list_collections()

if not collections:
    print("No collections found in the vector store.")
else:
    print("\nAvailable collections:")
    for collection in collections:
        print(f"- {collection.name} (count: {collection.count()})")
        print(f"  Metadata: {collection.metadata}")
