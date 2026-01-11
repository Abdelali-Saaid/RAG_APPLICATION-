from llama_index.core import VectorStoreIndex
from typing import List

class DocumentIndexer:
    """Handles Indexing of loaded documents into vector representations."""
    
    @staticmethod
    def create_index(documents: List):
        """Build a vector index from the provided documents."""
        if not documents:
            return None
        return VectorStoreIndex.from_documents(documents, show_progress=True)
    
    @staticmethod
    def add_to_index(index: VectorStoreIndex, documents: List):
        """Append new documents to an existing index."""
        for doc in documents:
            index.insert(doc)
        return index
