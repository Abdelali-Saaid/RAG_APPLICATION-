import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from .config import STORAGE_DIR

class IndexStorage:
    """Handles Storing and loading indices from persistent storage."""
    
    @staticmethod
    def persist_index(index: VectorStoreIndex):
        """Save index to disk."""
        if index:
            index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    @staticmethod
    def load_index():
        """Load index from disk if it exists."""
        if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            return load_index_from_storage(storage_context)
        return None
