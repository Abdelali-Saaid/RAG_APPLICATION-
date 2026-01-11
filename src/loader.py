from llama_index.core import SimpleDirectoryReader
from .config import DATA_DIR

class DocumentLoader:
    """Handles Parsing & Loading of documents."""
    
    @staticmethod
    def load_from_data_dir(required_exts: list = None):
        """
        Load data from the local data directory.
        Allows specifying extensions (e.g., ['.pdf', '.txt', '.md']).
        """
        reader = SimpleDirectoryReader(
            DATA_DIR, 
            required_exts=required_exts,
            recursive=True
        )
        return reader.load_data()
    
    @staticmethod
    def load_file(file_path: str):
        """Load a single specific file with robust parsing."""
        reader = SimpleDirectoryReader(input_files=[file_path])
        return reader.load_data()
