import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# LlamaIndex Global Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def init_settings():
    if not GROQ_API_KEY:
        return False
    
    # LLM: Llama 3.3 70B for high-precision logic
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    
    # Embedding: Optimized Local Model (No HF_TOKEN required, extremely fast)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    return True
