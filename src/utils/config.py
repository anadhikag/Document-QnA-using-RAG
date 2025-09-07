import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""

    # Vector database settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Embedding settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM settings  
    LLM_MODEL = os.getenv("LLM_MODEL", "ollama:phi3")

    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval settings
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Processing settings
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

    def __repr__(self):
        return f"""Config(
    CHROMA_PERSIST_DIR={self.CHROMA_PERSIST_DIR},
    EMBEDDING_MODEL={self.EMBEDDING_MODEL},
    LLM_MODEL={self.LLM_MODEL},
    CHUNK_SIZE={self.CHUNK_SIZE},
    CHUNK_OVERLAP={self.CHUNK_OVERLAP},
    TOP_K={self.TOP_K}
)"""