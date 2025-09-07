# Create embeddings module

# First, create the base embedding interface
embedding_interface_content = '''
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class EmbeddingInterface(ABC):
    """Abstract interface for embedding providers"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name"""
        pass
'''

# Sentence Transformer implementation
sentence_transformer_content = '''
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding_interface import EmbeddingInterface

class SentenceTransformerEmbeddings(EmbeddingInterface):
    """Sentence Transformers embedding provider"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize Sentence Transformer embeddings
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run on (cpu/cuda)
        """
        self._model_name = model_name
        self.device = device
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ValueError(f"Failed to load SentenceTransformer model {model_name}: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return []
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            raise ValueError(f"Failed to embed query: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self._dimension
    
    @property  
    def model_name(self) -> str:
        """Return model name"""
        return self._model_name
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode texts in batches for memory efficiency"""
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
'''

# OpenAI embeddings (alternative provider)
openai_embeddings_content = '''
from typing import List, Optional
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .embedding_interface import EmbeddingInterface

class OpenAIEmbeddings(EmbeddingInterface):
    """OpenAI embedding provider (alternative to SentenceTransformers)"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embeddings
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model_name: OpenAI embedding model name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        openai.api_key = self._api_key
        
        # Set dimension based on model
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536, 
            "text-embedding-3-large": 3072
        }
        self._dimension = model_dimensions.get(model_name, 1536)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using OpenAI API"""
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return []
            
            response = openai.Embedding.create(
                model=self._model_name,
                input=valid_texts
            )
            
            return [item['embedding'] for item in response['data']]
            
        except Exception as e:
            raise ValueError(f"Failed to generate OpenAI embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using OpenAI API"""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            response = openai.Embedding.create(
                model=self._model_name,
                input=[text]
            )
            
            return response['data'][0]['embedding']
        except Exception as e:
            raise ValueError(f"Failed to embed query with OpenAI: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return model name"""
        return self._model_name
'''

# Create __init__ file for embeddings
embeddings_init_content = '''
from .embedding_interface import EmbeddingInterface
from .sentence_transformer_embeddings import SentenceTransformerEmbeddings

try:
    from .openai_embeddings import OpenAIEmbeddings
except ImportError:
    # OpenAI not available, that's fine
    pass

__all__ = ["EmbeddingInterface", "SentenceTransformerEmbeddings"]

# Add OpenAI to exports if available
try:
    OpenAIEmbeddings
    __all__.append("OpenAIEmbeddings")
except NameError:
    pass
'''

# Write all embedding files
with open(os.path.join("document_qna", "src", "embeddings", "embedding_interface.py"), "w") as f:
    f.write(embedding_interface_content.strip())

with open(os.path.join("document_qna", "src", "embeddings", "sentence_transformer_embeddings.py"), "w") as f:
    f.write(sentence_transformer_content.strip())

with open(os.path.join("document_qna", "src", "embeddings", "openai_embeddings.py"), "w") as f:
    f.write(openai_embeddings_content.strip())

# Update the init file
with open(os.path.join("document_qna", "src", "embeddings", "__init__.py"), "w") as f:
    f.write(embeddings_init_content.strip())

print("✅ Created complete embeddings module with modular provider interface")