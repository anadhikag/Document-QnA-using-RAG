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