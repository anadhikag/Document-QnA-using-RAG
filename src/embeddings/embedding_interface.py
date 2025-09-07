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