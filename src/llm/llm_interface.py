from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# CORRECTED IMPORT: Removed the '..' to make it an absolute import
# Python knows to look in the 'src' folder for 'chunking'
from chunking.overlap_chunker import Chunk

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def generate_answer(
        self, 
        context_chunks: List[Chunk], 
        question: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate an answer based on context chunks and question"""
        pass

    def generate_completion(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1
    ) -> str:
        """This method is optional for this app, but kept for interface consistency."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name"""
        pass
