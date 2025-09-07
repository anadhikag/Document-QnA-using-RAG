import logging
from typing import List, Dict, Any, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from chunking.overlap_chunker import Chunk
from llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class OllamaProvider(LLMInterface):
    """Ollama local LLM provider"""

    def __init__(
        self,
        model: str = "phi3",
        host: str = "localhost",
        port: int = 11434,
        timeout: int = 120
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama package not installed. Install with: pip install ollama")

        self._model_name = model
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self._verify_connection()

    def _verify_connection(self):
        """Verify that the model is available and the connection is successful."""
        try:
            self.client.list()
            logger.info("Ollama connection successful.")
        except Exception as e:
            error_msg = f"Failed to connect to Ollama: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    def generate_answer(
        self,
        context_chunks: List[Chunk],
        question: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate an answer using retrieved context chunks."""
        if not context_chunks:
            return "I don't have enough information to answer your question based on the provided documents."

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"[Source {i}: {chunk.metadata.get('source', 'Unknown')}]"
            context_parts.append(f"{source_info}\n{chunk.page_content}\n")
        context_text = "\n".join(context_parts)

        prompt = self._build_qa_prompt(context_text, question)

        try:
            response = self.client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": max_tokens or 512,
                    "stop": ["\n\n"] 
                }
            )
            answer = response.get('response', '').strip()
            return answer if answer else "I couldn't generate a proper answer. Please try rephrasing."
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            return f"Error from Ollama: {str(e)}"

    def _build_qa_prompt(self, context: str, question: str) -> str:
        return f"""You are an expert Q&A assistant. Your task is to answer the user's question based ONLY on the provided context.

Follow these rules:
1.  Be direct and concise.
2.  Use a professional tone.
3.  If the answer is not in the context, state that clearly.
4.  Do not mention the source numbers in your answer.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""

    @property
    def model_name(self) -> str:
        return self._model_name

