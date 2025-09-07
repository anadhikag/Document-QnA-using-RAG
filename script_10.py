# Create LLM module with modular providers

# Base LLM interface
llm_interface_content = '''
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..chunking.overlap_chunker import Chunk

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
    
    @abstractmethod
    def generate_completion(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1
    ) -> str:
        """Generate a completion for a given prompt"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name"""
        pass
'''

# Ollama provider
ollama_provider_content = '''
import logging
from typing import List, Dict, Any, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from ..chunking.overlap_chunker import Chunk
from .llm_interface import LLMInterface

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
        """
        Initialize Ollama provider
        
        Args:
            model: Model name (e.g., 'phi3', 'llama2', 'mistral')
            host: Ollama server host
            port: Ollama server port
            timeout: Request timeout in seconds
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama package not installed. Install with: pip install ollama")
        
        self._model_name = model
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Configure ollama client
        self.client = ollama.Client(host=f"http://{host}:{port}")
        
        # Test connection and model availability
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the model is available"""
        try:
            # Test with a simple prompt
            response = self.client.generate(
                model=self._model_name,
                prompt="Hello",
                options={"num_predict": 1}
            )
            logger.info(f"Ollama model '{self._model_name}' verified and ready")
        except Exception as e:
            error_msg = f"Failed to connect to Ollama model '{self._model_name}': {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def generate_answer(
        self, 
        context_chunks: List[Chunk], 
        question: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate an answer using retrieved context chunks"""
        
        if not context_chunks:
            return "I don't have enough information to answer your question based on the provided documents."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', chunk.metadata.get('paragraph', 'N/A'))
            
            context_parts.append(
                f"[Source {i}: {source}, Page: {page}]\\n{chunk.page_content}\\n"
            )
        
        context_text = "\\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_qa_prompt(context_text, question)
        
        # Generate response
        try:
            response = self.client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": max_tokens or 512,
                    "stop": ["\\n\\nHuman:", "\\n\\nUser:"]
                }
            )
            
            answer = response.get('response', '').strip()
            
            if not answer:
                return "I couldn't generate a proper answer. Please try rephrasing your question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    def generate_completion(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1
    ) -> str:
        """Generate a completion for a given prompt"""
        
        try:
            response = self.client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 256
                }
            )
            
            return response.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}")
            return f"Error generating completion: {str(e)}"
    
    def _build_qa_prompt(self, context: str, question: str) -> str:
        """Build a prompt for question answering with context"""
        
        return f"""You are a helpful assistant that answers questions based on provided context. Use only the information given in the context to answer the question. If you cannot find the answer in the context, say so clearly.

When citing information, reference the source number (e.g., "According to Source 1" or "As mentioned in Source 2, Page 5").

Context:
{context}

Question: {question}

Answer:"""
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self._model_name
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            models = self.client.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False
'''

# OpenAI provider (alternative)
openai_llm_content = '''
import logging
from typing import List, Dict, Any, Optional
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..chunking.overlap_chunker import Chunk
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMInterface):
    """OpenAI LLM provider (alternative to Ollama)"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1
    ):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            temperature: Default temperature for generation
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self._model_name = model
        self.temperature = temperature
        
        # Set API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        openai.api_key = api_key
        
        # Test connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify OpenAI API connection"""
        try:
            # Test with a simple completion
            response = openai.ChatCompletion.create(
                model=self._model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            logger.info(f"OpenAI model '{self._model_name}' verified and ready")
        except Exception as e:
            error_msg = f"Failed to connect to OpenAI model '{self._model_name}': {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def generate_answer(
        self, 
        context_chunks: List[Chunk], 
        question: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate an answer using retrieved context chunks"""
        
        if not context_chunks:
            return "I don't have enough information to answer your question based on the provided documents."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', chunk.metadata.get('paragraph', 'N/A'))
            
            context_parts.append(
                f"[Source {i}: {source}, Page: {page}]\\n{chunk.page_content}\\n"
            )
        
        context_text = "\\n".join(context_parts)
        
        # Build messages
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers questions based on provided context. Use only the information given in the context to answer the question. When citing information, reference the source number and page."
            },
            {
                "role": "user",
                "content": f"Context:\\n{context_text}\\n\\nQuestion: {question}\\n\\nAnswer:"
            }
        ]
        
        # Generate response
        try:
            response = openai.ChatCompletion.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens or 512,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            if not answer:
                return "I couldn't generate a proper answer. Please try rephrasing your question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    def generate_completion(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate a completion for a given prompt"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or 256,
                temperature=temperature or self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}")
            return f"Error generating completion: {str(e)}"
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self._model_name
'''

# Create __init__ file for LLM module
llm_init_content = '''
from .llm_interface import LLMInterface

# Import available providers
try:
    from .ollama_provider import OllamaProvider
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .openai_provider import OpenAIProvider
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

__all__ = ["LLMInterface"]

if OLLAMA_AVAILABLE:
    __all__.append("OllamaProvider")

if OPENAI_AVAILABLE:
    __all__.append("OpenAIProvider")
'''

# Write all LLM files
with open(os.path.join("document_qna", "src", "llm", "llm_interface.py"), "w") as f:
    f.write(llm_interface_content.strip())

with open(os.path.join("document_qna", "src", "llm", "ollama_provider.py"), "w") as f:
    f.write(ollama_provider_content.strip())

with open(os.path.join("document_qna", "src", "llm", "openai_provider.py"), "w") as f:
    f.write(openai_llm_content.strip())

# Update the init file
with open(os.path.join("document_qna", "src", "llm", "__init__.py"), "w") as f:
    f.write(llm_init_content.strip())

print("✅ Created complete LLM module with modular provider interface (Ollama + OpenAI)")