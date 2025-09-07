# Create configuration and utilities files

# utils/config.py
config_content = '''
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
'''

# utils/helpers.py
helpers_content = '''
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`]', '"', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)
    
    return text.strip()

def format_citations(sources: List[Dict[str, Any]]) -> str:
    """Format source citations for display"""
    if not sources:
        return "No sources available."
    
    citations = []
    for i, source in enumerate(sources, 1):
        source_name = source.get("source", "Unknown")
        page = source.get("page", "N/A")
        score = source.get("score", 0)
        
        citation = f"[{i}] {source_name}"
        if page != "N/A":
            citation += f" (page {page})"
        citation += f" - similarity: {score:.3f}"
        
        citations.append(citation)
    
    return "\\n".join(citations)

def export_to_markdown(chat_history: List[Tuple[str, str, List[Dict]]]) -> str:
    """Export chat history to markdown format"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Document QnA Session Export

**Generated**: {timestamp}
**Total Questions**: {len(chat_history)}

---

"""
    
    for i, (question, answer, sources) in enumerate(chat_history, 1):
        markdown += f"""## Question {i}

**Q**: {question}

**A**: {answer}

"""
        if sources:
            markdown += "**Sources:**\\n"
            for j, source in enumerate(sources, 1):
                source_name = source.get("source", "Unknown")
                page = source.get("page", "N/A")
                score = source.get("score", 0)
                content_preview = source.get("content", "")[:150]
                
                markdown += f"{j}. **{source_name}**"
                if page != "N/A":
                    markdown += f" (page {page})"
                markdown += f" - *similarity: {score:.3f}*\\n"
                markdown += f"   > {content_preview}...\\n\\n"
        
        markdown += "---\\n\\n"
    
    return markdown

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def extract_page_number(text: str) -> int:
    """Extract page number from text or metadata"""
    # Try to find page patterns
    page_patterns = [
        r"page[\\s]*([0-9]+)",
        r"p[\\s]*([0-9]+)",
        r"\\[([0-9]+)\\]"
    ]
    
    for pattern in page_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return int(matches[0])
            except ValueError:
                continue
    
    return 1  # Default to page 1

def calculate_similarity_score(query_embedding, doc_embedding) -> float:
    """Calculate cosine similarity between embeddings"""
    import numpy as np
    
    # Convert to numpy arrays if needed
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding)
    if not isinstance(doc_embedding, np.ndarray):
        doc_embedding = np.array(doc_embedding)
    
    # Calculate cosine similarity
    dot_product = np.dot(query_embedding, doc_embedding)
    norm_a = np.linalg.norm(query_embedding)
    norm_b = np.linalg.norm(doc_embedding)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))
'''

# Write utils files
with open(os.path.join("document_qna", "src", "utils", "config.py"), "w") as f:
    f.write(config_content.strip())

with open(os.path.join("document_qna", "src", "utils", "helpers.py"), "w") as f:
    f.write(helpers_content.strip())

print("✅ Created utils/config.py and utils/helpers.py")