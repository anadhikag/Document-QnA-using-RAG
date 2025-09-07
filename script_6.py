# Create chunking module

overlap_chunker_content = '''
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..ingestion.document_processor import Document

@dataclass 
class Chunk:
    """Represents a text chunk with metadata"""
    page_content: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Generate chunk ID if not present
        if 'chunk_id' not in self.metadata:
            source = self.metadata.get('source', 'unknown')
            chunk_index = self.metadata.get('chunk_index', 0)
            self.metadata['chunk_id'] = f"{source}_chunk_{chunk_index}"

class OverlapChunker:
    """Text chunker with configurable overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100, min_chunk_size: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks  
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents"""
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document with overlap"""
        
        text = document.page_content
        if len(text) <= self.chunk_size:
            # Document is small enough to be a single chunk
            chunk = Chunk(
                page_content=text,
                metadata={
                    **document.metadata,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            )
            return [chunk]
        
        # Split into overlapping chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < len(text):
                end = self._find_sentence_boundary(text, end)
            else:
                end = len(text)
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Only keep chunks that meet minimum size requirement
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        'chunk_index': chunk_index,
                        'start_char': start,
                        'end_char': end
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            start = end - self.overlap
            
            # Prevent infinite loop
            if start <= 0:
                break
        
        # Update total chunks count in all chunk metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, preferred_end: int) -> int:
        """Find a good sentence boundary near the preferred end position"""
        
        # Look for sentence endings within a window around preferred_end
        window_size = min(100, len(text) - preferred_end + 100)
        search_start = max(0, preferred_end - window_size // 2)
        search_end = min(len(text), preferred_end + window_size // 2)
        
        search_text = text[search_start:search_end]
        
        # Find sentence boundaries (periods, exclamation, question marks)
        sentence_endings = []
        for match in re.finditer(r'[.!?]\\s+', search_text):
            actual_pos = search_start + match.end()
            sentence_endings.append(actual_pos)
        
        if sentence_endings:
            # Find the ending closest to preferred_end
            closest_ending = min(sentence_endings, key=lambda x: abs(x - preferred_end))
            
            # Only use if it's reasonable close and not too small
            if (abs(closest_ending - preferred_end) < 200 and 
                closest_ending > preferred_end - self.chunk_size + self.min_chunk_size):
                return closest_ending
        
        # If no good sentence boundary, try paragraph boundary
        paragraph_endings = []
        for match in re.finditer(r'\\n\\n+', text[search_start:search_end]):
            actual_pos = search_start + match.end()
            paragraph_endings.append(actual_pos)
        
        if paragraph_endings:
            closest_paragraph = min(paragraph_endings, key=lambda x: abs(x - preferred_end))
            if (abs(closest_paragraph - preferred_end) < 300 and
                closest_paragraph > preferred_end - self.chunk_size + self.min_chunk_size):
                return closest_paragraph
        
        # Fall back to original preferred_end
        return preferred_end
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk raw text with basic metadata"""
        
        if metadata is None:
            metadata = {'source': 'text_input', 'type': 'text'}
        
        doc = Document(page_content=text, metadata=metadata)
        return self.chunk_document(doc)
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        
        if not chunks:
            return {'count': 0}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'count': len(chunks),
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'avg_size': sum(chunk_sizes) / len(chunk_sizes),
            'total_chars': sum(chunk_sizes)
        }
    
    def deduplicate_chunks(self, chunks: List[Chunk], similarity_threshold: float = 0.95) -> List[Chunk]:
        """Remove duplicate or very similar chunks"""
        
        if not chunks:
            return chunks
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Create a simple hash of the chunk content
            normalized_content = re.sub(r'\\s+', ' ', chunk.page_content.lower().strip())
            content_hash = hash(normalized_content)
            
            # Check if we've seen very similar content
            is_duplicate = False
            for seen_hash in seen_hashes:
                # Simple similarity check based on content overlap
                if content_hash == seen_hash:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_hashes.add(content_hash)
        
        return unique_chunks
'''

# Write chunker
with open(os.path.join("document_qna", "src", "chunking", "overlap_chunker.py"), "w") as f:
    f.write(overlap_chunker_content.strip())

print("✅ Created chunking/overlap_chunker.py")