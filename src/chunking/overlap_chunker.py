import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ingestion.document_processor import Document

@dataclass 
class Chunk:
    """Represents a text chunk with metadata"""
    page_content: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        if 'chunk_id' not in self.metadata:
            source = self.metadata.get('source', 'unknown')
            chunk_index = self.metadata.get('chunk_index', 0)
            self.metadata['chunk_id'] = f"{source}_chunk_{chunk_index}"

class OverlapChunker:
    """Text chunker with configurable overlap"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.page_content
        if len(text) <= self.chunk_size:
            return [Chunk(
                page_content=text,
                metadata={**document.metadata, 'chunk_index': 0, 'total_chunks': 1}
            )]

        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                boundary = self._find_sentence_boundary(text, end)
                end = boundary if boundary > start else end
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        'chunk_index': chunk_index,
                        'start_char': start,
                        'end_char': end
                    }
                ))
                chunk_index += 1
            
            next_start = start + self.chunk_size - self.overlap
            if next_start <= start: 
                break
            start = next_start
            
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks

    def _find_sentence_boundary(self, text: str, preferred_end: int) -> int:
        search_range = text[max(0, preferred_end - 100):preferred_end]
        sentence_ends = [m.start() for m in re.finditer(r'[.!?\\n] +', search_range)]
        if sentence_ends:
            return max(0, preferred_end - 100) + sentence_ends[-1] + 1
        return preferred_end
