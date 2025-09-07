import logging
from typing import List, Optional
from vectorstore.chroma_vectorstore import ChromaVectorStore
from chunking.overlap_chunker import Chunk

logger = logging.getLogger(__name__)

class TopKRetriever:
    """A simplified retriever that fetches top-k results from a vector store."""

    def __init__(self, vectorstore: ChromaVectorStore, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k

    def query(self, query_text: str, k: Optional[int] = None) -> List[Chunk]:
        if not query_text:
            return []

        num_results = k or self.k
        
        try:
            results = self.vectorstore.query(query_text=query_text, k=num_results)
            
            logger.info(f"Retrieved {len(results)} chunks for the query.")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval from vector store failed: {e}")
            return []

