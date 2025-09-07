import logging
from typing import List, Optional

# Use the correct absolute imports
from vectorstore.chroma_vectorstore import ChromaVectorStore
from chunking.overlap_chunker import Chunk

logger = logging.getLogger(__name__)

class TopKRetriever:
    """A simplified retriever that fetches top-k results from a vector store."""

    def __init__(self, vectorstore: ChromaVectorStore, k: int = 5):
        """
        Initialize the retriever.
        Args:
            vectorstore: The vector store to retrieve from.
            k: The default number of top results to return.
        """
        self.vectorstore = vectorstore
        self.k = k

    def query(self, query_text: str, k: Optional[int] = None) -> List[Chunk]:
        """
        Retrieve the top-k most relevant chunks for a given query.

        Args:
            query_text: The user's question or query string.
            k: The number of results to return (overrides the default).

        Returns:
            A list of the most relevant Chunk objects.
        """
        if not query_text:
            return []

        num_results = k or self.k
        
        try:
            # --- KEY CHANGE ---
            # The vectorstore.query method now directly returns a list of Chunks.
            # We can return this result directly. The old, more complex logic is no longer needed.
            results = self.vectorstore.query(query_text=query_text, k=num_results)
            
            logger.info(f"Retrieved {len(results)} chunks for the query.")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval from vector store failed: {e}")
            return []

