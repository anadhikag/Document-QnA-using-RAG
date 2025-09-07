# Create retrieval module

top_k_retriever_content = '''
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..chunking.overlap_chunker import Chunk
from ..vectorstore.chroma_vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)

class TopKRetriever:
    """Top-K retrieval with similarity scoring and optional reranking"""
    
    def __init__(
        self,
        vectorstore: ChromaVectorStore,
        k: int = 5,
        score_threshold: float = 0.0,
        enable_reranking: bool = False
    ):
        """
        Initialize Top-K retriever
        
        Args:
            vectorstore: Vector store to retrieve from
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            enable_reranking: Whether to enable semantic reranking
        """
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold
        self.enable_reranking = enable_reranking
    
    def query(
        self, 
        query_text: str, 
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Chunk]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query_text: Query string
            k: Number of results (override default)
            filter_metadata: Metadata filters
            score_threshold: Minimum score threshold (override default)
            
        Returns:
            List of relevant chunks ordered by relevance
        """
        
        k = k or self.k
        score_threshold = score_threshold or self.score_threshold
        
        try:
            # Retrieve from vector store
            results_with_scores = self.vectorstore.query(
                query_text=query_text,
                k=k * 2 if self.enable_reranking else k,  # Get more if reranking
                filter_metadata=filter_metadata
            )
            
            # Filter by score threshold
            filtered_results = [
                (chunk, score) for chunk, score in results_with_scores
                if score >= score_threshold
            ]
            
            if not filtered_results:
                logger.info(f"No results found above threshold {score_threshold}")
                return []
            
            # Optional reranking
            if self.enable_reranking and len(filtered_results) > k:
                filtered_results = self._rerank_results(query_text, filtered_results, k)
            
            # Return just the chunks (scores are stored in metadata)
            chunks = [chunk for chunk, _ in filtered_results[:k]]
            
            logger.info(f"Retrieved {len(chunks)} chunks for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def query_with_sources(
        self,
        query_text: str,
        k: Optional[int] = None,
        group_by_source: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query and return results with enhanced source information
        
        Args:
            query_text: Query string
            k: Number of results
            group_by_source: Whether to group results by source document
            
        Returns:
            List of dicts with chunk, score, and source information
        """
        
        chunks = self.query(query_text, k)
        
        results = []
        for chunk in chunks:
            result = {
                'chunk': chunk,
                'content': chunk.page_content,
                'score': chunk.metadata.get('score', 0.0),
                'source': chunk.metadata.get('source', 'Unknown'),
                'page': chunk.metadata.get('page', chunk.metadata.get('paragraph', 'N/A')),
                'metadata': chunk.metadata
            }
            results.append(result)
        
        if group_by_source:
            results = self._group_by_source(results)
        
        return results
    
    def _rerank_results(
        self, 
        query_text: str, 
        results: List[Tuple[Chunk, float]], 
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank results using additional semantic features
        
        This is a placeholder for more sophisticated reranking.
        Could be extended with cross-encoders, BM25, etc.
        """
        
        # For now, just return top results sorted by score
        # In a production system, you might use a cross-encoder here
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _group_by_source(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group results by source document for better organization"""
        
        source_groups = {}
        
        for result in results:
            source = result['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
        
        # Flatten while maintaining source grouping
        grouped_results = []
        for source, source_results in source_groups.items():
            # Sort within each source by score
            source_results.sort(key=lambda x: x['score'], reverse=True)
            grouped_results.extend(source_results)
        
        return grouped_results
    
    def get_context_window(
        self, 
        query_text: str, 
        window_size: int = 2000,
        k: Optional[int] = None
    ) -> str:
        """
        Get a larger context window by combining and trimming retrieved chunks
        
        Args:
            query_text: Query string
            window_size: Maximum characters in context window
            k: Number of chunks to retrieve
            
        Returns:
            Combined context string
        """
        
        chunks = self.query(query_text, k)
        
        if not chunks:
            return ""
        
        # Combine chunks with source indicators
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'Unknown')
            page = chunk.metadata.get('page', chunk.metadata.get('paragraph', 'N/A'))
            
            # Add source header
            header = f"[Source: {source}, Page: {page}]\\n"
            chunk_text = f"{header}{chunk.page_content}\\n\\n"
            
            # Check if adding this chunk would exceed window size
            if current_length + len(chunk_text) > window_size:
                # Try to fit partial chunk
                remaining_space = window_size - current_length - len(header) - 10
                if remaining_space > 100:  # Only if we have reasonable space left
                    partial_content = chunk.page_content[:remaining_space] + "..."
                    context_parts.append(f"{header}{partial_content}")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    def search_by_metadata(
        self, 
        metadata_filters: Dict[str, Any], 
        query_text: Optional[str] = None,
        k: Optional[int] = None
    ) -> List[Chunk]:
        """
        Search documents by metadata filters with optional text query
        
        Args:
            metadata_filters: Metadata key-value filters
            query_text: Optional text query for hybrid search
            k: Number of results
            
        Returns:
            List of matching chunks
        """
        
        if query_text:
            # Hybrid search: metadata filter + text similarity
            return self.query(
                query_text=query_text,
                k=k,
                filter_metadata=metadata_filters
            )
        else:
            # Pure metadata search - this would need additional ChromaDB functionality
            # For now, we'll do a broad query and filter results
            # In production, you'd want to implement this more efficiently
            all_results = self.vectorstore.query(
                query_text="",  # Empty query to get all
                k=1000  # Large number to get all results
            )
            
            # Filter by metadata
            filtered_chunks = []
            for chunk, _ in all_results:
                if self._matches_metadata_filter(chunk.metadata, metadata_filters):
                    filtered_chunks.append(chunk)
                    if len(filtered_chunks) >= (k or self.k):
                        break
            
            return filtered_chunks
    
    def _matches_metadata_filter(
        self, 
        metadata: Dict[str, Any], 
        filters: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria"""
        
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != expected_value:
                return False
        
        return True
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance"""
        
        collection_stats = self.vectorstore.get_collection_stats()
        
        return {
            **collection_stats,
            'retriever_k': self.k,
            'score_threshold': self.score_threshold,
            'reranking_enabled': self.enable_reranking
        }
'''

# Write retriever
with open(os.path.join("document_qna", "src", "retrieval", "top_k_retriever.py"), "w") as f:
    f.write(top_k_retriever_content.strip())

print("✅ Created retrieval/top_k_retriever.py")