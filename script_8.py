# Create ChromaDB vector store module

chroma_vectorstore_content = '''
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..chunking.overlap_chunker import Chunk
from ..embeddings.embedding_interface import EmbeddingInterface

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """ChromaDB vector store with persistent storage"""
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents", 
        embeddings: Optional[EmbeddingInterface] = None
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embeddings: Embedding provider interface
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embeddings
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize collection
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get existing collection"""
        
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} documents")
            
        except ValueError:
            # Collection doesn't exist, create it
            if self.embeddings is None:
                # Use default ChromaDB embedding function if none provided
                embedding_function = embedding_functions.DefaultEmbeddingFunction()
            else:
                # Use custom embedding function wrapper
                embedding_function = ChromaEmbeddingWrapper(self.embeddings)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection '{self.collection_name}'")
    
    def add_documents(self, chunks: List[Chunk]) -> List[str]:
        """Add document chunks to the vector store"""
        
        if not chunks:
            return []
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            
            # Prepare metadata (ChromaDB requires flat dict)
            metadata = self._flatten_metadata(chunk.metadata)
            
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append(metadata)
            
            # Generate embeddings if using custom embeddings
            if self.embeddings:
                embedding = self.embeddings.embed_query(chunk.page_content)
                embeddings.append(embedding)
        
        try:
            # Add to collection
            if embeddings:  # Custom embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:  # Use ChromaDB's default embedding
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(chunks)} documents to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            raise ValueError(f"Failed to add documents: {str(e)}")
    
    def query(
        self, 
        query_text: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Query the vector store and return relevant chunks with scores"""
        
        try:
            # Perform query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=filter_metadata
            )
            
            # Convert results to chunks with scores
            chunks_with_scores = []
            
            if results['documents'] and results['documents'][0]:  # Check if results exist
                for i in range(len(results['documents'][0])):
                    # Reconstruct chunk
                    chunk = Chunk(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )
                    
                    # Get similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    # Add score to metadata for easy access
                    chunk.metadata['score'] = similarity
                    
                    chunks_with_scores.append((chunk, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {str(e)}")
            return []
    
    def similarity_search(
        self, 
        query_text: str, 
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Chunk]:
        """Perform similarity search and return chunks above threshold"""
        
        results_with_scores = self.query(query_text, k)
        
        # Filter by score threshold and return only chunks
        filtered_chunks = []
        for chunk, score in results_with_scores:
            if score >= score_threshold:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        
        try:
            count = self.collection.count()
            
            return {
                'name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        
        try:
            # Get all document IDs
            all_results = self.collection.get()
            if all_results['ids']:
                self.collection.delete(ids=all_results['ids'])
            logger.info(f"Cleared all documents from collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metadata for ChromaDB compatibility"""
        
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif value is None:
                flattened[key] = ""
            else:
                # Convert complex types to strings
                flattened[key] = str(value)
        
        return flattened

class ChromaEmbeddingWrapper:
    """Wrapper to use custom embedding provider with ChromaDB"""
    
    def __init__(self, embeddings: EmbeddingInterface):
        self.embeddings = embeddings
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Embedding function interface for ChromaDB"""
        return self.embeddings.embed_documents(texts)
'''

# Write ChromaDB vector store
with open(os.path.join("document_qna", "src", "vectorstore", "chroma_vectorstore.py"), "w") as f:
    f.write(chroma_vectorstore_content.strip())

print("✅ Created vectorstore/chroma_vectorstore.py")