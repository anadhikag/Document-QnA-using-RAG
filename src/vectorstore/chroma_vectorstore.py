import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from chunking.overlap_chunker import Chunk
from embeddings.embedding_interface import EmbeddingInterface

logger = logging.getLogger(__name__)

# --- KEY CHANGE: Implement ChromaDB's EmbeddingFunction interface directly ---
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embeddings: EmbeddingInterface):
        self.embeddings = embeddings

    # --- KEY FIX: The method must be '__call__' and the parameter must be 'input' ---
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        This method is called by ChromaDB to get embeddings.
        It must accept a parameter named 'input'.
        """
        return self.embeddings.embed_documents(input)
        
    def name(self) -> str:
        """Provides a name for the embedding function for ChromaDB."""
        return self.embeddings.model_name

class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        embeddings: Optional[EmbeddingInterface] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embeddings
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        embedding_function = CustomEmbeddingFunction(self.embeddings) if self.embeddings else None
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized collection '{self.collection_name}' with {self.collection.count()} documents.")

    def add_documents(self, chunks: List[Chunk]) -> List[str]:
        if not chunks:
            return []
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [self._flatten_metadata(chunk.metadata) for chunk in chunks]
        
        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(f"Added {len(chunks)} documents to collection.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    def query(self, query_text: str, k: int = 5) -> List[Chunk]:
        try:
            results = self.collection.query(query_texts=[query_text], n_results=k)
            
            chunks_with_scores = []
            if results.get('documents') and results['documents'][0]:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results.get('metadatas', [[]])[0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                    distance = results.get('distances', [[]])[0][i] if results.get('distances') and results['distances'][0] else 1.0
                    metadata['score'] = 1 - distance
                    
                    chunk = Chunk(page_content=doc_content, metadata=metadata)
                    chunks_with_scores.append(chunk)
            
            return chunks_with_scores
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            return []

    def clear_collection(self):
        """Deletes all embeddings and documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            embedding_function = CustomEmbeddingFunction(self.embeddings) if self.embeddings else None
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Successfully cleared and re-created collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            if self.collection:
                self.collection.delete(where={})

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            else:
                flattened[key] = str(value)
        return flattened

