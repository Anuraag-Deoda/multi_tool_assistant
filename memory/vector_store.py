# ============================================================================
# FILE: memory/vector_store.py
# ============================================================================

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not installed. Run: pip install chromadb")


@dataclass
class Document:
    """Represents a document in the vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """Represents a search result"""
    document: Document
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.document.content,
            "metadata": self.document.metadata,
            "score": self.score
        }


class VectorStore:
    """
    Vector database for semantic search using ChromaDB.
    
    Features:
    - Store documents with embeddings
    - Semantic similarity search
    - Metadata filtering
    - Persistent storage
    """
    
    def __init__(
        self,
        config: Any,
        embedding_fn: Optional[callable] = None,
        collection_name: str = None
    ):
        self.config = config
        self.embedding_fn = embedding_fn
        self.collection_name = collection_name or config.COLLECTION_NAME
        
        if not CHROMADB_AVAILABLE:
            self._use_fallback = True
            self._fallback_store: List[Document] = []
            print("⚠️ Using fallback in-memory store (install chromadb for full features)")
            return
        
        self._use_fallback = False
        
        # Initialize ChromaDB
        db_path = Path(config.VECTOR_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Assistant memory store"}
        )
        
        print(f"📊 Vector store initialized. Documents: {self.collection.count()}")
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        if self.embedding_fn:
            return self.embedding_fn(text)
        
        # Fallback: simple bag-of-words style embedding
        # This is just for testing - real implementation needs proper embeddings
        words = text.lower().split()
        # Create a simple 384-dim vector based on word hashes
        embedding = [0.0] * 384
        for word in words:
            idx = hash(word) % 384
            embedding[idx] += 1.0
        # Normalize
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        return embedding
    
    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        doc_id: str = None
    ) -> str:
        """Add a document to the vector store"""
        doc_id = doc_id or self._generate_id(content)
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now().isoformat()
        
        if self._use_fallback:
            doc = Document(id=doc_id, content=content, metadata=metadata)
            self._fallback_store.append(doc)
            return doc_id
        
        # Get embedding
        embedding = self._get_embedding(content)
        
        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Add multiple documents"""
        ids = []
        for doc in documents:
            doc_id = self.add_document(
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                doc_id=doc.get("id")
            )
            ids.append(doc_id)
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Semantic search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Filter results by metadata
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.TOP_K_RESULTS
        
        if self._use_fallback:
            return self._fallback_search(query, top_k)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search in ChromaDB
        where = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                doc = Document(
                    id=doc_id,
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                # Convert distance to similarity score (lower distance = higher similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1.0 / (1.0 + distance)  # Convert to 0-1 range
                
                search_results.append(SearchResult(document=doc, score=score))
        
        return search_results
    
    def _fallback_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Simple keyword-based fallback search"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self._fallback_store:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / len(query_words)
                results.append(SearchResult(document=doc, score=score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        if self._use_fallback:
            for doc in self._fallback_store:
                if doc.id == doc_id:
                    return doc
            return None
        
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        
        if result and result['ids']:
            return Document(
                id=result['ids'][0],
                content=result['documents'][0],
                metadata=result['metadatas'][0] if result['metadatas'] else {}
            )
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        if self._use_fallback:
            self._fallback_store = [d for d in self._fallback_store if d.id != doc_id]
            return True
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def update_document(
        self,
        doc_id: str,
        content: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Update a document"""
        if self._use_fallback:
            for doc in self._fallback_store:
                if doc.id == doc_id:
                    if content:
                        doc.content = content
                    if metadata:
                        doc.metadata.update(metadata)
                    return True
            return False
        
        try:
            updates = {}
            if content:
                updates["documents"] = [content]
                updates["embeddings"] = [self._get_embedding(content)]
            if metadata:
                updates["metadatas"] = [metadata]
            
            self.collection.update(ids=[doc_id], **updates)
            return True
        except Exception:
            return False
    
    def get_all_documents(self, limit: int = 100) -> List[Document]:
        """Get all documents"""
        if self._use_fallback:
            return self._fallback_store[:limit]
        
        result = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        documents = []
        if result and result['ids']:
            for i, doc_id in enumerate(result['ids']):
                documents.append(Document(
                    id=doc_id,
                    content=result['documents'][i],
                    metadata=result['metadatas'][i] if result['metadatas'] else {}
                ))
        
        return documents
    
    def count(self) -> int:
        """Get total document count"""
        if self._use_fallback:
            return len(self._fallback_store)
        return self.collection.count()
    
    def clear(self):
        """Clear all documents"""
        if self._use_fallback:
            self._fallback_store = []
            return
        
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Assistant memory store"}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": self.count(),
            "collection_name": self.collection_name,
            "using_fallback": self._use_fallback
        }