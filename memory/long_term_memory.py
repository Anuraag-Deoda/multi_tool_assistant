# ============================================================================
# FILE: memory/long_term_memory.py
# ============================================================================

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class MemoryType(Enum):
    """Types of memories"""
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"
    USER_INFO = "user_info"


@dataclass
class Memory:
    """Represents a single memory"""
    id: str
    content: str
    memory_type: str
    category: str
    importance: float  # 0.0 to 1.0
    created_at: str
    last_accessed: str
    access_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        return cls(**data)


class LongTermMemory:
    """
    Long-term memory storage and retrieval system.
    
    Features:
    - Store facts, preferences, and learned information
    - Categorized memory storage
    - Importance-based retrieval
    - Integration with vector store for semantic search
    - User profile management
    """
    
    def __init__(
        self,
        config: Any,
        vector_store: Optional[Any] = None,
        storage_path: str = None
    ):
        self.config = config
        self.vector_store = vector_store
        self.storage_path = Path(storage_path or config.MEMORY_FILE_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> memory_ids
        
        # User profile
        self.user_profile: Dict[str, Any] = {
            "name": None,
            "preferences": {},
            "facts": [],
            "context": {}
        }
        
        # Load existing memories
        self._load_memories()
        self._load_user_profile()
    
    def _generate_id(self) -> str:
        """Generate unique memory ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        category: str = "general",
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: The information to remember
            memory_type: Type of memory (fact, preference, experience, etc.)
            category: Category for organization
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        memory_id = self._generate_id()
        now = datetime.now().isoformat()
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            category=category,
            importance=min(1.0, max(0.0, importance)),
            created_at=now,
            last_accessed=now,
            access_count=0,
            metadata=metadata or {}
        )
        
        # Store in memory dict
        self.memories[memory_id] = memory
        
        # Add to category index
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(memory_id)
        
        # Also store in vector store for semantic search
        if self.vector_store:
            self.vector_store.add_document(
                content=content,
                metadata={
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "category": category,
                    "importance": importance
                },
                doc_id=f"memory_{memory_id}"
            )
        
        # Save to disk
        self._save_memories()
        
        print(f"💾 Remembered: {content[:50]}...")
        return memory_id
    
    def recall(
        self,
        query: str = None,
        category: str = None,
        memory_type: str = None,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """
        Recall memories based on criteria.
        
        Args:
            query: Semantic search query
            category: Filter by category
            memory_type: Filter by type
            limit: Maximum results
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memories
        """
        results = []
        
        # If query provided and vector store available, use semantic search
        if query and self.vector_store:
            search_results = self.vector_store.search(
                query=query,
                top_k=limit * 2,  # Get more to filter
                filter_metadata={"memory_type": memory_type} if memory_type else None
            )
            
            for result in search_results:
                memory_id = result.document.metadata.get("memory_id")
                if memory_id and memory_id in self.memories:
                    memory = self.memories[memory_id]
                    if memory.importance >= min_importance:
                        if category is None or memory.category == category:
                            results.append(memory)
                            self._update_access(memory_id)
        else:
            # Fallback to filtering
            for memory in self.memories.values():
                if memory.importance < min_importance:
                    continue
                if category and memory.category != category:
                    continue
                if memory_type and memory.memory_type != memory_type:
                    continue
                if query and query.lower() not in memory.content.lower():
                    continue
                results.append(memory)
        
        # Sort by importance and recency
        results.sort(
            key=lambda m: (m.importance, m.access_count),
            reverse=True
        )
        
        return results[:limit]
    
    def recall_by_id(self, memory_id: str) -> Optional[Memory]:
        """Recall a specific memory by ID"""
        if memory_id in self.memories:
            self._update_access(memory_id)
            return self.memories[memory_id]
        return None
    
    def forget(self, memory_id: str) -> bool:
        """Remove a memory"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Remove from category index
        if memory.category in self.categories:
            self.categories[memory.category] = [
                mid for mid in self.categories[memory.category]
                if mid != memory_id
            ]
        
        # Remove from vector store
        if self.vector_store:
            self.vector_store.delete_document(f"memory_{memory_id}")
        
        # Remove from memory dict
        del self.memories[memory_id]
        
        self._save_memories()
        return True
    
    def update_importance(self, memory_id: str, importance: float) -> bool:
        """Update memory importance"""
        if memory_id in self.memories:
            self.memories[memory_id].importance = min(1.0, max(0.0, importance))
            self._save_memories()
            return True
        return False
    
    def _update_access(self, memory_id: str):
        """Update access timestamp and count"""
        if memory_id in self.memories:
            self.memories[memory_id].last_accessed = datetime.now().isoformat()
            self.memories[memory_id].access_count += 1
    
    # User Profile Methods
    
    def set_user_info(self, key: str, value: Any):
        """Set user profile information"""
        if key == "name":
            self.user_profile["name"] = value
        else:
            self.user_profile["context"][key] = value
        
        # Also store as memory
        self.remember(
            content=f"User {key}: {value}",
            memory_type="user_info",
            category="user_profile",
            importance=0.8
        )
        
        self._save_user_profile()
    
    def get_user_info(self, key: str = None) -> Any:
        """Get user profile information"""
        if key is None:
            return self.user_profile
        if key == "name":
            return self.user_profile.get("name")
        return self.user_profile.get("context", {}).get(key)
    
    def set_preference(self, key: str, value: Any):
        """Set user preference"""
        self.user_profile["preferences"][key] = value
        
        self.remember(
            content=f"User prefers {key}: {value}",
            memory_type="preference",
            category="preferences",
            importance=0.7
        )
        
        self._save_user_profile()
    
    def get_preference(self, key: str) -> Any:
        """Get user preference"""
        return self.user_profile.get("preferences", {}).get(key)
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all preferences"""
        return self.user_profile.get("preferences", {})
    
    # Context methods for RAG
    
    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 500
    ) -> str:
        """
        Get relevant context from memory for RAG.
        
        Returns formatted context string for inclusion in prompts.
        """
        relevant_memories = self.recall(query=query, limit=5)
        
        context_parts = []
        total_length = 0
        
        # Add user profile if available
        if self.user_profile.get("name"):
            profile_info = f"User's name is {self.user_profile['name']}."
            context_parts.append(profile_info)
            total_length += len(profile_info)
        
        # Add preferences
        prefs = self.get_all_preferences()
        if prefs:
            pref_str = "User preferences: " + ", ".join(
                f"{k}={v}" for k, v in list(prefs.items())[:5]
            )
            context_parts.append(pref_str)
            total_length += len(pref_str)
        
        # Add relevant memories
        for memory in relevant_memories:
            if total_length > max_tokens * 4:  # Rough character estimate
                break
            mem_str = f"[{memory.memory_type}] {memory.content}"
            context_parts.append(mem_str)
            total_length += len(mem_str)
        
        if not context_parts:
            return ""
        
        return "Relevant memory context:\n" + "\n".join(context_parts)
    
    # Persistence methods
    
    def _save_memories(self):
        """Save memories to disk"""
        data = {
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()},
            "categories": self.categories
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_memories(self):
        """Load memories from disk"""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.memories = {
                mid: Memory.from_dict(m)
                for mid, m in data.get("memories", {}).items()
            }
            self.categories = data.get("categories", {})
            
            print(f"📂 Loaded {len(self.memories)} memories")
            
        except Exception as e:
            print(f"⚠️ Failed to load memories: {e}")
    
    def _save_user_profile(self):
        """Save user profile to disk"""
        profile_path = Path(self.config.USER_PROFILE_PATH)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=2)
    
    def _load_user_profile(self):
        """Load user profile from disk"""
        profile_path = Path(self.config.USER_PROFILE_PATH)
        
        if not profile_path.exists():
            return
        
        try:
            with open(profile_path, 'r') as f:
                self.user_profile = json.load(f)
            print(f"👤 Loaded user profile")
        except Exception as e:
            print(f"⚠️ Failed to load user profile: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        type_counts = {}
        for memory in self.memories.values():
            type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "categories": list(self.categories.keys()),
            "type_distribution": type_counts,
            "user_profile_set": self.user_profile.get("name") is not None
        }
    
    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories as list of dicts"""
        return [m.to_dict() for m in self.memories.values()]
    
    def import_memories(self, memories: List[Dict[str, Any]]):
        """Import memories from list of dicts"""
        for mem_data in memories:
            memory = Memory.from_dict(mem_data)
            self.memories[memory.id] = memory
            
            if memory.category not in self.categories:
                self.categories[memory.category] = []
            self.categories[memory.category].append(memory.id)
        
        self._save_memories()