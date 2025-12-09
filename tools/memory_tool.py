# ============================================================================
# FILE: tools/memory_tool.py
# ============================================================================

from tools.base_tool import BaseTool
from typing import Dict, Any, Optional
from memory.long_term_memory import LongTermMemory
from memory.vector_store import VectorStore


class MemoryTool(BaseTool):
    """Tool for memory operations - remember, recall, and search"""
    
    def __init__(
        self,
        long_term_memory: LongTermMemory,
        vector_store: VectorStore
    ):
        self.ltm = long_term_memory
        self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        return "memory_manager"
    
    @property
    def description(self) -> str:
        return """Manage long-term memory. Use this tool to:
        - Remember important facts, preferences, or information about the user
        - Recall previously stored information
        - Search through stored knowledge
        - Set user preferences and profile information
        
        Operations:
        - 'remember': Store new information (requires 'content' and optionally 'category', 'importance')
        - 'recall': Retrieve stored information (requires 'query')
        - 'search': Semantic search through memories (requires 'query')
        - 'set_preference': Store user preference (requires 'key' and 'value')
        - 'get_preferences': Get all user preferences
        - 'set_user_info': Store user information (requires 'key' and 'value')
        - 'forget': Remove a memory (requires 'memory_id')
        """
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["remember", "recall", "search", "set_preference", 
                            "get_preferences", "set_user_info", "forget", "get_context"],
                    "description": "The memory operation to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Content to remember (for 'remember' operation)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'recall' and 'search' operations)"
                },
                "category": {
                    "type": "string",
                    "description": "Category for the memory (e.g., 'personal', 'work', 'preferences')"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0-1 (higher = more important)"
                },
                "key": {
                    "type": "string",
                    "description": "Key for preference or user info"
                },
                "value": {
                    "type": "string",
                    "description": "Value for preference or user info"
                },
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID (for 'forget' operation)"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    
    def execute(
        self,
        operation: str,
        content: str = None,
        query: str = None,
        category: str = "general",
        importance: float = 0.5,
        key: str = None,
        value: str = None,
        memory_id: str = None
    ) -> Dict[str, Any]:
        """Execute memory operation"""
        
        print(f"🧠 Memory operation: {operation}")
        
        if operation == "remember":
            if not content:
                return {"success": False, "error": "Content required for remember operation"}
            
            mem_id = self.ltm.remember(
                content=content,
                memory_type="fact",
                category=category,
                importance=importance
            )
            
            return {
                "success": True,
                "message": f"Remembered: {content[:50]}...",
                "memory_id": mem_id
            }
        
        elif operation == "recall":
            if not query:
                return {"success": False, "error": "Query required for recall operation"}
            
            memories = self.ltm.recall(query=query, limit=5)
            
            if not memories:
                return {
                    "success": True,
                    "found": False,
                    "message": "No relevant memories found"
                }
            
            return {
                "success": True,
                "found": True,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "category": m.category,
                        "importance": m.importance
                    }
                    for m in memories
                ]
            }
        
        elif operation == "search":
            if not query:
                return {"success": False, "error": "Query required for search operation"}
            
            results = self.vector_store.search(query=query, top_k=5)
            
            return {
                "success": True,
                "results": [
                    {
                        "content": r.document.content,
                        "score": r.score,
                        "metadata": r.document.metadata
                    }
                    for r in results
                ]
            }
        
        elif operation == "set_preference":
            if not key or value is None:
                return {"success": False, "error": "Key and value required"}
            
            self.ltm.set_preference(key, value)
            
            return {
                "success": True,
                "message": f"Preference set: {key} = {value}"
            }
        
        elif operation == "get_preferences":
            prefs = self.ltm.get_all_preferences()
            
            return {
                "success": True,
                "preferences": prefs
            }
        
        elif operation == "set_user_info":
            if not key or value is None:
                return {"success": False, "error": "Key and value required"}
            
            self.ltm.set_user_info(key, value)
            
            return {
                "success": True,
                "message": f"User info set: {key} = {value}"
            }
        
        elif operation == "forget":
            if not memory_id:
                return {"success": False, "error": "Memory ID required"}
            
            success = self.ltm.forget(memory_id)
            
            return {
                "success": success,
                "message": "Memory forgotten" if success else "Memory not found"
            }
        
        elif operation == "get_context":
            if not query:
                query = ""
            
            context = self.ltm.get_relevant_context(query)
            
            return {
                "success": True,
                "context": context
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }


class KnowledgeBaseTool(BaseTool):
    """Tool for managing a knowledge base with documents"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        return "knowledge_base"
    
    @property
    def description(self) -> str:
        return """Manage a knowledge base of documents and information.
        Use this to store and search through documents, articles, or any text content.
        
        Operations:
        - 'add': Add a document to the knowledge base
        - 'search': Search for relevant documents
        - 'get': Get a specific document by ID
        - 'delete': Remove a document
        - 'list': List all documents
        """
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "search", "get", "delete", "list"],
                    "description": "The operation to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Document content (for 'add' operation)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search' operation)"
                },
                "doc_id": {
                    "type": "string",
                    "description": "Document ID (for 'get' and 'delete' operations)"
                },
                "title": {
                    "type": "string",
                    "description": "Document title (for 'add' operation)"
                },
                "source": {
                    "type": "string",
                    "description": "Document source (for 'add' operation)"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    
    def execute(
        self,
        operation: str,
        content: str = None,
        query: str = None,
        doc_id: str = None,
        title: str = None,
        source: str = None
    ) -> Dict[str, Any]:
        """Execute knowledge base operation"""
        
        print(f"📚 Knowledge base operation: {operation}")
        
        if operation == "add":
            if not content:
                return {"success": False, "error": "Content required"}
            
            metadata = {"title": title or "Untitled", "source": source or "unknown"}
            doc_id = self.vector_store.add_document(content=content, metadata=metadata)
            
            return {
                "success": True,
                "doc_id": doc_id,
                "message": f"Document added: {title or doc_id}"
            }
        
        elif operation == "search":
            if not query:
                return {"success": False, "error": "Query required"}
            
            results = self.vector_store.search(query=query, top_k=5)
            
            return {
                "success": True,
                "results": [
                    {
                        "id": r.document.id,
                        "content": r.document.content[:200] + "..." if len(r.document.content) > 200 else r.document.content,
                        "title": r.document.metadata.get("title", "Untitled"),
                        "score": round(r.score, 3)
                    }
                    for r in results
                ]
            }
        
        elif operation == "get":
            if not doc_id:
                return {"success": False, "error": "Document ID required"}
            
            doc = self.vector_store.get_document(doc_id)
            
            if doc:
                return {
                    "success": True,
                    "document": {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                }
            return {"success": False, "error": "Document not found"}
        
        elif operation == "delete":
            if not doc_id:
                return {"success": False, "error": "Document ID required"}
            
            success = self.vector_store.delete_document(doc_id)
            
            return {
                "success": success,
                "message": "Document deleted" if success else "Document not found"
            }
        
        elif operation == "list":
            docs = self.vector_store.get_all_documents(limit=20)
            
            return {
                "success": True,
                "count": len(docs),
                "documents": [
                    {
                        "id": d.id,
                        "title": d.metadata.get("title", "Untitled"),
                        "preview": d.content[:100] + "..." if len(d.content) > 100 else d.content
                    }
                    for d in docs
                ]
            }
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}