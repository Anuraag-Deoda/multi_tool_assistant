# ============================================================================
# FILE: memory/conversation_memory.py
# ============================================================================

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Message:
    """Represents a conversation message"""
    role: str
    content: str
    timestamp: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)


@dataclass
class ConversationSummary:
    """Represents a conversation summary"""
    summary: str
    message_count: int
    start_time: str
    end_time: str
    key_topics: List[str] = None
    
    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationMemory:
    """
    Manages conversation history with summarization capabilities.
    
    Features:
    - Store and retrieve conversation messages
    - Automatic summarization when context gets too long
    - Rolling window for active context
    - Persistent storage of conversation history
    """
    
    def __init__(
        self,
        config: Any,
        summarizer_fn: Optional[callable] = None,
        storage_path: str = "./conversations"
    ):
        self.config = config
        self.summarizer_fn = summarizer_fn  # LLM function for summarization
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current conversation
        self.messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_messages = 0
        
        # Load previous conversations
        self._load_previous_summaries()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.total_messages += 1
        
        # Check if summarization is needed
        if len(self.messages) >= self.config.CONVERSATION_WINDOW_SIZE:
            self._trigger_summarization()
        
        return message
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM context.
        Includes recent summaries + active messages.
        """
        context = []
        
        # Add summaries as system context
        if self.summaries:
            summary_text = self._format_summaries_for_context()
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary_text}"
            })
        
        # Add recent messages (limited by MAX_CONTEXT_MESSAGES)
        recent_messages = self.messages[-self.config.MAX_CONTEXT_MESSAGES:]
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return context
    
    def get_full_history(self) -> List[Message]:
        """Get complete conversation history"""
        return self.messages.copy()
    
    def _trigger_summarization(self):
        """Summarize older messages to maintain context window"""
        if not self.summarizer_fn:
            # Simple fallback if no summarizer provided
            self._simple_summarization()
            return
        
        # Get messages to summarize (keep recent ones active)
        messages_to_summarize = self.messages[:-self.config.MAX_CONTEXT_MESSAGES]
        
        if len(messages_to_summarize) < 5:  # Not enough to summarize
            return
        
        print("📝 Summarizing conversation history...")
        
        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(messages_to_summarize)
        
        # Generate summary using LLM
        summary_prompt = f"""Summarize the following conversation, capturing:
1. Key topics discussed
2. Important information learned
3. Any user preferences or facts
4. Decisions made or actions taken

Conversation:
{conversation_text}

Provide a concise summary:"""
        
        try:
            summary_text = self.summarizer_fn(summary_prompt)
            
            # Extract key topics (simple extraction)
            topics = self._extract_topics(summary_text)
            
            summary = ConversationSummary(
                summary=summary_text,
                message_count=len(messages_to_summarize),
                start_time=messages_to_summarize[0].timestamp,
                end_time=messages_to_summarize[-1].timestamp,
                key_topics=topics
            )
            
            self.summaries.append(summary)
            
            # Remove summarized messages from active list
            self.messages = self.messages[-self.config.MAX_CONTEXT_MESSAGES:]
            
            # Save summary to disk
            self._save_summary(summary)
            
            print(f"✅ Summarized {summary.message_count} messages")
            
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            self._simple_summarization()
    
    def _simple_summarization(self):
        """Fallback simple summarization without LLM"""
        messages_to_summarize = self.messages[:-self.config.MAX_CONTEXT_MESSAGES]
        
        if len(messages_to_summarize) < 3:
            return
        
        # Create simple summary
        topics = []
        for msg in messages_to_summarize:
            if msg.role == "user":
                # Extract first few words as topic hint
                words = msg.content.split()[:5]
                topics.append(" ".join(words))
        
        summary = ConversationSummary(
            summary=f"Previous discussion covered: {'; '.join(topics[:5])}",
            message_count=len(messages_to_summarize),
            start_time=messages_to_summarize[0].timestamp,
            end_time=messages_to_summarize[-1].timestamp,
            key_topics=topics[:5]
        )
        
        self.summaries.append(summary)
        self.messages = self.messages[-self.config.MAX_CONTEXT_MESSAGES:]
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages into readable text for summarization"""
        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def _format_summaries_for_context(self) -> str:
        """Format summaries for inclusion in context"""
        if not self.summaries:
            return ""
        
        # Take most recent summaries
        recent_summaries = self.summaries[-3:]  # Last 3 summaries
        
        parts = []
        for i, summary in enumerate(recent_summaries, 1):
            parts.append(f"[Session {i}] {summary.summary}")
            if summary.key_topics:
                parts.append(f"  Topics: {', '.join(summary.key_topics[:5])}")
        
        return "\n".join(parts)
    
    def _extract_topics(self, summary_text: str) -> List[str]:
        """Extract key topics from summary text"""
        # Simple keyword extraction
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 
                       'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'must',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                       'from', 'as', 'into', 'through', 'during', 'before',
                       'after', 'above', 'below', 'between', 'under', 'again',
                       'further', 'then', 'once', 'and', 'but', 'or', 'nor',
                       'so', 'yet', 'both', 'each', 'few', 'more', 'most',
                       'other', 'some', 'such', 'no', 'not', 'only', 'own',
                       'same', 'than', 'too', 'very', 'just', 'also'}
        
        words = summary_text.lower().split()
        topics = []
        for word in words:
            word_clean = ''.join(c for c in word if c.isalnum())
            if len(word_clean) > 4 and word_clean not in common_words:
                if word_clean not in topics:
                    topics.append(word_clean)
        
        return topics[:10]
    
    def _save_summary(self, summary: ConversationSummary):
        """Save summary to disk"""
        filepath = self.storage_path / f"summary_{self.session_id}.json"
        
        existing_summaries = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_summaries = json.load(f)
        
        existing_summaries.append(summary.to_dict())
        
        with open(filepath, 'w') as f:
            json.dump(existing_summaries, f, indent=2)
    
    def _load_previous_summaries(self):
        """Load summaries from previous sessions"""
        summary_files = sorted(self.storage_path.glob("summary_*.json"))
        
        for filepath in summary_files[-5:]:  # Load last 5 session summaries
            try:
                with open(filepath, 'r') as f:
                    summaries_data = json.load(f)
                    for s_data in summaries_data:
                        self.summaries.append(ConversationSummary(**s_data))
            except Exception as e:
                print(f"⚠️ Failed to load summary {filepath}: {e}")
    
    def save_conversation(self):
        """Save current conversation to disk"""
        filepath = self.storage_path / f"conversation_{self.session_id}.json"
        
        data = {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "summaries": [s.to_dict() for s in self.summaries],
            "total_messages": self.total_messages
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def search_history(self, query: str, limit: int = 5) -> List[Message]:
        """Simple keyword search in conversation history"""
        query_lower = query.lower()
        results = []
        
        for msg in reversed(self.messages):
            if query_lower in msg.content.lower():
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            "session_id": self.session_id,
            "active_messages": len(self.messages),
            "total_messages": self.total_messages,
            "summaries_count": len(self.summaries),
            "topics_discussed": list(set(
                topic 
                for s in self.summaries 
                for topic in (s.key_topics or [])
            ))
        }
    
    def clear(self):
        """Clear current conversation"""
        self.save_conversation()  # Save before clearing
        self.messages = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")