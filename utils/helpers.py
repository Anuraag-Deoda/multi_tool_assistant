# ============================================================================
# FILE: utils/helpers.py (Updated with reasoning utilities)
# ============================================================================

import re
from typing import List, Dict, Any, Tuple
from datetime import datetime


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract potential entities from text"""
    entities = {
        "names": [],
        "locations": [],
        "numbers": [],
        "dates": [],
        "emails": []
    }
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities["emails"] = re.findall(email_pattern, text)
    
    # Extract numbers
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    entities["numbers"] = re.findall(number_pattern, text)
    
    # Extract dates (simple patterns)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    ]
    for pattern in date_patterns:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
    
    return entities


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to human readable"""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return iso_timestamp


def calculate_similarity_score(text1: str, text2: str) -> float:
    """Simple word overlap similarity score"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    return text.strip()


# ==========================================================================
# Reasoning Utilities (NEW)
# ==========================================================================

def estimate_complexity(text: str) -> Tuple[int, List[str]]:
    """
    Estimate the complexity of a query/task.
    
    Returns:
        Tuple of (complexity_score 1-10, list of complexity indicators)
    """
    indicators = []
    score = 1
    
    text_lower = text.lower()
    
    # Check for multi-step indicators
    multi_step_patterns = [
        (r'\b(first|then|next|after|finally)\b', "sequential steps"),
        (r'\b(and|also|additionally)\b.*\b(and|also|additionally)\b', "multiple requirements"),
        (r'\d+\s*(step|phase|stage)', "numbered steps"),
    ]
    
    for pattern, indicator in multi_step_patterns:
        if re.search(pattern, text_lower):
            score += 1
            indicators.append(indicator)
    
    # Check for analytical keywords
    analytical_keywords = [
        "analyze", "compare", "evaluate", "assess", "investigate",
        "determine", "calculate", "derive", "prove", "demonstrate"
    ]
    for keyword in analytical_keywords:
        if keyword in text_lower:
            score += 1
            indicators.append(f"analytical: {keyword}")
            break
    
    # Check for open-ended questions
    open_ended_patterns = [
        (r'\b(how|why|what if)\b', "open-ended question"),
        (r'\b(best|optimal|most effective)\b', "optimization query"),
        (r'\b(pros and cons|advantages|disadvantages)\b', "comparison request"),
    ]
    
    for pattern, indicator in open_ended_patterns:
        if re.search(pattern, text_lower):
            score += 1
            indicators.append(indicator)
    
    # Check text length
    word_count = len(text.split())
    if word_count > 50:
        score += 1
        indicators.append("long query")
    if word_count > 100:
        score += 1
        indicators.append("very long query")
    
    # Check for multiple questions
    question_count = text.count('?')
    if question_count > 1:
        score += question_count - 1
        indicators.append(f"{question_count} questions")
    
    # Cap at 10
    score = min(10, score)
    
    return score, indicators


def format_reasoning_steps(steps: List[str], numbered: bool = True) -> str:
    """Format reasoning steps for display"""
    if numbered:
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    else:
        return "\n".join([f"• {step}" for step in steps])


def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """Extract key points from text"""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Filter and clean
    points = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20 and len(sent) < 200:
            points.append(sent)
    
    # Return top sentences (could be enhanced with importance scoring)
    return points[:max_points]


def merge_reasoning_chains(chains: List[Dict]) -> Dict:
    """Merge multiple reasoning chains into one"""
    if not chains:
        return {"steps": [], "conclusion": ""}
    
    if len(chains) == 1:
        return chains[0]
    
    merged_steps = []
    conclusions = []
    
    for i, chain in enumerate(chains):
        merged_steps.append(f"Approach {i+1}:")
        for step in chain.get("steps", []):
            merged_steps.append(f"  - {step}")
        conclusions.append(chain.get("conclusion", ""))
    
    return {
        "steps": merged_steps,
        "conclusions": conclusions,
        "merged": True
    }


def validate_plan_structure(plan: Dict) -> Tuple[bool, List[str]]:
    """Validate a plan structure"""
    errors = []
    
    if "steps" not in plan:
        errors.append("Missing 'steps' field")
        return False, errors
    
    steps = plan["steps"]
    if not steps:
        errors.append("Plan has no steps")
        return False, errors
    
    step_ids = set()
    for i, step in enumerate(steps):
        # Check required fields
        if "step_id" not in step:
            errors.append(f"Step {i} missing 'step_id'")
        else:
            step_ids.add(step["step_id"])
        
        if "description" not in step:
            errors.append(f"Step {i} missing 'description'")
    
    # Check dependencies
    for step in steps:
        for dep in step.get("dependencies", []):
            if dep not in step_ids:
                errors.append(f"Step {step.get('step_id', '?')} has unknown dependency: {dep}")
    
    return len(errors) == 0, errors