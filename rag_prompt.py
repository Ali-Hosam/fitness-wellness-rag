"""
rag_prompt.py - Prompt Formatting Module

PURPOSE:
Takes retrieved chunks and formats them into a well-structured prompt for the LLM.

WHY:
LLMs work better with explicit instructions and structured context. This module:
1. Takes messy database results (tuples with columns)
2. Formats them into readable context
3. Creates a clear prompt template that tells the LLM how to answer
4. Limits context size to avoid overwhelming the model

DESIGN:
The prompt template uses a "System Message" pattern:
- System: Instructions for how to behave
- Context: The knowledge base excerpts
- Question: What the user is asking
"""

from typing import List, Tuple

# Type hint for clarity:
# A chunk is: (chunk_id, chapter, section, source, chunk_text, word_count, similarity)
RetrievedChunk = Tuple[int, str, str, str, str, int, float]


def format_chunks_as_context(chunks: List[RetrievedChunk]) -> str:
    """
    Convert database chunks into readable context for the LLM.
    
    Args:
        chunks: List of tuples from search_similar_chunks()
        
    Returns:
        Formatted context string ready to include in a prompt
        
    WHY NEEDED:
    - Database returns raw tuples; we need human-readable text
    - Each chunk includes metadata (chapter, section, similarity)
    - We need to present this in a way that helps the LLM understand the source
    """
    
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id, chapter, section, source, chunk_text, word_count, similarity = chunk
        
        # Format: [Source 1 of 5]
        # Chapter: Healthy Behaviors
        # Section: Dimensions of Wellness
        # Relevance: 0.77 (how well it matches the question)
        # Text: ...
        
        context_block = f"""[Source {i} of {len(chunks)}]
Chapter: {chapter}
Section: {section if section else 'N/A'}
Relevance Score: {similarity:.2f}
---
{chunk_text}
---"""
        
        context_parts.append(context_block)
    
    # Join all chunks with a separator
    return "\n\n".join(context_parts)

# history=[]
# system_message = """You are a helpful fitness and wellness expert assistant. 
# Your role is to answer questions about wellness, fitness, and nutrition based ONLY on the provided knowledge base.

# IMPORTANT RULES:
# 1. Answer ONLY based on the provided context
# 2. If the context doesn't contain information to answer the question, say "I don't have enough information to answer this"
# 3. Be specific and cite which part of the context you're using
# 4. Keep your answer clear and concise
# 5. Do not add information from outside sources
# 6. answer in a formal and friendly tone, suitable for a general audience
# 7.the user current and previous questions are associated with role user
# 8.the assistant previous answers are associated with role assistant"""
# history.append({"role": "system", "content": system_message})

def create_rag_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """
    Create a complete prompt that combines:
    1. System instructions (how to behave)
    2. Context (knowledge base information)
    3. User question (what they're asking)
    
    Args:
        question: The user's question
        chunks: Retrieved relevant chunks
        
    Returns:
        Complete prompt ready to send to LLM
        
    WHY STRUCTURED LIKE THIS:
    - System message sets boundaries (answer based ONLY on context, be honest if you don't know)
    - Context gives the knowledge
    - Question is clear and specific
    - Separation makes it clear to the LLM what is instruction vs. content vs. question
    """
    # history.append({"role": "user", "content": question})
    context = format_chunks_as_context(chunks)
    
#     system_message = """You are a helpful fitness and wellness expert assistant. 
# Your role is to answer questions about wellness, fitness, and nutrition based ONLY on the provided knowledge base.

# IMPORTANT RULES:
# 1. Answer ONLY based on the provided context
# 2. If the context doesn't contain information to answer the question, say "I don't have enough information to answer this"
# 3. Be specific and cite which part of the context you're using
# 4. Keep your answer clear and concise
# 5. Do not add information from outside sources
# 6. answer in a formal and friendly tone, suitable for a general audience"""
    
    # prompt = f"""{system_message}
    prompt = f"""
    
===== KNOWLEDGE BASE CONTEXT =====
{context}

===== USER QUESTION =====
Question: {question}

Please provide a comprehensive answer based ONLY on the context above."""
    
    return prompt


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count for text.
    
    WHY:
    LLMs have token limits (e.g., Gemini has different limits).
    We estimate tokens to avoid sending too much context.
    
    This is a rough approximation: ~4 characters = 1 token (varies by model)
    """
    return len(text) // 4


def truncate_context_if_needed(context: str, max_tokens: int = 2000) -> str:
    """
    If context is too large, remove lower-relevance chunks.
    
    WHY:
    We retrieved 5 chunks, but they might total more tokens than the LLM can handle.
    Instead of sending partial/cut-off text, we remove entire chunks starting with
    the lowest relevance scores.
    """
    estimated_tokens = estimate_tokens(context)
    
    if estimated_tokens <= max_tokens:
        return context
    
    # If we need to truncate, we'd parse chunks and remove the lowest ones
    # For now, this is a placeholder - we'll handle it in the generator
    print(f"WARNING: Context is ~{estimated_tokens} tokens, max is {max_tokens}. May need truncation.")
    return context


if __name__ == "__main__":
    # EXAMPLE USAGE
    from LLM_retrieval import search_similar_chunks
    
    question = "What are the nine dimensions of wellness?"
    chunks = search_similar_chunks(question, limit=5)
    
    # Step 1: Format context
    context = format_chunks_as_context(chunks)
    print("=== FORMATTED CONTEXT ===")
    print(context)
    print("\n")
    
    # Step 2: Create full prompt
    prompt = create_rag_prompt(question, chunks)
    print("=== FULL PROMPT (for LLM) ===")
    print(prompt)
    print("\n")
    
    # Step 3: Check token estimate
    tokens = estimate_tokens(prompt)
    print(f"Estimated token count: {tokens}")
