# src/core/utils/text_chunking.py
import re
from typing import List, Dict, Any, Optional

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters with specified overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: The target size of each chunk
        chunk_overlap: The number of characters to overlap between chunks
        
    Returns:
        A list of text chunks
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're at the end of the text, just use the end
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a good breaking point (paragraph, sentence, or word boundary)
        # First try to break at a paragraph
        next_para = text.find('\n\n', start + chunk_size // 2, end)
        if next_para != -1:
            end = next_para + 2  # Include the newlines
        else:
            # Try to break at a sentence
            next_sentence = text.find('. ', start + chunk_size // 2, end)
            if next_sentence != -1:
                end = next_sentence + 2  # Include the period and space
            else:
                # Fall back to breaking at a word boundary
                next_space = text.rfind(' ', start + chunk_size // 2, end)
                if next_space != -1:
                    end = next_space + 1  # Include the space
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move the start point, accounting for overlap
        start = max(start + chunk_size - chunk_overlap, end - chunk_overlap)
    
    return chunks

def chunk_with_metadata(text: str, metadata: Dict[Any, Any], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Create chunks with associated metadata.
    
    Args:
        text: The text to split into chunks
        metadata: Metadata dict to associate with each chunk
        chunk_size: The target size of each chunk
        chunk_overlap: The number of characters to overlap between chunks
        
    Returns:
        A list of dicts with 'text' and 'metadata' keys
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    result = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        # Add chunk information to metadata
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_count"] = len(chunks)
        
        result.append({
            "text": chunk,
            "metadata": chunk_metadata
        })
    
    return result