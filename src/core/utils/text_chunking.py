# src/core/utils/text_chunking.py
import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data with better error handling
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded with comprehensive fallback"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
    ]
    
    success = False
    for resource_path, resource_name in required_data:
        try:
            nltk.data.find(resource_path)
            success = True
            logger.debug(f"Found NLTK resource: {resource_name}")
            break  # If we find any working tokenizer, we're good
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
                success = True
                logger.info(f"Downloaded NLTK resource: {resource_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource_name}: {e}")
                continue
    
    if not success:
        logger.warning("Could not initialize NLTK sentence tokenizer. Will use fallback regex-based tokenization.")
    
    return success

# Initialize NLTK data at module level
_nltk_available = ensure_nltk_data()

def smart_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                    preserve_structure: bool = True) -> List[str]:
    """
    Enhanced text chunking that preserves semantic structure and context.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        preserve_structure: Whether to preserve document structure
        
    Returns:
        List of semantically coherent text chunks
    """
    if not text or not text.strip():
        return []
        
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    if preserve_structure:
        # First, try to split by major sections
        sections = _split_by_sections(text)
        
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = _split_by_paragraphs_and_sentences(section, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
    else:
        chunks = _split_by_paragraphs_and_sentences(text, chunk_size, chunk_overlap)
    
    # Add overlap between chunks for better context
    overlapped_chunks = _add_semantic_overlap(chunks, chunk_overlap)
    
    return overlapped_chunks

def _split_by_sections(text: str) -> List[str]:
    """Split text by major sections (headers, page breaks, etc.)"""
    # Detect section markers
    section_patterns = [
        r'\n\s*===.*?===\s*\n',  # === markers
        r'\n\s*#{1,3}\s+.*?\n',   # Markdown headers
        r'\n\s*\d+\.\s+[A-Z][^.]*\n',  # Numbered sections
        r'\n\s*[A-Z][^.]*:\s*\n',      # Title: format
        r'\n\s*Page\s+\d+\s*\n',       # Page breaks
    ]
    
    combined_pattern = '|'.join(section_patterns)
    sections = re.split(combined_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Filter out empty sections and very small ones
    valid_sections = [s.strip() for s in sections if len(s.strip()) > 50]
    
    return valid_sections if valid_sections else [text]

def _split_by_paragraphs_and_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by paragraphs first, then sentences if needed"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = _get_overlap_text(current_chunk, chunk_overlap)
            
            # If paragraph itself is too large, split by sentences
            if len(paragraph) > chunk_size:
                sentence_chunks = _split_by_sentences(paragraph, chunk_size, chunk_overlap)
                if current_chunk:
                    # Merge first sentence chunk with current
                    if sentence_chunks:
                        sentence_chunks[0] = current_chunk + " " + sentence_chunks[0]
                        current_chunk = ""
                
                if sentence_chunks:
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
            else:
                current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by sentences using NLTK or fallback regex"""
    global _nltk_available
    
    try:
        if _nltk_available:
            sentences = sent_tokenize(text)
        else:
            # Fallback to regex-based sentence splitting
            sentences = _regex_sentence_split(text)
    except Exception as e:
        logger.warning(f"Sentence tokenization failed: {e}, using regex fallback")
        sentences = _regex_sentence_split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = _get_overlap_text(current_chunk, chunk_overlap) + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def _regex_sentence_split(text: str) -> List[str]:
    """Fallback regex-based sentence splitting when NLTK is not available"""
    # Enhanced regex pattern for sentence boundaries
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n\s*(?=[A-Z])'
    
    # Split by the pattern
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter out very short fragments
            cleaned_sentences.append(sentence)
    
    # If regex splitting didn't work well, fall back to simple splitting
    if len(cleaned_sentences) < 2 and len(text) > 100:
        # Simple fallback: split by periods followed by spaces and capital letters
        simple_split = re.split(r'\.\s+(?=[A-Z])', text)
        cleaned_sentences = [s.strip() + '.' for s in simple_split if s.strip()]
        if cleaned_sentences and not cleaned_sentences[-1].endswith('.'):
            cleaned_sentences[-1] = cleaned_sentences[-1].rstrip('.') + '.'
    
    return cleaned_sentences if cleaned_sentences else [text]

def _get_overlap_text(text: str, overlap_size: int) -> str:
    """Get the last part of text for overlap"""
    if len(text) <= overlap_size:
        return text
    
    # Try to break at sentence boundary
    overlap_text = text[-overlap_size:]
    sentence_start = overlap_text.find('. ')
    if sentence_start != -1:
        return overlap_text[sentence_start + 2:]
    
    # Try to break at word boundary
    word_start = overlap_text.find(' ')
    if word_start != -1:
        return overlap_text[word_start + 1:]
    
    return overlap_text

def _add_semantic_overlap(chunks: List[str], overlap_size: int) -> List[str]:
    """Add meaningful overlap between chunks"""
    if len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        current_chunk = chunks[i]
        
        # Add overlap from previous chunk
        overlap = _get_overlap_text(prev_chunk, overlap_size)
        if overlap and overlap not in current_chunk[:len(overlap)+50]:
            overlapped_chunk = overlap + " " + current_chunk
        else:
            overlapped_chunk = current_chunk
            
        overlapped_chunks.append(overlapped_chunk)
    
    return overlapped_chunks

# Keep the original function for backward compatibility
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Legacy function - now uses smart chunking"""
    return smart_chunk_text(text, chunk_size, chunk_overlap)

def chunk_with_metadata(text: str, metadata: Dict[Any, Any], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Create chunks with associated metadata using smart chunking.
    """
    chunks = smart_chunk_text(text, chunk_size, chunk_overlap)
    
    result = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_count"] = len(chunks)
        chunk_metadata["chunk_type"] = _identify_chunk_type(chunk)
        
        result.append({
            "text": chunk,
            "metadata": chunk_metadata
        })
    
    return result

def _identify_chunk_type(chunk: str) -> str:
    """Identify the type of content in a chunk"""
    chunk_lower = chunk.lower()
    
    if any(word in chunk_lower for word in ['error', 'exception', 'failed', 'warning']):
        return 'error_log'
    elif any(word in chunk_lower for word in ['step', 'procedure', 'install', 'configure']):
        return 'procedure'
    elif any(word in chunk_lower for word in ['table', 'data', 'values', 'statistics']):
        return 'data'
    elif len(chunk.split()) < 50:  # Short chunks might be headers or summaries
        return 'summary'
    else:
        return 'content'