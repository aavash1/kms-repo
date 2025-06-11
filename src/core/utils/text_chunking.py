# src/core/utils/enhanced_text_chunking.py
import re
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_type: str
    confidence: float
    structure_markers: List[str]
    language_detected: Optional[str] = None
    formatting_preserved: bool = False

class RobustTextChunker:
    """
    Robust text chunker designed for highly unstructured documents
    with unpredictable formatting, mixed content types, and various file formats
    """
    
    def __init__(self):
        self.ensure_nltk_data()
        
        # Patterns for detecting different content types
        self.structure_patterns = {
            'table_like': [
                r'^\s*\|.*\|\s*$',  # Table rows with pipes
                r'^\s*[\d\w]+\s+[\d\w]+\s+[\d\w]+.*$',  # Column-like data
                r'^\s*[-=]{3,}\s*$',  # Table separators
            ],
            'list_items': [
                r'^\s*[•·▪▫◦‣⁃]\s+',  # Bullet points
                r'^\s*[\da-zA-Z][\.\)]\s+',  # Numbered/lettered lists
                r'^\s*[-*+]\s+',  # Dash/asterisk bullets
            ],
            'headers': [
                r'^\s*#{1,6}\s+.*$',  # Markdown headers
                r'^\s*[A-Z][^.!?]*:?\s*$',  # All caps potential headers
                r'^\s*\d+\.?\s*[A-Z][^.]*$',  # Numbered sections
            ],
            'code_or_technical': [
                r'[{}()\[\]<>]',  # Contains brackets/braces
                r'[=:]{2,}',  # Multiple equals or colons
                r'\b(function|class|def|var|int|string)\b',  # Code keywords
            ],
            'metadata_lines': [
                r'^\s*(Page|페이지)\s*\d+',  # Page numbers
                r'^\s*(Date|날짜|작성일)[:：]\s*',  # Date lines
                r'^\s*(File|파일|문서)[:：]\s*',  # File references
            ]
        }
    
    def ensure_nltk_data(self):
        """Ensure required NLTK data with fallback"""
        try:
            nltk.data.find('tokenizers/punkt')
            self.nltk_available = True
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                self.nltk_available = True
            except:
                self.nltk_available = False
                logger.warning("NLTK not available, using regex-based splitting")
    
    def smart_chunk_unstructured_text(self, text: str, chunk_size: int = 1000, 
                                    chunk_overlap: int = 200, 
                                    preserve_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Main chunking method optimized for unstructured documents
        
        Returns list of dictionaries with 'text' and 'metadata' keys
        """
        if not text or not text.strip():
            return []
        
        # Preprocess text to handle common formatting issues
        processed_text = self._preprocess_unstructured_text(text)
        
        if len(processed_text) <= chunk_size:
            return [{
                'text': processed_text,
                'metadata': ChunkMetadata(
                    chunk_type='single_chunk',
                    confidence=1.0,
                    structure_markers=[]
                ).__dict__
            }]
        
        # Analyze text structure
        structure_analysis = self._analyze_text_structure(processed_text)
        
        # Choose chunking strategy based on structure analysis
        if structure_analysis['has_clear_structure']:
            chunks = self._structure_aware_chunking(processed_text, chunk_size, chunk_overlap)
        else:
            chunks = self._robust_unstructured_chunking(processed_text, chunk_size, chunk_overlap)
        
        # Add overlap and finalize
        final_chunks = self._add_intelligent_overlap(chunks, chunk_overlap)
        
        return final_chunks
    
    def _preprocess_unstructured_text(self, text: str) -> str:
        """
        Preprocess text to handle common formatting issues in unstructured documents
        """
        # Remove excessive whitespace while preserving intentional spacing
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        
        # Handle common OCR artifacts
        text = re.sub(r'[^\S\n]{3,}', ' ', text)  # Multiple non-newline whitespace
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix sentence spacing
        
        # Handle page breaks and headers/footers
        text = re.sub(r'\n\s*(Page|페이지)\s*\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Clean up line breaks around punctuation
        text = re.sub(r'\n+([.,:;!?])', r'\1', text)
        
        return text.strip()
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to determine its structural characteristics
        """
        lines = text.split('\n')
        analysis = {
            'has_clear_structure': False,
            'dominant_patterns': [],
            'line_count': len(lines),
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'structure_confidence': 0.0
        }
        
        pattern_matches = {pattern_type: 0 for pattern_type in self.structure_patterns}
        
        for line in lines:
            for pattern_type, patterns in self.structure_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        pattern_matches[pattern_type] += 1
                        break
        
        # Determine if text has clear structure
        total_lines = len([line for line in lines if line.strip()])
        if total_lines > 0:
            structure_ratio = sum(pattern_matches.values()) / total_lines
            analysis['structure_confidence'] = structure_ratio
            analysis['has_clear_structure'] = structure_ratio > 0.3
            
            # Find dominant patterns
            analysis['dominant_patterns'] = [
                pattern_type for pattern_type, count in pattern_matches.items()
                if count > total_lines * 0.1
            ]
        
        return analysis
    
    def _structure_aware_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Chunking that attempts to preserve document structure
        """
        chunks = []
        
        # Split by major structural boundaries first
        sections = self._split_by_structural_boundaries(text)
        
        for section in sections:
            if len(section) <= chunk_size:
                chunk_metadata = self._analyze_chunk_content(section)
                chunks.append({
                    'text': section,
                    'metadata': chunk_metadata.__dict__
                })
            else:
                # Further split large sections
                sub_chunks = self._split_large_section(section, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _robust_unstructured_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Fallback chunking for highly unstructured text
        """
        chunks = []
        
        # Try paragraph-based splitting first
        paragraphs = self._extract_paragraphs(text)
        
        current_chunk = ""
        current_metadata_markers = []
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunk_metadata = ChunkMetadata(
                        chunk_type='unstructured_content',
                        confidence=0.7,
                        structure_markers=current_metadata_markers
                    )
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': chunk_metadata.__dict__
                    })
                
                # Start new chunk
                current_chunk = paragraph
                current_metadata_markers = self._extract_structure_markers(paragraph)
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                current_metadata_markers.extend(self._extract_structure_markers(paragraph))
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = ChunkMetadata(
                chunk_type='unstructured_content',
                confidence=0.7,
                structure_markers=current_metadata_markers
            )
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata.__dict__
            })
        
        return chunks
    
    def _split_by_structural_boundaries(self, text: str) -> List[str]:
        """
        Split text by detected structural boundaries
        """
        # Enhanced section splitting patterns
        boundary_patterns = [
            r'\n\s*={3,}\s*\n',  # === separators
            r'\n\s*-{3,}\s*\n',  # --- separators
            r'\n\s*#{1,3}\s+[^#\n]+\n',  # Markdown headers
            r'\n\s*\d+\.\s*[A-Z][^\n]{10,}\n',  # Numbered sections
            r'\n\s*[A-Z][^.!?\n]{20,}:\s*\n',  # Title: format
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in boundary_patterns)
        sections = re.split(combined_pattern, text, flags=re.MULTILINE)
        
        # Filter and clean sections
        valid_sections = []
        for section in sections:
            if section and len(section.strip()) > 30:  # Minimum section size
                valid_sections.append(section.strip())
        
        return valid_sections if valid_sections else [text]
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from unstructured text
        """
        # Split by double newlines first
        potential_paragraphs = text.split('\n\n')
        
        paragraphs = []
        for para in potential_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is very long, try to split by sentences
            if len(para) > 500:
                sentences = self._split_into_sentences(para)
                if len(sentences) > 1:
                    paragraphs.extend(sentences)
                else:
                    paragraphs.append(para)
            else:
                paragraphs.append(para)
        
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with fallback for unstructured content
        """
        if self.nltk_available:
            try:
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except:
                pass
        
        # Fallback regex-based sentence splitting
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and validate sentences
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                valid_sentences.append(sentence)
        
        return valid_sentences if valid_sentences else [text]
    
    def _split_large_section(self, section: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Split large sections while preserving as much structure as possible
        """
        chunks = []
        
        # Try splitting by paragraphs first
        paragraphs = self._extract_paragraphs(section)
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunk_metadata = self._analyze_chunk_content(current_chunk)
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': chunk_metadata.__dict__
                    })
                current_chunk = paragraph
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
        
        if current_chunk.strip():
            chunk_metadata = self._analyze_chunk_content(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata.__dict__
            })
        
        return chunks
    
    def _analyze_chunk_content(self, chunk: str) -> ChunkMetadata:
        """
        Analyze chunk content to determine its type and characteristics
        """
        chunk_lower = chunk.lower()
        structure_markers = self._extract_structure_markers(chunk)
        
        # Determine chunk type
        if any(word in chunk_lower for word in ['error', 'exception', 'failed', 'warning', '오류', '실패']):
            chunk_type = 'error_log'
            confidence = 0.9
        elif any(word in chunk_lower for word in ['step', 'procedure', 'install', 'configure', '단계', '절차']):
            chunk_type = 'procedure'
            confidence = 0.8
        elif len(structure_markers) > 0:
            chunk_type = 'structured_content'
            confidence = 0.8
        elif any(pattern in chunk for pattern in ['|', '---', '===', '```']):
            chunk_type = 'technical_content'
            confidence = 0.7
        elif len(chunk.split()) < 50:
            chunk_type = 'summary'
            confidence = 0.6
        else:
            chunk_type = 'general_content'
            confidence = 0.5
        
        return ChunkMetadata(
            chunk_type=chunk_type,
            confidence=confidence,
            structure_markers=structure_markers,
            formatting_preserved=len(structure_markers) > 0
        )
    
    def _extract_structure_markers(self, text: str) -> List[str]:
        """
        Extract structural markers from text
        """
        markers = []
        
        for pattern_type, patterns in self.structure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    markers.append(pattern_type)
                    break
        
        return markers
    
    def _add_intelligent_overlap(self, chunks: List[Dict[str, Any]], overlap_size: int) -> List[Dict[str, Any]]:
        """
        Add intelligent overlap between chunks based on content type
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]['text']
            current_chunk = chunks[i]['text']
            current_metadata = chunks[i]['metadata'].copy()
            
            # Determine overlap strategy based on chunk types
            prev_type = chunks[i-1]['metadata'].get('chunk_type', 'general_content')
            current_type = current_metadata.get('chunk_type', 'general_content')
            
            if prev_type == 'procedure' and current_type == 'procedure':
                # For procedural content, use larger overlap
                size=int(min(overlap_size * 1.5, 300))
                overlap = self._get_semantic_overlap(prev_chunk, size)
            elif prev_type == 'technical_content' or current_type == 'technical_content':
                # For technical content, use precise overlap
                overlap = self._get_semantic_overlap(prev_chunk, overlap_size)
            else:
                # Default overlap
                overlap = self._get_semantic_overlap(prev_chunk, overlap_size)
            
            if overlap and overlap not in current_chunk[:len(overlap)+50]:
                overlapped_text = overlap + '\n' + current_chunk
                current_metadata['has_overlap'] = True
            else:
                overlapped_text = current_chunk
                current_metadata['has_overlap'] = False
            
            overlapped_chunks.append({
                'text': overlapped_text,
                'metadata': current_metadata
            })
        
        return overlapped_chunks
    
    def _get_semantic_overlap(self, text: str, overlap_size: int) -> str:
        """
        Get semantically meaningful overlap from text
        """
        overlap_size=int(overlap_size)
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        
        # Try to break at sentence boundary
        sentence_end = max(
            overlap_text.rfind('. '),
            overlap_text.rfind('! '),
            overlap_text.rfind('? ')
        )
        
        if sentence_end > overlap_size * 0.3:  # At least 30% of overlap
            return overlap_text[sentence_end + 2:]
        
        # Try to break at paragraph boundary
        para_break = overlap_text.rfind('\n\n')
        if para_break > overlap_size * 0.2:  # At least 20% of overlap
            return overlap_text[para_break + 2:]
        
        # Try to break at word boundary
        word_break = overlap_text.rfind(' ')
        if word_break > overlap_size * 0.1:  # At least 10% of overlap
            return overlap_text[word_break + 1:]
        
        return overlap_text

# Backward compatibility functions
def smart_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                    preserve_structure: bool = True) -> List[str]:
    """
    Backward compatible function that returns only text chunks
    """
    chunker = RobustTextChunker()
    chunks_with_metadata = chunker.smart_chunk_unstructured_text(
        text, chunk_size, chunk_overlap, preserve_structure
    )
    return [chunk['text'] for chunk in chunks_with_metadata]

def chunk_with_metadata(text: str, metadata: Dict[Any, Any], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Enhanced version that includes both original metadata and chunk analysis
    """
    chunker = RobustTextChunker()
    chunks_with_analysis = chunker.smart_chunk_unstructured_text(
        text, chunk_size, chunk_overlap
    )
    
    result = []
    for i, chunk_data in enumerate(chunks_with_analysis):
        combined_metadata = metadata.copy()
        combined_metadata.update({
            "chunk_index": i,
            "chunk_count": len(chunks_with_analysis),
            "chunk_analysis": chunk_data['metadata']
        })
        
        result.append({
            "text": chunk_data['text'],
            "metadata": combined_metadata
        })
    
    return result