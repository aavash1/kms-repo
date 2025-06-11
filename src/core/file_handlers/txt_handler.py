#src/core/file_handlers/txt_handler.py
# txt_handler.py
import logging
import asyncio
from typing import List, Optional
import chardet

from .base_handler import FileHandler

logger = logging.getLogger(__name__)

class TXTHandler(FileHandler):
    """TXTHandler extracts text from plain text files with encoding detection."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        logger.debug("TXTHandler initialized.")

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a TXT file with automatic encoding detection."""
        try:
            # Read file in binary mode first to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Detect encoding
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result.get('encoding', 'utf-8')
            confidence = encoding_result.get('confidence', 0)
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Try detected encoding first, fallback to common encodings
            encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'ascii']
            
            for enc in encodings_to_try:
                if enc is None:
                    continue
                try:
                    text = raw_data.decode(enc)
                    logger.debug(f"Successfully decoded with encoding: {enc}")
                    return text
                except (UnicodeDecodeError, LookupError):
                    logger.debug(f"Failed to decode with encoding: {enc}")
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            text = raw_data.decode('utf-8', errors='replace')
            logger.warning(f"Used UTF-8 with error replacement for file: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from TXT file {file_path}: {e}")
            return ""

    async def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from TXT content in memory with encoding detection.
        
        Args:
            file_content: Raw bytes of the TXT file.
            
        Returns:
            str: Extracted text, or empty string if extraction fails.
        """
        try:
            # Detect encoding
            encoding_result = chardet.detect(file_content)
            encoding = encoding_result.get('encoding', 'utf-8')
            confidence = encoding_result.get('confidence', 0)
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Try detected encoding first, fallback to common encodings
            encodings_to_try = [encoding, 'utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'ascii']
            
            for enc in encodings_to_try:
                if enc is None:
                    continue
                try:
                    text = file_content.decode(enc)
                    logger.debug(f"Successfully decoded with encoding: {enc}")
                    return text
                except (UnicodeDecodeError, LookupError):
                    logger.debug(f"Failed to decode with encoding: {enc}")
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            text = file_content.decode('utf-8', errors='replace')
            logger.warning("Used UTF-8 with error replacement for TXT content in memory")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from TXT content in memory: {e}")
            return ""

    async def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """TXT files don't contain structured tables, return empty list."""
        return []