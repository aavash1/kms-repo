#src/core/file_handlers/rtf_handler.py
# rtf_handler.py
import logging
import asyncio
from typing import List, Optional
import re

from .base_handler import FileHandler

logger = logging.getLogger(__name__)

class RTFHandler(FileHandler):
    """RTFHandler extracts text from RTF (Rich Text Format) files."""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        logger.debug("RTFHandler initialized.")

    def _strip_rtf_formatting(self, rtf_content: str) -> str:
        """
        Strip RTF formatting codes to extract plain text.
        This is a simplified RTF parser for basic text extraction.
        """
        try:
            # Remove RTF header
            if rtf_content.startswith('{\\rtf'):
                # Find the first occurrence of actual text after RTF controls
                text = rtf_content
            else:
                text = rtf_content
            
            # Remove RTF control words and groups
            # Remove control words like \rtf1, \ansi, \deff0, etc.
            text = re.sub(r'\\[a-z]+\d*\s?', ' ', text)
            
            # Remove control symbols like \', \{, \}
            text = re.sub(r'\\[^a-z\s]', '', text)
            
            # Remove remaining braces
            text = re.sub(r'[{}]', '', text)
            
            # Clean up multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Remove any remaining backslashes
            text = text.replace('\\', '')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error stripping RTF formatting: {e}")
            return rtf_content

    async def extract_text(self, file_path: str) -> str:
        """Extract text from an RTF file."""
        try:
            # Try to import striprtf for better RTF parsing
            try:
                from striprtf.striprtf import rtf_to_text
                use_striprtf = True
            except ImportError:
                logger.warning("striprtf library not available, using basic RTF parser")
                use_striprtf = False
            
            # Read the RTF file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    rtf_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ['cp1252', 'iso-8859-1', 'ascii']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            rtf_content = f.read()
                        logger.debug(f"Successfully read RTF file with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, read as binary and decode with error handling
                    with open(file_path, 'rb') as f:
                        raw_content = f.read()
                    rtf_content = raw_content.decode('utf-8', errors='replace')
                    logger.warning(f"Used UTF-8 with error replacement for RTF file: {file_path}")
            
            # Extract text using available method
            if use_striprtf:
                try:
                    text = rtf_to_text(rtf_content)
                except Exception as e:
                    logger.warning(f"striprtf failed, falling back to basic parser: {e}")
                    text = self._strip_rtf_formatting(rtf_content)
            else:
                text = self._strip_rtf_formatting(rtf_content)
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from RTF file: {file_path}")
                return ""
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from RTF file {file_path}: {e}")
            return ""

    async def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from RTF content in memory.
        
        Args:
            file_content: Raw bytes of the RTF file.
            
        Returns:
            str: Extracted text, or empty string if extraction fails.
        """
        try:
            # Try to import striprtf for better RTF parsing
            try:
                from striprtf.striprtf import rtf_to_text
                use_striprtf = True
            except ImportError:
                logger.warning("striprtf library not available, using basic RTF parser")
                use_striprtf = False
            
            # Decode the RTF content
            try:
                rtf_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ['cp1252', 'iso-8859-1', 'ascii']:
                    try:
                        rtf_content = file_content.decode(encoding)
                        logger.debug(f"Successfully decoded RTF content with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use utf-8 with error handling
                    rtf_content = file_content.decode('utf-8', errors='replace')
                    logger.warning("Used UTF-8 with error replacement for RTF content in memory")
            
            # Extract text using available method
            if use_striprtf:
                try:
                    text = rtf_to_text(rtf_content)
                except Exception as e:
                    logger.warning(f"striprtf failed, falling back to basic parser: {e}")
                    text = self._strip_rtf_formatting(rtf_content)
            else:
                text = self._strip_rtf_formatting(rtf_content)
            
            if not text or not text.strip():
                logger.warning("No text extracted from RTF content in memory")
                return ""
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from RTF content in memory: {e}")
            return ""

    async def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """RTF files may contain tables, but this basic implementation doesn't extract them."""
        return []