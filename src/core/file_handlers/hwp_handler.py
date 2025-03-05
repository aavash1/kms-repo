import os
import logging
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional
import olefile
import zlib
import struct
import binascii
import re

from .base_handler import FileHandler
from src.core.ocr.tesseract_wrapper import TesseractOCR
from src.core.config import load_ocr_config

logger = logging.getLogger(__name__)

def extract_status_codes(text):
    """Extract status codes from text using regex patterns."""
    import re
    patterns = [
        r'[Ss]tatus\s+[Cc]ode\s+(\d+)',  # Matches "Status Code XXX" or "status code XXX"
        r'[Ss]tatus\s+(\d+)',            # Matches "Status XXX" or "status XXX"
    ]
    status_codes = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            status_codes.add(match.group(1))
    return list(status_codes)

class HWPHandler(FileHandler):
    """HWPHandler extracts text from HWP files."""
    
    # Add binary artifacts patterns to remove
    ARTIFACTS_PATTERNS = [
        r'[╗]',                  # Table borders
        r'[捤獥]',               # Table artifacts
        r'[汤捯]',               # Table artifacts
        r'[氠瑢]',               # Table artifacts
        r'[\x00-\x1F\x7F-\xFF]'  # Control characters and extended ASCII
    ]
    
    def __init__(self):
        self.ocr = TesseractOCR()
        self.config = load_ocr_config().get('hwp', {})
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_regex = re.compile('|'.join(self.ARTIFACTS_PATTERNS))
        self.status_codes = []
        logger.debug("HWPHandler initialized with temp directory: %s", self.temp_dir.name)

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            logger.debug("HWPHandler temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"HWPHandler temporary directory cleanup failed: {e}")

    def get_status_codes(self):
        """Return the list of status codes found in the last processed document."""
        return self.status_codes
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from HWP file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Open HWP file
            f = olefile.OleFileIO(file_path)
            dirs = f.listdir()

            # Validate HWP file
            if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
                raise Exception("Not a valid HWP file.")

            self.status_codes = []
            # Check if document is compressed
            header = f.openstream("FileHeader")
            header_data = header.read()
            is_compressed = (header_data[36] & 1) == 1

            # Get all body sections
            sections = self._get_body_sections(dirs)
            
            # Extract text content
            text_parts = []
            all_status_codes=set()

            
            # Get text from sections
            for section in sections:
                section_text = self._extract_section_text(f, section, is_compressed)
                if section_text:
                    text_parts.append(section_text)
                    section_codes = extract_status_codes(section_text)
                    all_status_codes.update(section_codes)

            # Clean and combine all text
            combined_text = '\n\n'.join(text_parts)
            cleaned_text = self._clean_text(combined_text)

            self.status_codes = list(all_status_codes)
            
            logger.debug(f"Extracted text length: {len(cleaned_text)}")
            return cleaned_text

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            self.status_codes=[]
            return ""
        finally:
            if 'f' in locals():
                f.close()

    def _get_body_sections(self, dirs: List[List[str]]) -> List[str]:
        """Get all body section paths in order."""
        nums = []
        for d in dirs:
            if d[0] == "BodyText":
                try:
                    section_num = int(d[1][len("Section"):])
                    nums.append(section_num)
                except ValueError:
                    continue
        
        return ["BodyText/Section" + str(x) for x in sorted(nums)]

    def _extract_section_text(self, f: olefile.OleFileIO, section: str, is_compressed: bool) -> str:
        """Extract text from a single section."""
        try:
            # Read section data
            bodytext = f.openstream(section)
            data = bodytext.read()

            # Decompress if needed
            if is_compressed:
                try:
                    unpacked_data = zlib.decompress(data, -15)
                except zlib.error:
                    logger.warning(f"Failed to decompress section {section}")
                    return ""
            else:
                unpacked_data = data

            # Extract text from section
            section_text = []
            i = 0
            size = len(unpacked_data)
            
            while i < size:
                try:
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    rec_len = (header >> 20) & 0xfff

                    if rec_type in [67]:  # Text record type
                        rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                        text = rec_data.decode('utf-16')
                        if text.strip():
                            # Clean the text before adding
                            cleaned = self._clean_line(text)
                            if cleaned.strip():
                                section_text.append(cleaned)

                    i += 4 + rec_len
                except (struct.error, UnicodeDecodeError) as e:
                    logger.warning(f"Error processing record at position {i}: {e}")
                    i += 4  # Skip problematic record

            return '\n'.join(section_text)

        except Exception as e:
            logger.error(f"Error extracting text from section {section}: {e}")
            return ""

    def _clean_line(self, text: str) -> str:
        """Clean a single line of text."""
        # Remove binary artifacts
        text = self.artifacts_regex.sub('', text)
        
        # Remove any remaining non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char in '\n')
        
        return text.strip()

    def _clean_text(self, text: str) -> str:
        """Clean and format the extracted text."""
        if not text:
            return ""

        # Split into lines and clean
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            line = self._clean_line(line)
            if not line:
                cleaned_lines.append('')
                continue

            # Handle list items and indentation
            if line.startswith(('1.', '2.', '3.', '4.', '->', '-', 'ㅇ')):
                cleaned_lines.append('    ' + line)
            elif ':' in line:
                # Format key-value pairs
                key, value = line.split(':', 1)
                cleaned_lines.append(f"{key.strip()}: {value.strip()}")
            else:
                cleaned_lines.append(line)

        # Remove consecutive empty lines
        result = []
        prev_empty = False
        for line in cleaned_lines:
            is_empty = not bool(line.strip())
            if not (is_empty and prev_empty):
                result.append(line)
            prev_empty = is_empty

        return '\n'.join(result)

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Tables are extracted as part of the text content."""
        return []