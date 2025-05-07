# hwp_handler.py
import os
import logging
from pathlib import Path
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import olefile
import zlib
import struct
import binascii
import re
import torch
import asyncio

from .base_handler import FileHandler
from src.core.ocr.tesseract_wrapper import TesseractOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModel, AutoTokenizer
from src.core.config import load_ocr_config

logger = logging.getLogger(__name__)

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
    
    def __init__(self, model_manager=None):
        self.ocr = TesseractOCR()
        self.config = load_ocr_config().get('hwp', {})
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_regex = re.compile('|'.join(self.ARTIFACTS_PATTERNS))

        self.model_manager = model_manager
        if model_manager:
            self.device = model_manager.get_device()
            self.trocr_processor = model_manager.get_trocr_processor()
            self.trocr_model = model_manager.get_trocr_model()
            self.bert_tokenizer = model_manager.get_klue_tokenizer()
            self.bert_model = model_manager.get_klue_bert()
        else:
            self.device = None
            self.trocr_processor = None
            self.trocr_model = None
            self.bert_tokenizer = None
            self.bert_model = None
        
        logger.debug("HWPHandler initialized with temp directory: %s", self.temp_dir.name)

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            logger.debug("HWPHandler temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"HWPHandler temporary directory cleanup failed: {e}")

    async def extract_text(self, file_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """Extract text from HWP file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Run CPU-intensive operations in a thread pool
            def process_hwp():
                # Open HWP file
                try:
                    logger.debug(f"Opening HWP file: {file_path}")
                    f = olefile.OleFileIO(file_path)
                    
                    try:
                        dirs = f.listdir()
                        logger.debug(f"HWP directories: {dirs}")

                        # Check if this is a valid HWP file
                        if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
                            raise Exception("Not a valid HWP file.")

                        # Check if document is compressed
                        header = f.openstream("FileHeader")
                        header_data = header.read()
                        is_compressed = (header_data[36] & 1) == 1
                        logger.debug(f"HWP file is compressed: {is_compressed}")

                        # Get all body sections
                        sections = self._get_body_sections(dirs)
                        logger.debug(f"Found {len(sections)} sections: {sections}")
                        
                        # Extract text content
                        text_parts = []
                        
                        # Get text from sections
                        for section in sections:
                            try:
                                section_text = self._extract_section_text(f, section, is_compressed)
                                if section_text:
                                    text_parts.append(section_text)
                            except Exception as e:
                                logger.error(f"Error extracting text from section {section}: {e}", exc_info=True)
                                continue

                        # If no text was extracted, try alternative method
                        if not text_parts:
                            logger.warning("No text extracted using primary method, trying alternative...")
                            try:
                                alt_text = self._extract_alternative_text(f, is_compressed)
                                if alt_text:
                                    text_parts.append(alt_text)
                            except Exception as e:
                                logger.error(f"Alternative text extraction failed: {e}", exc_info=True)

                        # Clean and combine all text
                        combined_text = '\n\n'.join(text_parts)
                        cleaned_text = self._clean_text(combined_text)
                        
                        logger.debug(f"Extracted text length: {len(cleaned_text)}")
                        return cleaned_text, []  # Empty list for tables, which we don't extract separately
                    finally:
                        f.close()
                except Exception as e:
                    logger.error(f"HWP text extraction failed: {e}", exc_info=True)
                    return "", []
                
            # Execute in thread pool
            text, tables = await asyncio.to_thread(process_hwp)
            return text, tables

        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            return "", []

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
        """Extract text from a single section with improved error handling."""
        try:
            # Read section data
            bodytext = f.openstream(section)
            data = bodytext.read()

            # Decompress if needed
            if is_compressed:
                try:
                    # Use wbits=-15 for raw deflate data
                    unpacked_data = zlib.decompress(data, -15)
                except zlib.error:
                    try:
                        # Try with zlib header/trailer (default)
                        unpacked_data = zlib.decompress(data)
                    except zlib.error:
                        logger.warning(f"Failed to decompress section {section} with standard methods")
                        # Try with alternative decompression
                        unpacked_data = self._alternative_decompress(data)
            else:
                unpacked_data = data

            # Extract text from section
            section_text = []
            i = 0
            size = len(unpacked_data)
            
            while i < size:
                try:
                    if i + 4 > size:
                        break
                        
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    rec_len = (header >> 20) & 0xfff

                    if rec_len == 0 or i + 4 + rec_len > size:
                        # Invalid record length, skip to next byte
                        i += 1
                        continue

                    if rec_type in [67]:  # Text record type
                        rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                        
                        # Try different encodings for Korean text
                        for encoding in ['utf-16-le', 'utf-16', 'cp949', 'euc-kr']:
                            try:
                                text = rec_data.decode(encoding)
                                if text.strip():
                                    cleaned = self._clean_line(text)
                                    if cleaned.strip():
                                        section_text.append(cleaned)
                                break  # If successful, stop trying encodings
                            except UnicodeDecodeError:
                                continue
                                
                    i += 4 + rec_len
                except (struct.error, UnicodeError) as e:
                    logger.warning(f"Error processing record at position {i}: {e}")
                    i += 1  # Skip problematic record

            return '\n'.join(section_text)

        except Exception as e:
            logger.error(f"Error extracting text from section {section}: {e}", exc_info=True)
            raise

    def _alternative_decompress(self, data):
        """Try alternative decompression methods."""
        try:
            # Try skipping first few bytes which might be header
            for skip in [2, 4, 8, 10]:
                if len(data) <= skip:
                    continue
                try:
                    return zlib.decompress(data[skip:], -15)
                except zlib.error:
                    pass
                    
            # If none of the above worked, raise an error
            raise ValueError("Could not decompress data with any method")
        except Exception as e:
            logger.error(f"Alternative decompression failed: {e}")
            raise

    def _extract_alternative_text(self, ole_file, is_compressed):
        """Extract text using an alternative approach for problematic files."""
        try:
            # Try to extract from document summary
            if ole_file.exists("\x05DocumentSummaryInformation"):
                summary = ole_file.openstream("\x05DocumentSummaryInformation")
                summary_data = summary.read()
                # Look for text patterns in the summary
                text = self._extract_text_from_binary(summary_data)
                if text:
                    return text
                    
            # Try scanning all streams for text content
            for stream_path in ole_file.listdir():
                if isinstance(stream_path, list) and len(stream_path) > 0:
                    path = "/".join(stream_path)
                    try:
                        stream = ole_file.openstream(path)
                        data = stream.read()
                        
                        # If compressed, try to decompress
                        if is_compressed and b'PK\x03\x04' in data[:10]:  # Check for ZIP signature
                            try:
                                data = zlib.decompress(data, -15)
                            except:
                                pass
                                
                        # Extract text from binary data
                        text = self._extract_text_from_binary(data)
                        if text:
                            return text
                    except:
                        continue
                        
            return ""
        except Exception as e:
            logger.error(f"Alternative text extraction failed: {e}")
            return ""

    def _extract_text_from_binary(self, data):
        """Try to extract readable text from binary data."""
        # Check for UTF-16 text which is common in HWP
        try:
            # Look for sequences of valid UTF-16 characters
            text_chunks = []
            for i in range(0, len(data) - 1, 2):
                chunk = data[i:i+2000]  # Process in chunks
                try:
                    decoded = chunk.decode('utf-16-le')
                    # Only keep chunks with actual text (at least some Korean chars)
                    if any(0xAC00 <= ord(c) <= 0xD7A3 for c in decoded):
                        text_chunks.append(decoded)
                except:
                    pass
                    
            if text_chunks:
                return "\n".join(text_chunks)
                
            # Try EUC-KR as a fallback for Korean text
            text_chunks = []
            for i in range(0, len(data), 100):
                chunk = data[i:i+1000]
                try:
                    decoded = chunk.decode('euc-kr')
                    if any(0xAC00 <= ord(c) <= 0xD7A3 for c in decoded):
                        text_chunks.append(decoded)
                except:
                    pass
                    
            if text_chunks:
                return "\n".join(text_chunks)
        except Exception as e:
            logger.debug(f"Text extraction from binary failed: {e}")
            
        return ""

    def _clean_line(self, text: str) -> str:
        """Clean a single line of text."""
        if not text:
            return ""
            
        # Remove binary artifacts
        text = self.artifacts_regex.sub('', text)
        
        # Remove any remaining non-printable characters except newlines
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n')
        
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

    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Tables are extracted as part of the text content."""
        # Tables are included in the text content for HWP files
        # Just return an empty list
        return []

    def _extract_handwritten_text(self, image_data):
        try:
            if not self.trocr_model or not self.trocr_processor:
                return ""
                
            with torch.no_grad():
                inputs = self.trocr_processor(images=image_data, return_tensors="pt").to(self.trocr_model.device)
                generated_ids = self.trocr_model.generate(inputs.pixel_values)
                text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return text.strip()
        except Exception as e:
            logger.error(f"Handwritten text extraction failed: {str(e)}")
            return ""

    def _convert_text_to_image(self, text):
        # Placeholder: Convert text to image for TrOCR (e.g., using PIL)
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 50), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, fill='black')
        return img