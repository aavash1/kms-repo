# hwp_handler.py
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

    async def extract_text(self, file_path: str) -> str:
        """Extract text from HWP file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Run CPU-intensive operations in a thread pool
            def process_hwp():
                # Open HWP file
                f = olefile.OleFileIO(file_path)
                try:
                    dirs = f.listdir()

                    # Validate HWP file
                    if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
                        raise Exception("Not a valid HWP file.")

                    # Check if document is compressed
                    header = f.openstream("FileHeader")
                    header_data = header.read()
                    is_compressed = (header_data[36] & 1) == 1

                    # Get all body sections
                    sections = self._get_body_sections(dirs)
                    
                    # Extract text content
                    text_parts = []
                    
                    # Get text from sections
                    for section in sections:
                        section_text = self._extract_section_text(f, section, is_compressed)
                        if section_text:
                            text_parts.append(section_text)

                    # Clean and combine all text
                    combined_text = '\n\n'.join(text_parts)
                    cleaned_text = self._clean_text(combined_text)
                    
                    logger.debug(f"Extracted text length: {len(cleaned_text)}")
                    return cleaned_text
                finally:
                    f.close()
                    
            # Execute in thread pool
            result = await asyncio.to_thread(process_hwp)
            return result

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

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
                            # Optional: Use TrOCR for handwritten text if detected
                            if any(ord(c) > 0x4E00 for c in text):  # Simple check for CJK characters
                                image_data = self._convert_text_to_image(text)  # Hypothetical method
                                text = self._extract_handwritten_text(image_data) or text
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