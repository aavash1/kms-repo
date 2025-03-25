# msg_handler.py
import logging
import re
from typing import List, Dict, Any, Optional
import extract_msg
import tempfile
import os
from pathlib import Path
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel

from .base_handler import FileHandler
from .image_handler import ImageHandler
from .pdf_handler import PDFHandler
from .doc_handler import AdvancedDocHandler

logger = logging.getLogger(__name__)

class MSGHandler(FileHandler):
    """MSGHandler extracts text from Outlook MSG files, including attachments."""
    
    def __init__(self, model_manager=None):
        self.image_handler = ImageHandler(model_manager=model_manager, languages=['ko', 'en'])
        self.pdf_handler = PDFHandler(model_manager=model_manager)
        self.doc_handler = AdvancedDocHandler(model_manager=model_manager)
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        self.bert_tokenizer = model_manager.get_klue_tokenizer()
        self.bert_model = model_manager.get_klue_bert()
        
        logger.debug("MSGHandler initialized with temp directory: %s", self.temp_dir.name)

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            logger.debug("MSGHandler temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"MSGHandler temporary directory cleanup failed: {e}")

    def extract_text(self, file_path: str) -> str:
        """Extract text from an MSG file, including its body and attachments."""
        try:
            msg = extract_msg.Message(file_path)
            text_parts = []

            # Extract email body
            body = self._extract_body(msg)
            if body:
                text_parts.append(body)

            # Process attachments
            attachment_texts = self._process_attachments(msg)
            if attachment_texts:
                text_parts.append("=== ATTACHMENT TEXTS ===\n" + "\n".join(attachment_texts))

            msg.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from MSG file {file_path}: {e}")
            return ""

    def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from MSG file content in memory.
        
        Args:
            file_content: Raw bytes of the MSG file.
            
        Returns:
            str: Extracted text, or empty string if extraction fails.
        """
        try:
            # Create a temporary file to store the MSG content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.msg') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Extract text using the existing method
            text = self.extract_text(temp_file_path)

            # Clean up the temporary file
            os.unlink(temp_file_path)
            return text

        except Exception as e:
            logger.error(f"Error extracting text from MSG content in memory: {e}")
            return ""

    def _extract_body(self, msg: extract_msg.Message) -> str:
        """Extract the body of the email, preferring plain text over HTML."""
        try:
            # Prefer plain text body if available
            if msg.body:
                return msg.body.strip()

            # Fallback to HTML body if plain text is not available
            if msg.htmlBody:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(msg.htmlBody, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                return text.strip()

            return ""

        except Exception as e:
            logger.warning(f"Error extracting email body: {e}")
            return ""

    def _process_attachments(self, msg: extract_msg.Message) -> List[str]:
        """Process attachments in the MSG file and extract text from supported file types."""
        attachment_texts = []
        temp_dir = Path(self.temp_dir.name)

        for attachment in msg.attachments:
            try:
                # Skip if attachment data is not available
                if not attachment.data:
                    logger.debug(f"Skipping attachment with no data: {attachment.longFilename}")
                    continue

                # Determine file extension
                filename = attachment.longFilename or attachment.shortFilename or "attachment"
                ext = Path(filename).suffix.lower()

                # Save attachment to a temporary file
                temp_file_path = temp_dir / filename
                with open(temp_file_path, 'wb') as f:
                    f.write(attachment.data)

                # Process based on file type
                if ext in ['.png', '.jpg', '.jpeg']:
                    text = self.image_handler.extract_text(str(temp_file_path))
                    if text:
                        attachment_texts.append(f"Image Attachment ({filename}):\n{text}")
                elif ext == '.pdf':
                    text = self.pdf_handler.extract_text(str(temp_file_path))
                    if text:
                        attachment_texts.append(f"PDF Attachment ({filename}):\n{text}")
                elif ext in ['.doc', '.docx']:
                    text = self.doc_handler.extract_text(str(temp_file_path))
                    if text:
                        attachment_texts.append(f"Document Attachment ({filename}):\n{text}")
                else:
                    logger.debug(f"Unsupported attachment type: {ext} for {filename}")

                # Clean up temporary file
                temp_file_path.unlink()

            except Exception as e:
                logger.warning(f"Error processing attachment {attachment.longFilename}: {e}")
                continue

        return attachment_texts

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from attachments in the MSG file."""
        tables = []
        try:
            msg = extract_msg.Message(file_path)
            temp_dir = Path(self.temp_dir.name)

            for attachment in msg.attachments:
                try:
                    if not attachment.data:
                        continue

                    filename = attachment.longFilename or attachment.shortFilename or "attachment"
                    ext = Path(filename).suffix.lower()

                    # Save attachment to a temporary file
                    temp_file_path = temp_dir / filename
                    with open(temp_file_path, 'wb') as f:
                        f.write(attachment.data)

                    # Extract tables based on file type
                    if ext == '.pdf':
                        attachment_tables = self.pdf_handler.extract_tables(str(temp_file_path))
                        tables.extend(attachment_tables)
                    elif ext in ['.doc', '.docx']:
                        attachment_tables = self.doc_handler.extract_tables(str(temp_file_path))
                        tables.extend(attachment_tables)

                    temp_file_path.unlink()

                except Exception as e:
                    logger.warning(f"Error extracting tables from attachment {attachment.longFilename}: {e}")
                    continue

            msg.close()
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from MSG file {file_path}: {e}")
            return []