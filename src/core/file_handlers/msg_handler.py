# src/core/file_handlers/msg_handler.py
"""
Enhanced MSG handler with smart OCR for attachments.
"""

import logging
import re
from typing import List, Dict, Any, Optional
import extract_msg
import tempfile
import os
from pathlib import Path
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel
import asyncio

from .base_handler import FileHandler
from .image_handler import ImageHandler
from .pdf_handler import PDFHandler
from .doc_handler import AdvancedDocHandler

logger = logging.getLogger(__name__)

class MSGHandler(FileHandler):
    """
    Enhanced MSGHandler with smart OCR for processing email attachments.
    Uses the hybrid PDF handler for better attachment processing.
    """
    
    def __init__(self, model_manager=None, use_smart_ocr=True):
        self.image_handler = ImageHandler(model_manager=model_manager, languages=['ko', 'en'])
        
        # Use the enhanced handlers with smart OCR
        self.pdf_handler = PDFHandler(model_manager=model_manager, use_smart_ocr=use_smart_ocr)
        self.doc_handler = AdvancedDocHandler(model_manager=model_manager, use_smart_ocr=use_smart_ocr)
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.use_smart_ocr = use_smart_ocr
        
        self.model_manager = model_manager
        if model_manager:
            self.device = model_manager.get_device()
            self.handwritten_processor = model_manager.get_trocr_processor()
            self.handwritten_model = model_manager.get_trocr_model()
            self.bert_tokenizer = model_manager.get_klue_tokenizer()
            self.bert_model = model_manager.get_klue_bert()
        else:
            self.device = None
            self.handwritten_processor = None
            self.handwritten_model = None
            self.bert_tokenizer = None
            self.bert_model = None
        
        logger.debug("Enhanced MSGHandler initialized with smart OCR support")

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            logger.debug("MSGHandler temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"MSGHandler temporary directory cleanup failed: {e}")

    async def extract_text(self, file_path: str) -> str:
        """Extract text from an MSG file with smart attachment processing."""
        try:
            # Run extract_msg operations in a thread pool
            def process_msg():
                msg = extract_msg.Message(file_path)
                body = self._extract_body(msg)
                return msg, body
                
            msg, body = await asyncio.to_thread(process_msg)
            text_parts = []

            # Extract email body
            if body:
                text_parts.append("=== EMAIL BODY ===")
                text_parts.append(body)

            # Process attachments with smart OCR
            attachment_texts = await self._process_attachments_smart(msg)
            if attachment_texts:
                text_parts.append("=== ATTACHMENT TEXTS ===")
                text_parts.extend(attachment_texts)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from MSG file {file_path}: {e}")
            return ""

    async def extract_text_from_memory(self, file_content: bytes) -> str:
        """Extract text from MSG file content in memory with smart processing."""
        try:
            # Create a temporary file to store the MSG content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.msg') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Extract text using the existing method
            text = await self.extract_text(temp_file_path)

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

    async def _process_attachments_smart(self, msg: extract_msg.Message) -> List[str]:
        """Process attachments with smart OCR using enhanced handlers."""
        attachment_texts = []
        temp_dir = Path(self.temp_dir.name)

        # Get all attachments first
        attachment_tasks = []
        attachment_info = []
        
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
                
                attachment_info.append((filename, temp_file_path, ext))
                
                # Process based on file type using enhanced handlers
                if ext in ['.png', '.jpg', '.jpeg']:
                    attachment_tasks.append(self._process_image_attachment(temp_file_path, filename))
                elif ext == '.pdf':
                    attachment_tasks.append(self._process_pdf_attachment(temp_file_path, filename))
                elif ext in ['.doc', '.docx']:
                    attachment_tasks.append(self._process_doc_attachment(temp_file_path, filename))
                else:
                    logger.debug(f"Unsupported attachment type: {ext} for {filename}")
                    attachment_tasks.append(self._create_placeholder_result(filename, "Unsupported file type"))

            except Exception as e:
                logger.warning(f"Error processing attachment {attachment.longFilename}: {e}")
                attachment_tasks.append(self._create_placeholder_result(
                    attachment.longFilename or "unknown", f"Processing error: {e}"
                ))
                continue

        # Run all tasks concurrently
        if attachment_tasks:
            results = await asyncio.gather(*attachment_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Error processing attachment {attachment_info[i][0]}: {result}")
                    continue
                    
                if result and isinstance(result, str):
                    attachment_texts.append(result)
                
                # Clean up temporary file
                try:
                    temp_file_path = attachment_info[i][1]
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

        return attachment_texts

    async def _process_image_attachment(self, file_path: Path, filename: str) -> str:
        """Process image attachment with OCR."""
        try:
            text = await self.image_handler.extract_text(str(file_path))
            if text:
                return f"📷 Image Attachment ({filename}):\n{text}"
            else:
                return f"📷 Image Attachment ({filename}): No text detected"
        except Exception as e:
            logger.warning(f"Failed to process image attachment {filename}: {e}")
            return f"📷 Image Attachment ({filename}): Processing failed"

    async def _process_pdf_attachment(self, file_path: Path, filename: str) -> str:
        """Process PDF attachment with smart OCR."""
        try:
            text = await self.pdf_handler.extract_text(str(file_path))
            if text:
                return f"📄 PDF Attachment ({filename}):\n{text}"
            else:
                return f"📄 PDF Attachment ({filename}): No text extracted"
        except Exception as e:
            logger.warning(f"Failed to process PDF attachment {filename}: {e}")
            return f"📄 PDF Attachment ({filename}): Processing failed"

    async def _process_doc_attachment(self, file_path: Path, filename: str) -> str:
        """Process DOC/DOCX attachment with smart OCR."""
        try:
            text = await self.doc_handler.extract_text(str(file_path))
            if text:
                return f"📝 Document Attachment ({filename}):\n{text}"
            else:
                return f"📝 Document Attachment ({filename}): No text extracted"
        except Exception as e:
            logger.warning(f"Failed to process document attachment {filename}: {e}")
            return f"📝 Document Attachment ({filename}): Processing failed"

    async def _create_placeholder_result(self, filename: str, message: str) -> str:
        """Create a placeholder result for unsupported or failed attachments."""
        return f"📎 Attachment ({filename}): {message}"

    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from attachments in the MSG file using enhanced handlers."""
        tables = []
        try:
            # Open the MSG file in a thread
            def open_msg():
                return extract_msg.Message(file_path)
                
            msg = await asyncio.to_thread(open_msg)
            temp_dir = Path(self.temp_dir.name)

            # Track concurrent tasks
            table_tasks = []
            table_info = []  # To keep track of file paths

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

                    # Extract tables based on file type using enhanced handlers
                    if ext == '.pdf':
                        table_tasks.append(self.pdf_handler.extract_tables(str(temp_file_path)))
                        table_info.append(temp_file_path)
                    elif ext in ['.doc', '.docx']:
                        table_tasks.append(self.doc_handler.extract_tables(str(temp_file_path)))
                        table_info.append(temp_file_path)
                    else:
                        table_info.append(None)

                except Exception as e:
                    logger.warning(f"Error extracting tables from attachment {attachment.longFilename}: {e}")
                    continue

            # Process all table extraction tasks concurrently
            if table_tasks:
                results = await asyncio.gather(*table_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Error extracting tables: {result}")
                    else:
                        tables.extend(result)
                        
                    # Clean up temp files
                    temp_file_path = table_info[i]
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete {temp_file_path}: {e}")

            # Close MSG file
            await asyncio.to_thread(msg.close)
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from MSG file {file_path}: {e}")
            return []

    def get_handler_status(self) -> dict:
        """Get current handler status for debugging."""
        status = {
            "handler": "Enhanced MSG Handler",
            "smart_ocr_enabled": self.use_smart_ocr,
            "enhanced_handlers": {
                "pdf_handler": self.pdf_handler.get_handler_status() if hasattr(self.pdf_handler, 'get_handler_status') else "Available",
                "doc_handler": self.doc_handler.get_handler_status() if hasattr(self.doc_handler, 'get_handler_status') else "Available",
                "image_handler": "Available"
            },
            "ml_models_loaded": {
                "trocr": self.handwritten_model is not None,
                "bert": self.bert_model is not None,
            }
        }
        
        return status