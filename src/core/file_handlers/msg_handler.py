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
import asyncio

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
        
        logger.debug("MSGHandler initialized with temp directory: %s", self.temp_dir.name)

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            logger.debug("MSGHandler temporary directory cleaned up.")
        except Exception as e:
            logger.warning(f"MSGHandler temporary directory cleanup failed: {e}")

    async def extract_text(self, file_path: str) -> str:
        """Extract text from an MSG file, including its body and attachments."""
        try:
            # Run extract_msg operations in a thread pool
            def process_msg():
                msg = extract_msg.Message(file_path)
                body = self._extract_body(msg)
                msg.close()
                return msg, body
                
            msg, body = await asyncio.to_thread(process_msg)
            text_parts = []

            # Extract email body
            if body:
                text_parts.append(body)

            # Process attachments asynchronously
            attachment_texts = await self._process_attachments(msg)
            if attachment_texts:
                text_parts.append("=== ATTACHMENT TEXTS ===\n" + "\n".join(attachment_texts))

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from MSG file {file_path}: {e}")
            return ""

    async def extract_text_from_memory(self, file_content: bytes) -> str:
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

    async def _process_attachments(self, msg: extract_msg.Message) -> List[str]:
        """Process attachments in the MSG file and extract text from supported file types."""
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
                
                # Process based on file type
                if ext in ['.png', '.jpg', '.jpeg']:
                    attachment_tasks.append(self.image_handler.extract_text(str(temp_file_path)))
                elif ext == '.pdf':
                    attachment_tasks.append(self.pdf_handler.extract_text(str(temp_file_path)))
                elif ext in ['.doc', '.docx']:
                    attachment_tasks.append(self.doc_handler.extract_text(str(temp_file_path)))
                else:
                    logger.debug(f"Unsupported attachment type: {ext} for {filename}")
                    attachment_tasks.append(None)

            except Exception as e:
                logger.warning(f"Error processing attachment {attachment.longFilename}: {e}")
                attachment_tasks.append(None)
                continue

        # Run all tasks concurrently
        if attachment_tasks:
            results = await asyncio.gather(*[task for task in attachment_tasks if task is not None], 
                                          return_exceptions=True)
            
            # Match results with filenames and create formatted outputs
            result_idx = 0
            for i, (filename, temp_file_path, ext) in enumerate(attachment_info):
                if attachment_tasks[i] is None:
                    continue  # Skip unsupported types
                    
                text = results[result_idx]
                result_idx += 1
                
                # Handle exceptions
                if isinstance(text, Exception):
                    logger.warning(f"Error extracting text from {filename}: {text}")
                    continue
                    
                if text and isinstance(text, str):
                    type_name = "Image" if ext in ['.png', '.jpg', '.jpeg'] else "PDF" if ext == '.pdf' else "Document"
                    attachment_texts.append(f"{type_name} Attachment ({filename}):\n{text}")
                
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

        return attachment_texts

    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from attachments in the MSG file."""
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

                    # Extract tables based on file type
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