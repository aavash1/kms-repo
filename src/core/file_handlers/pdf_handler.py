# pdf_handler.py
import logging
import re
from typing import List, Dict, Any, Optional
import pdfplumber
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel
import asyncio

from .base_handler import FileHandler
from .image_handler import ImageHandler

logger = logging.getLogger(__name__)

class PDFHandler(FileHandler):
    """PDFHandler extracts text, tables, and images from PDF files."""
    
    def __init__(self, model_manager=None):
        self.image_handler = ImageHandler(model_manager=model_manager, languages=['ko', 'en'])
        
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        self.bert_tokenizer = model_manager.get_klue_tokenizer()
        self.bert_model = model_manager.get_klue_bert()
        
        logger.debug("PDFHandler initialized.")

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a PDF file, including text from embedded images."""
        try:
            # Run synchronous fitz.open in a thread
            doc = await asyncio.to_thread(fitz.open, file_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text directly from the page
                page_text = await asyncio.to_thread(page.get_text, "text")
                if page_text:
                    text_parts.append(page_text)

                # Extract images and perform OCR
                image_texts = await self._extract_images_from_page(page, page_num)
                if image_texts:
                    text_parts.append(f"=== Page {page_num + 1} Images ===\n" + "\n".join(image_texts))

            # Close the document in a thread
            await asyncio.to_thread(doc.close)
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    async def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from PDF content in memory.
        
        Args:
            file_content: Raw bytes of the PDF file.
            
        Returns:
            str: Extracted text, or empty string if extraction fails.
        """
        try:
            # Run synchronous fitz.open in a thread
            doc = await asyncio.to_thread(fitz.open, stream=file_content, filetype="pdf")
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text directly from the page
                page_text = await asyncio.to_thread(page.get_text, "text")
                if page_text:
                    text_parts.append(page_text)

                # Extract images and perform OCR
                image_texts = await self._extract_images_from_page(page, page_num)
                if image_texts:
                    text_parts.append(f"=== Page {page_num + 1} Images ===\n" + "\n".join(image_texts))

            # Close the document in a thread
            await asyncio.to_thread(doc.close)
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from PDF content in memory: {e}")
            return ""

    async def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[str]:
        """Extract images from a PDF page and perform OCR on them."""
        image_texts = []
        # Get images synchronously in a thread
        image_list = await asyncio.to_thread(page.get_images, full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                # Extract image in a thread
                base_image = await asyncio.to_thread(page.parent.extract_image, xref)
                image_bytes = base_image["image"]
                
                # Convert image bytes to a format suitable for OCR
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img_np is None:
                    logger.warning(f"Failed to decode image on page {page_num + 1}, image {img_index + 1}")
                    continue

                # Use ImageHandler to extract text from the image (async)
                image_text = await self.image_handler.extract_text_from_memory(image_bytes)
                if image_text:
                    image_texts.append(f"Image {img_index + 1}:\n{image_text}")

            except Exception as e:
                logger.warning(f"Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                continue

        return image_texts

    async def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """Extract tables from a PDF file using pdfplumber."""
        tables = []
        try:
            # Open and process with pdfplumber in a thread
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract tables in a thread
                    page_tables = await asyncio.to_thread(page.extract_tables)
                    for table in page_tables:
                        # Clean table data
                        cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                        tables.append(cleaned_table)
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from PDF {file_path}: {e}")
            return []