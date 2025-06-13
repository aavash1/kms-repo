# File: src/core/file_handlers/pdf_handler.py
"""
Hybrid PDF Handler that combines your existing ML models with smart OCR strategy.
Uses your ocr_config.yaml while maintaining your current model pipeline.
"""

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
from src.core.models.model_manager import ModelManager
from ..services.ocr_service import get_ocr_service
from src.core.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class PDFHandler(FileHandler):
    """
    Enhanced PDFHandler that combines ML models with intelligent OCR strategy.
    Supports both your existing pipeline and new configuration-based approach.
    """
    
    def __init__(self, model_manager=None, use_smart_ocr=True):
        """
        Initialize PDF handler with optional smart OCR strategy.
        
        Args:
            model_manager: Your existing ModelManager
            use_smart_ocr: Whether to use intelligent OCR strategy (True) or simple threshold (False)
        """
        if model_manager is None:
            model_manager = ModelManager()
            
        # Your existing setup
        self.image_handler = ImageHandler(model_manager=model_manager, languages=['ko', 'en'])
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        self.bert_tokenizer = model_manager.get_klue_tokenizer()
        self.bert_model = model_manager.get_klue_bert()
        
        # New smart OCR integration
        self.use_smart_ocr = use_smart_ocr
        if use_smart_ocr:
            try:
                self.ocr_service = get_ocr_service()
                self.ocr_config = ConfigLoader.load_ocr_config()
                config_status = self.ocr_service.get_config_status()
                self.smart_ocr_available = config_status["tesseract_available"] or config_status["easyocr_available"]
                logger.info(f"Smart OCR enabled: {self.smart_ocr_available}")
            except Exception as e:
                logger.warning(f"Smart OCR initialization failed: {e}, falling back to simple mode")
                self.use_smart_ocr = False
                self.smart_ocr_available = False
        
        logger.debug("Enhanced PDFHandler initialized.")

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file with intelligent OCR strategy.
        Combines your existing models with smart processing decisions.
        """
        try:
            doc = await asyncio.to_thread(fitz.open, file_path)
            text_parts = []
            total_text_length = 0

            # Phase 1: Extract all regular PDF text (same as your current approach)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = await asyncio.to_thread(page.get_text, "text")
                if page_text:
                    text_parts.append(page_text)
                    total_text_length += len(page_text.strip())

            # Phase 2: Decide OCR strategy
            if self.use_smart_ocr and self.smart_ocr_available:
                # Use intelligent OCR strategy
                should_process_images, ocr_mode = self._determine_smart_ocr_strategy(
                    total_text_length, len(doc)
                )
                
                if should_process_images:
                    logger.info(
                        f"PDF has {total_text_length} chars, using smart OCR mode: {ocr_mode}"
                    )
                    image_text_parts = await self._process_images_smart(doc, ocr_mode)
                    if image_text_parts:
                        text_parts.extend(image_text_parts)
                else:
                    logger.info(f"PDF has sufficient text ({total_text_length} chars), skipping OCR")
            else:
                # Use your original simple threshold approach
                if total_text_length < 100:
                    logger.info(f"PDF appears to be scanned (only {total_text_length} chars), processing images")
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        image_texts = await self._extract_images_from_page(page, page_num)
                        if image_texts:
                            text_parts.append(f"=== Page {page_num + 1} Images ===\n" + "\n".join(image_texts))
                else:
                    logger.info(f"PDF has sufficient text ({total_text_length} chars), skipping OCR")

            await asyncio.to_thread(doc.close)
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def _determine_smart_ocr_strategy(self, text_length: int, page_count: int) -> tuple[bool, str]:
        """
        Determine if OCR should be performed and what strategy to use.
        
        Returns:
            tuple: (should_process_images, ocr_mode)
        """
        text_density = text_length / page_count if page_count > 0 else 0
        
        # Your original logic as fallback
        if text_length < 100:
            return True, "full_legacy"  # Use your original method
        
        # Enhanced logic for better documents
        if text_length < 500:
            return True, "full_smart"     # Scanned documents
        elif text_length < 3000 or text_density < 500:
            return True, "selective"     # Mixed content
        elif text_length < 8000:
            return True, "minimal"       # Text-rich with potential images
        else:
            return False, "skip"         # Very text-rich documents
    
    async def _process_images_smart(self, doc, ocr_mode: str) -> List[str]:
        """Process images using smart OCR strategy."""
        image_text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            try:
                if ocr_mode == "full_legacy":
                    # Use your original method for backward compatibility
                    image_texts = await self._extract_images_from_page(page, page_num)
                    if image_texts:
                        image_text_parts.append(f"=== Page {page_num + 1} Images ===\n" + "\n".join(image_texts))
                
                elif ocr_mode in ["full_smart", "selective", "minimal"]:
                    # Use smart OCR with your config
                    images = await self.ocr_service.extract_images_from_pdf_page(page, page_num)
                    
                    if images:
                        # Determine context for better OCR
                        context = self._determine_image_context(page, images)
                        
                        # Choose between your models or config-based OCR
                        if context == "handwritten" and self.handwritten_model:
                            # Use your TrOCR for handwritten text
                            page_image_texts = await self._extract_with_trocr(images, page_num)
                        else:
                            # Use config-based OCR (Tesseract/EasyOCR)
                            engine = self._choose_ocr_engine_for_context(context)
                            page_image_texts = await self.ocr_service.extract_text_from_images(
                                images, ocr_mode.replace("_smart", ""), context, engine
                            )
                        
                        if page_image_texts:
                            page_section = f"=== Page {page_num + 1} Images ===\n" + "\n".join(page_image_texts)
                            image_text_parts.append(page_section)
                            logger.debug(f"Smart OCR processed {len(page_image_texts)} images on page {page_num + 1}")
            
            except Exception as e:
                logger.warning(f"Failed to process images on page {page_num + 1}: {e}")
                continue
        
        return image_text_parts
    
    async def _extract_with_trocr(self, images: List[Image.Image], page_num: int) -> List[str]:
        """Extract text using your existing TrOCR model for handwritten text."""
        image_texts = []
        
        for idx, image in enumerate(images):
            try:
                # Convert PIL Image to the format your TrOCR expects
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()
                
                # Use your existing image handler
                text = await self.image_handler.extract_text_from_memory(image_bytes)
                if text:
                    image_texts.append(f"[TrOCR Image {idx + 1}] {text}")
                
            except Exception as e:
                logger.warning(f"TrOCR failed for image {idx + 1} on page {page_num + 1}: {e}")
                continue
        
        return image_texts
    
    def _determine_image_context(self, page, images: List) -> str:
        """Determine the context of images for optimal OCR engine selection."""
        try:
            page_text = page.get_text("text").lower()
            
            # Check for handwritten content indicators
            if any(keyword in page_text for keyword in ['handwritten', '손글씨', '필기', 'note']):
                return "handwritten"
            
            # Check for other contexts
            if any(keyword in page_text for keyword in ['error', 'exception', 'traceback']):
                return "error_message"
            elif any(keyword in page_text for keyword in ['diagram', 'figure', 'chart']):
                return "diagram"
            elif any(keyword in page_text for keyword in ['한글', '한국어']):
                return "korean_text"
            else:
                return "general"
        
        except Exception:
            return "general"
    
    def _choose_ocr_engine_for_context(self, context: str) -> str:
        """Choose OCR engine based on context."""
        engine_preferences = {
            "handwritten": "auto",         # Let TrOCR handle this
            "korean_text": "easyocr",      # EasyOCR better for Korean
            "error_message": "tesseract",   # Tesseract good for clean text
            "diagram": "auto",
            "general": "auto"
        }
        
        return engine_preferences.get(context, "auto")

    # Keep your original methods for backward compatibility
    async def extract_text_from_memory(self, file_content: bytes) -> str:
        """Extract text from PDF content in memory (your original method)."""
        try:
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

            await asyncio.to_thread(doc.close)
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from PDF content in memory: {e}")
            return ""

    async def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[str]:
        """Extract images from a PDF page and perform OCR (your original method)."""
        image_texts = []
        image_list = await asyncio.to_thread(page.get_images, full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = await asyncio.to_thread(page.parent.extract_image, xref)
                image_bytes = base_image["image"]
                
                # Convert image bytes to a format suitable for OCR
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img_np is None:
                    logger.warning(f"Failed to decode image on page {page_num + 1}, image {img_index + 1}")
                    continue

                # Use your existing ImageHandler
                image_text = await self.image_handler.extract_text_from_memory(image_bytes)
                if image_text:
                    image_texts.append(f"Image {img_index + 1}:\n{image_text}")

            except Exception as e:
                logger.warning(f"Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                continue

        return image_texts

    async def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """Extract tables from a PDF file using pdfplumber (your original method)."""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = await asyncio.to_thread(page.extract_tables)
                    for table in page_tables:
                        cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                        tables.append(cleaned_table)
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from PDF {file_path}: {e}")
            return []
    
    def get_handler_status(self) -> dict:
        """Get current handler status for debugging."""
        status = {
            "handler": "Enhanced PDFHandler",
            "smart_ocr_enabled": self.use_smart_ocr,
            "smart_ocr_available": getattr(self, 'smart_ocr_available', False),
            "ml_models_loaded": {
                "trocr": self.handwritten_model is not None,
                "klue_bert": self.bert_model is not None,
                "image_handler": self.image_handler is not None
            }
        }
        
        if self.use_smart_ocr and hasattr(self, 'ocr_service'):
            status["ocr_service_status"] = self.ocr_service.get_config_status()
        
        return status