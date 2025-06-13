# File: src/core/services/ocr_service.py
"""
Enhanced OCR Service with EasyOCR-only configuration support.
Supports: PDF, Word, PowerPoint, Excel, Images, HTML content, etc.
"""

import asyncio
import logging
import yaml
import os
from typing import List, Tuple, Optional, Dict, Any
import io
import tempfile
from pathlib import Path

# Core libraries
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Optional EasyOCR support
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# File format specific libraries
import fitz  # PyMuPDF for PDFs
from docx import Document
from pptx import Presentation
import openpyxl
from bs4 import BeautifulSoup
import base64

logger = logging.getLogger(__name__)

class OCRService:
    """
    Centralized OCR service with EasyOCR-only configuration support.
    Supports multiple OCR engines and intelligent processing strategies.
    """
    
    def __init__(self, config_path: str = "src/configs/ocr_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._setup_ocr_engines()
        self.logger.info("OCRService initialized with EasyOCR-only configuration")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load OCR configuration from YAML file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                self.logger.info(f"Loaded OCR configuration from {config_path}")
                return config
            else:
                self.logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading OCR config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration fallback."""
        return {
            'tesseract': {
                'path': None,  # Use system PATH
                'lang': 'eng+kor',
                'psm': 4,
                'oem': 3
            },
            'easyocr': {
                'langs': ['en', 'ko'],
                'gpu': True
            },
            'poppler': {
                'path': None
            },
            'libreoffice': {
                'path': None
            }
        }
    
    def _setup_ocr_engines(self):
        """Setup OCR engines with EasyOCR preference."""
        
        # Setup Tesseract (check if enabled in config)
        tesseract_config = self.config.get('tesseract', {})
        tesseract_enabled = tesseract_config.get('enabled', True)
        
        if tesseract_enabled and tesseract_config.get('path'):
            pytesseract.pytesseract.tesseract_cmd = tesseract_config['path']
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                self.logger.info("Tesseract OCR engine available")
            except Exception as e:
                self.tesseract_available = False
                self.logger.warning(f"Tesseract not available: {e}")
        else:
            self.tesseract_available = False
            self.logger.info("Tesseract disabled - using EasyOCR only")
        
        # Setup EasyOCR (primary engine)
        self.easyocr_reader = None
        easyocr_config = self.config.get('easyocr', {})
        easyocr_enabled = easyocr_config.get('enabled', True)
        
        if EASYOCR_AVAILABLE and easyocr_enabled:
            try:
                self.easyocr_reader = easyocr.Reader(
                    easyocr_config.get('langs', ['en', 'ko']),
                    gpu=easyocr_config.get('gpu', True)
                )
                self.logger.info("EasyOCR engine initialized with GPU support")
            except Exception as e:
                self.logger.error(f"EasyOCR initialization failed: {e}")
                # Don't raise exception, fall back to Tesseract if available
                if not self.tesseract_available:
                    raise Exception("No OCR engines available")
        else:
            self.logger.warning("EasyOCR not available, falling back to Tesseract if enabled")
            if not self.tesseract_available:
                raise Exception("No OCR engines available")
        
        # Build OCR configurations
        self._build_ocr_configs()
    
    def _build_ocr_configs(self):
        """Build OCR configurations based on loaded config."""
        tesseract_config = self.config.get('tesseract', {})
        base_lang = tesseract_config.get('lang', 'eng+kor')
        base_psm = tesseract_config.get('psm', 4)
        base_oem = tesseract_config.get('oem', 3)
        
        self.ocr_configs = {
            'general': f'--psm {base_psm} --oem {base_oem} -l {base_lang}',
            'single_column': f'--psm 4 --oem {base_oem} -l {base_lang}',
            'single_word': f'--psm 8 --oem {base_oem} -l {base_lang}',
            'digits_only': f'--psm {base_psm} --oem {base_oem} -c tessedit_char_whitelist=0123456789',
            'alphanumeric': f'--psm {base_psm} --oem {base_oem} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
            'high_quality': f'--psm {base_psm} --oem {base_oem} -l {base_lang}',
            'korean_focused': f'--psm {base_psm} --oem {base_oem} -l kor+eng',
            'english_focused': f'--psm {base_psm} --oem {base_oem} -l eng'
        }
        
        self.logger.info(f"OCR configurations built with language: {base_lang}")
    
    async def extract_text_from_images(self, 
                                     images: List[Image.Image], 
                                     extraction_mode: str = "selective",
                                     image_context: str = "general",
                                     ocr_engine: str = "auto") -> List[str]:
        """Extract text using EasyOCR (or configured engine)."""
        
        if extraction_mode == "skip" or not images:
            return []
        
        extracted_texts = []
        
        for idx, image in enumerate(images):
            try:
                # Filter by size based on extraction mode
                if not self._should_process_image(image, extraction_mode):
                    continue
                
                # Enhance image for OCR
                enhanced_image = await self._enhance_image_for_ocr(image, image_context)
                
                # Choose OCR engine - prefer EasyOCR if available
                chosen_engine = self._choose_ocr_engine(ocr_engine, image_context)
                
                # Perform OCR with chosen engine
                if chosen_engine == "easyocr" and self.easyocr_reader:
                    text = await self._extract_with_easyocr(enhanced_image)
                else:
                    # Fallback to Tesseract if available
                    text = await self._extract_with_tesseract(enhanced_image, image_context)
                
                # Clean and validate text
                cleaned_text = self._clean_ocr_text(text)
                if cleaned_text and len(cleaned_text.strip()) > 5:
                    extracted_texts.append(f"[Image {idx + 1}] {cleaned_text}")
                    self.logger.debug(f"OCR extracted {len(cleaned_text)} chars from image {idx + 1} using {chosen_engine}")
                
            except Exception as e:
                self.logger.warning(f"Failed to extract text from image {idx + 1}: {e}")
                continue
        
        return extracted_texts
    
    async def _extract_with_tesseract(self, image: Image.Image, context: str) -> str:
        """Extract text using Tesseract OCR."""
        if not self.tesseract_available:
            raise Exception("Tesseract OCR not available")
        
        config = self._get_tesseract_config(context)
        return await asyncio.to_thread(pytesseract.image_to_string, image, config=config)
    
    async def _extract_with_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR."""
        if not self.easyocr_reader:
            raise Exception("EasyOCR not available")
        
        # Convert PIL image to numpy array for EasyOCR
        img_array = np.array(image)
        
        # Run EasyOCR
        results = await asyncio.to_thread(self.easyocr_reader.readtext, img_array)
        
        # Extract text from results
        text_parts = [result[1] for result in results if result[2] > 0.5]  # Confidence > 0.5
        return " ".join(text_parts)
    
    def _choose_ocr_engine(self, preferred_engine: str, context: str) -> str:
        """Choose OCR engine - prefer EasyOCR if available."""
        
        # Check configuration preference
        ocr_prefs = self.config.get('ocr_preferences', {})
        default_engine = ocr_prefs.get('default_engine', 'auto')
        
        # Force EasyOCR if configured
        if default_engine == 'easyocr':
            if self.easyocr_reader:
                return "easyocr"
            else:
                self.logger.warning("EasyOCR requested but not available, falling back")
        
        # Handle specific engine requests
        if preferred_engine == "easyocr" and self.easyocr_reader:
            return "easyocr"
        elif preferred_engine == "tesseract" and self.tesseract_available:
            return "tesseract"
        elif preferred_engine == "auto":
            # Auto-choose: prefer EasyOCR, fallback to Tesseract
            if self.easyocr_reader:
                return "easyocr"
            elif self.tesseract_available:
                return "tesseract"
        
        # Final determination
        if self.easyocr_reader:
            return "easyocr"
        elif self.tesseract_available:
            return "tesseract"
        else:
            raise Exception("No OCR engine available")
    
    def _get_tesseract_config(self, context: str) -> str:
        """Get Tesseract configuration based on context."""
        context_mapping = {
            "screenshot": "general",
            "scanned_document": "high_quality",
            "diagram": "single_column",
            "error_message": "general",
            "code": "alphanumeric",
            "table": "single_column",
            "korean_text": "korean_focused",
            "english_text": "english_focused",
            "mixed_language": "general"
        }
        
        config_key = context_mapping.get(context, "general")
        return self.ocr_configs.get(config_key, self.ocr_configs["general"])
    
    async def extract_images_from_pdf_page(self, page, page_num: int) -> List[Image.Image]:
        """Extract images from a PDF page."""
        images = []
        try:
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    else:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                    
                    images.append(pil_image)
                    pix = None
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {img_index} from PDF page {page_num}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error extracting images from PDF page {page_num}: {e}")
        
        return images
    
    def determine_ocr_strategy(self, text_length: int, page_count: int = 1, file_type: str = "unknown") -> str:
        """Determine OCR strategy based on document characteristics."""
        text_density = text_length / page_count if page_count > 0 else 0
        
        # File type specific logic
        if file_type in ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]:
            return "full"  # Pure image files always get full OCR
        
        if file_type in ["pptx", "ppt"]:
            return "selective" if text_length > 1000 else "full"
        
        if file_type in ["xlsx", "xls"]:
            return "minimal" if text_length > 2000 else "selective"
        
        # General strategy for other file types
        if text_length < 500:
            return "full"
        elif text_length < 3000 or text_density < 500:
            return "selective"
        elif text_length < 8000:
            return "minimal"
        else:
            return "skip"
    
    def _should_process_image(self, image: Image.Image, extraction_mode: str) -> bool:
        """Determine if an image should be processed based on extraction mode."""
        width, height = image.size
        
        if extraction_mode == "full":
            return width >= 50 and height >= 50
        elif extraction_mode == "selective":
            return width >= 150 and height >= 100
        elif extraction_mode == "minimal":
            return width >= 250 and height >= 150
        else:
            return False
    
    async def _enhance_image_for_ocr(self, image: Image.Image, context: str = "general") -> Image.Image:
        """Enhance image quality for better OCR results."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Context-specific enhancements
            if context == "screenshot":
                enhanced = image
            elif context == "scanned_document":
                enhanced = image.filter(ImageFilter.MedianFilter(size=3))
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.2)
            elif context == "korean_text":
                # Korean text often needs different enhancement
                enhanced = ImageEnhance.Contrast(image).enhance(1.2)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)
            else:
                enhanced = ImageEnhance.Contrast(image).enhance(1.1)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.1)
            
            # Convert to numpy for OpenCV processing
            img_array = np.array(enhanced)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            final_image = Image.fromarray(img_array)
            
            # Resize if too small
            width, height = final_image.size
            if width < 300 or height < 300:
                scale_factor = max(300 / width, 300 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                final_image = final_image.resize((new_width, new_height), Image.LANCZOS)
            
            return final_image
        
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _clean_ocr_text(self, raw_text: str) -> str:
        """Clean and normalize OCR-extracted text."""
        if not raw_text:
            return ""
        
        # Remove excessive whitespace
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Remove single characters that are likely OCR errors
        words = cleaned.split()
        meaningful_words = [word for word in words if len(word) > 1 or word.isdigit() or word in ['a', 'I']]
        
        result = ' '.join(meaningful_words)
        
        # Filter out obvious OCR garbage
        if len([c for c in result if c.isalnum()]) < len(result) * 0.3:
            return ""
        
        return result.strip()
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration status with EasyOCR focus."""
        ocr_prefs = self.config.get('ocr_preferences', {})
        
        return {
            "mode": "EasyOCR-Preferred" if self.easyocr_reader else "Tesseract-Only",
            "tesseract_available": self.tesseract_available,
            "tesseract_enabled": self.config.get('tesseract', {}).get('enabled', True),
            "easyocr_available": self.easyocr_reader is not None,
            "easyocr_enabled": self.config.get('easyocr', {}).get('enabled', True),
            "easyocr_languages": self.config.get('easyocr', {}).get('langs'),
            "gpu_acceleration": self.config.get('easyocr', {}).get('gpu', False),
            "default_engine": ocr_prefs.get('default_engine', 'auto')
        }

# Singleton instance
_ocr_service = None

def get_ocr_service(config_path: str = "src/configs/ocr_config.yaml") -> OCRService:
    """Get singleton OCR service instance with configuration."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(config_path)
    return _ocr_service

# Additional convenience function for EasyOCR-only mode
def get_easyocr_only_service(config_path: str = "src/configs/ocr_config.yaml") -> OCRService:
    """Get OCR service configured for EasyOCR-only operation."""
    return OCRService(config_path)