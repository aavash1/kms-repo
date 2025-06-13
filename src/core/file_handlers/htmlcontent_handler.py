# src/core/file_handlers/htmlcontent_handler.py
"""
Enhanced HTML content handler with smart OCR for embedded images.
"""

import requests
from bs4 import BeautifulSoup, SoupStrainer
import re
import io
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tempfile
import os
import json
import pandas as pd
from urllib.parse import urljoin, urlparse
import logging
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel
import asyncio
import base64
from PIL import Image

from .image_handler import ImageHandler

# Import hybrid OCR system
try:
    from ..services.ocr_service import get_ocr_service
    from ..utils.config_loader import ConfigLoader
    HYBRID_OCR_AVAILABLE = True
except ImportError:
    HYBRID_OCR_AVAILABLE = False

try:
    from ..mariadb_db.mariadb_connector import MariaDBConnector
except ImportError:
    MariaDBConnector = None

logger = logging.getLogger(__name__)

class HTMLContentHandler:
    """Enhanced HTML content handler with smart OCR capabilities."""
    
    def __init__(self, languages=['ko', 'en'], base_url=None, model_manager=None, use_smart_ocr=True):
        """
        Initialize the HTMLContentHandler with smart OCR capabilities.
        
        Args:
            languages: List of language codes for OCR (e.g., ['ko','en'])
            base_url: Optional base URL for resolving relative image paths
            model_manager: Optional model manager for accessing preloaded models
            use_smart_ocr: Whether to use smart OCR strategy
        """
        self.languages = languages
        self.base_url = base_url
        self.image_handler = ImageHandler(model_manager=model_manager, languages=languages)

        # Your existing model setup
        if model_manager:
            self.trocr_processor = model_manager.get_trocr_processor()
            self.trocr_model = model_manager.get_trocr_model()
            self.bert_tokenizer = model_manager.get_klue_tokenizer()
            self.bert_model = model_manager.get_klue_bert()
        else:
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                r"C:\AI_Models\local_cache\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee",
                local_files_only=True,
                use_fast=True
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                r"C:\AI_Models\local_cache\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee",
                local_files_only=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.trocr_model.eval()

            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                r"C:\AI_Models\local_cache\models--klue--bert-base\snapshots\77c8b3d707df785034b4e50f2da5d37be5f0f546",
                local_files_only=True
            )
            self.bert_model = AutoModel.from_pretrained(
                r"C:\AI_Models\local_cache\models--klue--bert-base\snapshots\77c8b3d707df785034b4e50f2da5d37be5f0f546",
                local_files_only=True
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.eval()

        # Hybrid OCR integration
        self.use_smart_ocr = use_smart_ocr and HYBRID_OCR_AVAILABLE
        if self.use_smart_ocr:
            try:
                self.ocr_service = get_ocr_service()
                self.ocr_config = ConfigLoader.load_ocr_config()
                config_status = self.ocr_service.get_config_status()
                self.smart_ocr_available = config_status["tesseract_available"] or config_status["easyocr_available"]
                logger.info(f"HTML Handler smart OCR enabled: {self.smart_ocr_available}")
            except Exception as e:
                logger.warning(f"Smart OCR initialization failed: {e}, falling back to legacy mode")
                self.use_smart_ocr = False
                self.smart_ocr_available = False

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content."""
        logger.debug(f"Raw HTML content for extraction: {html_content[:500]}...")
        if not html_content or not html_content.strip():
            logger.warning("HTML content is empty or whitespace.")
            return ""

        # Parse HTML with a broader set of tags
        parse_only = SoupStrainer(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'li', 'td', 'th', 'a', 'b', 'i', 'strong', 'em', 'pre', 'code'])
        soup = BeautifulSoup(html_content, 'html.parser', parse_only=parse_only)
        
        lines = []
        for elem in soup.find_all(True):
            if elem.name in ['script', 'style']:
                continue
            if elem.string:
                text = elem.string.strip()
                if text:
                    lines.append(text)
            elif elem.strings:
                text = ' '.join(s.strip() for s in elem.strings if s.strip())
                if text:
                    lines.append(text)
            else:
                # Fallback for nested tags
                text = elem.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)

        if not lines:
            # Fallback: Extract all text if no specific tags matched
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)
                logger.debug("Fallback extraction used: extracted text from entire HTML.")

        combined_text = '\n'.join(line for line in lines if line)
        if not combined_text:
            logger.warning("No text extracted from HTML after parsing.")
        return combined_text

    def _extract_images_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract image URLs from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                images.append({"src": src, "alt": img.get('alt', ""), "bytes": None, "text": ""})
        return images

    def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None

    async def _process_image_smart(self, image_info: Dict[str, Any], ocr_mode: str, context: str) -> Optional[str]:
        """Process an image with smart OCR strategy."""
        try:
            # Handle base64 images
            if image_info['src'].startswith('data:image'):
                image_data = image_info['src'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                # Download external image
                image_bytes = await asyncio.to_thread(self._download_image, image_info['src'])
                if not image_bytes:
                    return None
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Use smart OCR service
            if context == "handwritten" and self.trocr_model:
                # Use your TrOCR for handwritten content
                return await self._extract_with_trocr_html(pil_image)
            else:
                # Use smart OCR service
                engine = self._choose_ocr_engine_for_context(context)
                extracted_texts = await self.ocr_service.extract_text_from_images(
                    [pil_image], ocr_mode, context, engine
                )
                return extracted_texts[0] if extracted_texts else None
                
        except Exception as e:
            logger.error(f"Error processing image {image_info['src']} with smart OCR: {e}")
            return None

    async def _extract_with_trocr_html(self, image: 'Image.Image') -> str:
        """Extract text using your TrOCR model for handwritten content."""
        try:
            with torch.no_grad():
                inputs = self.trocr_processor(images=image, return_tensors="pt").to(self.trocr_model.device)
                generated_ids = self.trocr_model.generate(inputs.pixel_values)
                extracted_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return extracted_text.strip()
        except Exception as e:
            logger.error(f"TrOCR processing failed: {e}")
            return ""

    def _determine_image_context_html(self, html_content: str, images: List[Dict[str, Any]]) -> str:
        """Determine the context of images in HTML for optimal OCR."""
        try:
            html_text = html_content.lower()
            
            # Check for specific contexts
            if any(keyword in html_text for keyword in ['handwritten', '손글씨', 'signature', 'note']):
                return "handwritten"
            elif any(keyword in html_text for keyword in ['error', 'exception', 'traceback', '오류']):
                return "error_message"
            elif any(keyword in html_text for keyword in ['screenshot', '스크린샷', 'capture']):
                return "screenshot"
            elif any(keyword in html_text for keyword in ['diagram', 'chart', '도표', 'graph']):
                return "diagram"
            elif any(keyword in html_text for keyword in ['한글', '한국어', 'korean']):
                return "korean_text"
            else:
                return "general"
                
        except Exception:
            return "general"

    def _determine_smart_ocr_strategy(self, text_length: int, image_count: int) -> tuple[bool, str]:
        """Determine if smart OCR should be used for HTML content."""
        if not self.use_smart_ocr or not self.smart_ocr_available:
            return True, "full_legacy"  # Use original method
            
        # HTML content strategy
        if text_length < 200:
            return True, "full"      # Image-heavy content
        elif text_length < 1000 or image_count > 3:
            return True, "selective" # Mixed content
        elif text_length < 3000:
            return True, "minimal"   # Text-rich with some images
        else:
            return False, "skip"     # Very text-rich content

    def _choose_ocr_engine_for_context(self, context: str) -> str:
        """Choose OCR engine based on HTML image context."""
        engine_preferences = {
            "handwritten": "auto",         # Let TrOCR handle this
            "korean_text": "easyocr",      # EasyOCR better for Korean
            "error_message": "tesseract",   # Tesseract good for clean text
            "screenshot": "tesseract",      # Screenshots usually have clean text
            "diagram": "tesseract",         # Diagrams usually have clean text
            "general": "auto"
        }
        
        return engine_preferences.get(context, "auto")

    async def process_html(
        self,
        html_content: str,
        download_images: bool = True,
        extract_image_text: bool = True,
        ocr_engine: str = "tesseract"
    ) -> Dict[str, Any]:
        """Process HTML content with smart OCR strategy."""
        result = {"html_text": "", "images": []}

        # Extract text from HTML
        html_text = self._extract_text_from_html(html_content)
        result["html_text"] = html_text

        # Extract images if requested
        if download_images:
            images = self._extract_images_from_html(html_content)
            result["images"] = images

            if extract_image_text and images:
                # Determine OCR strategy
                should_process_images, ocr_mode = self._determine_smart_ocr_strategy(
                    len(html_text), len(images)
                )
                
                if should_process_images:
                    if self.use_smart_ocr and self.smart_ocr_available and ocr_mode != "full_legacy":
                        # Use smart OCR
                        logger.info(f"HTML has {len(html_text)} chars and {len(images)} images, using smart OCR mode: {ocr_mode}")
                        context = self._determine_image_context_html(html_content, images)
                        
                        for image in images:
                            image_text = await self._process_image_smart(image, ocr_mode, context)
                            image["text"] = image_text if image_text else ""
                    else:
                        # Use original method
                        for image in images:
                            if "bytes" in image and image["bytes"]:
                                image_text = await self.image_handler.extract_text_from_memory(
                                    image["bytes"],
                                    ocr_engine=ocr_engine
                                )
                                image["text"] = image_text if image_text else ""

        return result

    async def process_html_file(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Process HTML file with smart OCR."""
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return await self.process_html(html_content, **kwargs)

    async def process_html_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Process HTML from URL with smart OCR."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if self.base_url is None:
                self.base_url = url
            return await self.process_html(response.text, **kwargs)
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {'html_text': '', 'images': []}

    async def process_html_from_db(self, html_content: str, base_url: str = None, **kwargs) -> Dict[str, Any]:
        """Process HTML from database with smart OCR."""
        if base_url:
            self.base_url = base_url
        return await self.process_html(html_content, **kwargs)

    def get_combined_text(self, result: Dict[str, Any], include_alt_text: bool = True) -> str:
        """Get combined text from HTML and images."""
        combined_text = [result['html_text']]
        if 'images' in result and result['images']:
            combined_text.append("\n\n--- TEXT FROM IMAGES ---\n")
            for i, img in enumerate(result['images'], 1):
                combined_text.append(f"\nImage {i}:")
                if include_alt_text and img['alt']:
                    combined_text.append(f"Alt text: {img['alt']}")
                if img['text']:
                    combined_text.append(f"OCR text:\n{img['text']}")
                else:
                    combined_text.append("No text extracted from image.")
        return '\n'.join(combined_text)

    def save_result_as_json(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save result as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_combined_text(self, result: Dict[str, Any], output_path: Union[str, Path], 
                          include_alt_text: bool = True) -> None:
        """Save combined text to file."""
        combined_text = self.get_combined_text(result, include_alt_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

    async def extract_text(self, html_content: str, download_images: bool = True, 
                          ocr_engine: str = "tesseract") -> str:
        """Extract text from HTML with smart OCR."""
        result = await self.process_html(
            html_content,
            download_images=download_images,
            extract_image_text=download_images,
            ocr_engine=ocr_engine
        )
        return self.get_combined_text(result)

    async def process_from_db(self, db_connector=None, error_code_id=None, row=None, include_images=True):
        """Process HTML from database with smart OCR."""
        if row is not None:
            html_content = row.get('content', '')
            if include_images:
                base_url = None
                if 'url' in row and row['url']:
                    parsed_url = urlparse(row['url'])
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                self.base_url = base_url
            result = await self.process_html(html_content, download_images=include_images)
            result['metadata'] = {k: v for k, v in row.items() if k != 'content'}
            return result
        elif db_connector is not None and error_code_id is not None:
            if not hasattr(db_connector, 'get_files_by_error_code'):
                raise ValueError("db_connector must be a valid MariaDBConnector instance")
            all_results = []
            try:
                reports_df = db_connector.get_files_by_error_code(error_code_id)
                for _, row in reports_df.iterrows():
                    try:
                        result = await self.process_from_db(row=row, include_images=include_images)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing report: {e}")
                return all_results
            except Exception as e:
                logger.error(f"Error fetching reports from database: {e}")
                return []
        else:
            raise ValueError("Must provide either a row to process or both db_connector and error_code_id")

    def get_handler_status(self) -> dict:
        """Get current handler status for debugging."""
        status = {
            "handler": "Enhanced HTML Content Handler",
            "smart_ocr_enabled": self.use_smart_ocr,
            "smart_ocr_available": getattr(self, 'smart_ocr_available', False),
            "languages": self.languages,
            "ml_models_loaded": {
                "trocr": self.trocr_model is not None,
                "bert": self.bert_model is not None,
            }
        }
        
        if self.use_smart_ocr and hasattr(self, 'ocr_service'):
            status["ocr_service_status"] = self.ocr_service.get_config_status()
        
        return status

# Helper function to run async methods synchronously
def run_async(coro):
    """Helper function to run async methods synchronously."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return asyncio.create_task(coro)
    return loop.run_until_complete(coro)