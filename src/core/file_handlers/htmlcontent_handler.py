# src/core/file_handlers/htmlcontent_handler.py
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
import asyncio  # Added for async support

from .image_handler import ImageHandler
try:
    from ..mariadb_db.mariadb_connector import MariaDBConnector
except ImportError:
    MariaDBConnector = None

logger = logging.getLogger(__name__)

class HTMLContentHandler:
    def __init__(self, languages=['ko', 'en'], base_url=None, model_manager=None):
        """
        Initialize the HTMLContentHandler with specified languages for OCR.
        
        Args:
            languages: List of language codes for OCR (e.g., ['ko','en'])
            base_url: Optional base URL for resolving relative image paths
            model_manager: Optional model manager for accessing preloaded models
        """
        self.languages = languages
        self.base_url = base_url
        self.image_handler = ImageHandler(model_manager=model_manager, languages=languages)

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

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract text from HTML content.
        
        Args:
            html_content: Raw HTML content as a string.
            
        Returns:
            Extracted text as a string.
        """
        logger.debug(f"Raw HTML content for extraction: {html_content[:500]}...")  # Log the first 500 characters
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
        """
        Extract image URLs from HTML content.
        
        Args:
            html_content: Raw HTML content as a string.
            
        Returns:
            List of dictionaries containing image information.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                images.append({"src": src, "alt": img.get('alt', ""), "bytes": None, "text": ""})
        return images

    def _download_image(self, url: str) -> Optional[bytes]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None

    async def _process_image(self, image_info: Dict[str, Any], ocr_engine: str = "tesseract") -> Optional[str]:
        """
        Process an image asynchronously, downloading and extracting text.
        
        Args:
            image_info: Dictionary with image metadata (src, alt, etc.)
            ocr_engine: OCR engine to use for text extraction
            
        Returns:
            Optional[str]: Extracted text from the image or None if failed
        """
        image_content = await asyncio.to_thread(self._download_image, image_info['src'])
        if image_content:
            try:
                # Assuming ImageHandler.extract_text_from_memory supports ocr_engine
                return self.image_handler.extract_text_from_memory(image_content, ocr_engine=ocr_engine)
            except Exception as e:
                logger.error(f"Error extracting text from image {image_info['src']}: {e}")
        return None

    async def process_html(
        self,
        html_content: str,
        download_images: bool = True,
        extract_image_text: bool = True,
        ocr_engine: str = "tesseract"
    ) -> Dict[str, Any]:
        """
        Process HTML content, extract text and optionally images.
        
        Args:
            html_content: Raw HTML content as a string.
            download_images: Whether to download images from <img> tags.
            extract_image_text: Whether to extract text from images using OCR.
            ocr_engine: OCR engine to use for image text extraction.
            
        Returns:
            Dict with extracted text and image information.
        """
        result = {"html_text": "", "images": []}

        # Extract text from HTML
        html_text = self._extract_text_from_html(html_content)
        result["html_text"] = html_text

        # Extract images if requested
        if download_images:
            images = self._extract_images_from_html(html_content)
            result["images"] = images

            if extract_image_text:
                for image in images:
                    if "bytes" in image:
                        image_text = await self.image_handler.extract_text_from_memory(
                            image["bytes"],
                            ocr_engine=ocr_engine
                        )
                        image["text"] = image_text if image_text else ""

        return result

    async def process_html_file(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return await self.process_html(html_content, **kwargs)

    async def process_html_url(self, url: str, **kwargs) -> Dict[str, Any]:
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
        if base_url:
            self.base_url = base_url
        return await self.process_html(html_content, **kwargs)

    def get_combined_text(self, result: Dict[str, Any], include_alt_text: bool = True) -> str:
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_combined_text(self, result: Dict[str, Any], output_path: Union[str, Path], 
                          include_alt_text: bool = True) -> None:
        combined_text = self.get_combined_text(result, include_alt_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

    async def extract_text(self, html_content: str, download_images: bool = True, 
                          ocr_engine: str = "tesseract") -> str:
        result = await self.process_html(
            html_content,
            download_images=download_images,
            extract_image_text=download_images,
            ocr_engine=ocr_engine
        )
        return self.get_combined_text(result)

    async def process_from_db(self, db_connector=None, error_code_id=None, row=None, include_images=True):
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

# Helper function to run async methods synchronously
def run_async(coro):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return asyncio.create_task(coro)
    return loop.run_until_complete(coro)