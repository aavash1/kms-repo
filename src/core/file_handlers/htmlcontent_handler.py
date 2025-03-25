# htmlcontent_handler.py
import requests
from bs4 import BeautifulSoup
import re
import io
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import tempfile
import os
import json
import pandas as pd
from urllib.parse import urljoin, urlparse
import logging

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel

# Import the ImageHandler for image text extraction
from .image_handler import ImageHandler
# Import the MariaDBConnector for database access (with conditional import in case it's not available)
try:
    from ..mariadb_db.mariadb_connector import MariaDBConnector
except ImportError:
    # Log this or handle differently if needed
    MariaDBConnector = None

class HTMLContentHandler:
    def __init__(self, languages=['ko', 'en'], base_url=None, model_manager=None):
        """
        Initialize the HTMLContentHandler with specified languages for OCR.
        
        Args:
            languages: list of language codes for OCR (e.g., ['ko','en'] for Korean and English)
            base_url: Optional base URL for resolving relative image paths
        """
        self.languages = languages
        self.base_url = base_url
        # Initialize the ImageHandler for image processing
        self.image_handler = ImageHandler(model_manager=model_manager, languages=languages)

        if model_manager:
            self.trocr_processor = model_manager.get_trocr_processor()
            self.trocr_model = model_manager.get_trocr_model()
            self.bert_tokenizer = model_manager.get_klue_tokenizer()
            self.bert_model = model_manager.get_klue_bert()
        else:
            # Fallback to loading models directly
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
        Extract plain text from HTML content.
        
        Args:
            html_content: The HTML content as a string
            
        Returns:
            str: Extracted text with preserved structure
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()
            
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_images_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract image information from HTML content.
        
        Args:
            html_content: The HTML content as a string
            
        Returns:
            List[Dict]: List of dictionaries with image info
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        image_tags = soup.find_all('img')
        
        images = []
        for img in image_tags:
            # Extract src attribute
            src = img.get('src', '')
            if not src:
                continue
                
            # Handle relative URLs if base_url is provided
            if self.base_url and not bool(urlparse(src).netloc):
                src = urljoin(self.base_url, src)
                
            # Extract alt text and other attributes
            alt = img.get('alt', '')
            width = img.get('width', '')
            height = img.get('height', '')
            
            image_info = {
                'src': src,
                'alt': alt,
                'width': width,
                'height': height,
                'original_tag': str(img)
            }
            
            images.append(image_info)
            
        return images
    
    def _download_image(self, url: str) -> Optional[bytes]:
        """
        Download image content from URL.
        
        Args:
            url: URL of the image
            
        Returns:
            bytes: Image content as bytes, or None if download fails
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.content
        except Exception as e:
            logging.error(f"Error downloading image from {url}: {e}")
            return None
    
    def process_html(self, html_content: str, download_images: bool = True, 
                    extract_image_text: bool = True, ocr_engine: str = "tesseract") -> Dict[str, Any]:
        """
        Process HTML content to extract text and image information.
        
        Args:
            html_content: The HTML content as a string
            download_images: Whether to download and process images
            extract_image_text: Whether to extract text from images using OCR
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', or 'auto')
            
        Returns:
            Dict: Dictionary with extracted text and image information
        """
        result = {
            'html_text': self._extract_text_from_html(html_content),
            'images': []
        }
        
        if download_images:
            images = self._extract_images_from_html(html_content)
            for image_info in images:
                image_data = {
                    'src': image_info['src'],
                    'alt': image_info['alt'],
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'text': None
                }
                
                if extract_image_text:
                    image_content = self._download_image(image_info['src'])
                    if image_content:
                        # Extract text from image using ImageHandler
                        try:
                            image_text = self.image_handler.extract_text_from_memory(image_content)
                            image_data['text'] = image_text
                        except Exception as e:
                            logging.error(f"Error extracting text from image: {e}")
                
                result['images'].append(image_data)
        
        return result
    
    def process_html_file(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process HTML from a file.
        
        Args:
            file_path: Path to the HTML file
            **kwargs: Additional arguments to pass to process_html
            
        Returns:
            Dict: Dictionary with extracted text and image information
        """
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return self.process_html(html_content, **kwargs)
    
    def process_html_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Process HTML from a URL.
        
        Args:
            url: URL to fetch and process
            **kwargs: Additional arguments to pass to process_html
            
        Returns:
            Dict: Dictionary with extracted text and image information
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Set base_url for relative image URLs if not already set
            if self.base_url is None:
                self.base_url = url
                
            return self.process_html(response.text, **kwargs)
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            return {'html_text': '', 'images': []}
    
    def process_html_from_db(self, html_content: str, base_url: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process HTML content retrieved from a database.
        
        Args:
            html_content: HTML content string from database
            base_url: Base URL for resolving relative image paths
            **kwargs: Additional arguments to pass to process_html
            
        Returns:
            Dict: Dictionary with extracted text and image information
        """
        # Set base_url if provided
        if base_url:
            self.base_url = base_url
            
        return self.process_html(html_content, **kwargs)
    
    def get_combined_text(self, result: Dict[str, Any], include_alt_text: bool = True) -> str:
        """
        Get combined text from HTML and images.
        
        Args:
            result: The result dictionary from process_html
            include_alt_text: Whether to include image alt text
            
        Returns:
            str: Combined text from HTML and images
        """
        combined_text = [result['html_text']]
        
        if 'images' in result and result['images']:
            combined_text.append("\n\n--- TEXT FROM IMAGES ---\n")
            
            for i, img in enumerate(result['images'], 1):
                combined_text.append(f"\nImage {i}:")
                
                # Add alt text if available and requested
                if include_alt_text and img['alt']:
                    combined_text.append(f"Alt text: {img['alt']}")
                
                # Add OCR text if available
                if img['text']:
                    combined_text.append(f"OCR text:\n{img['text']}")
                else:
                    combined_text.append("No text extracted from image.")
        
        return "\n".join(combined_text)
    
    def save_result_as_json(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save the processing result as a JSON file.
        
        Args:
            result: The result dictionary from process_html
            output_path: Path to save the JSON output
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def save_combined_text(self, result: Dict[str, Any], output_path: Union[str, Path], 
                          include_alt_text: bool = True) -> None:
        """
        Save the combined text as a text file.
        
        Args:
            result: The result dictionary from process_html
            output_path: Path to save the text output
            include_alt_text: Whether to include image alt text
        """
        combined_text = self.get_combined_text(result, include_alt_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

    def extract_text(self, html_content: str, download_images: bool = True, 
                    ocr_engine: str = "tesseract") -> str:
        """
        Simplified method to extract all text from HTML content including images.
        
        Args:
            html_content: The HTML content as a string
            download_images: Whether to download and process images
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', or 'auto')
            
        Returns:
            str: Combined text from HTML and images
        """
        result = self.process_html(
            html_content, 
            download_images=download_images, 
            extract_image_text=download_images,
            ocr_engine=ocr_engine
        )
        
        return self.get_combined_text(result)
    
    def process_from_db(self, db_connector=None, error_code_id=None, row=None, include_images=True):
        """
        Process HTML content directly from database using MariaDBConnector.
        
        This method can work in two ways:
        1. Pass a db_connector and error_code_id to fetch and process all matching reports
        2. Pass a single row from a DataFrame to process just that report
        
        Args:
            db_connector: MariaDBConnector instance (optional)
            error_code_id: Error code ID to fetch reports for (optional)
            row: Single DataFrame row with content and image URLs (optional)
            include_images: Whether to download and process images
            
        Returns:
            Dict or List[Dict]: Results of HTML processing
        """
        # If we have a specific row to process
        if row is not None:
            html_content = row.get('content', '')
            
            # Set up base URL for image processing if needed
            if include_images:
                # If we have a specific base URL in the row, use it
                # Otherwise, try to extract domain from any image URL
                base_url = None
                if 'url' in row and row['url']:
                    parsed_url = urlparse(row['url'])
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    
                self.base_url = base_url
            
            # Process the HTML content
            result = self.process_html(html_content, download_images=include_images)
            
            # Add metadata from the row
            result['metadata'] = {k: v for k, v in row.items() if k != 'content'}
            
            return result
            
        # If we need to fetch from database
        elif db_connector is not None and error_code_id is not None:
            # Ensure we have a valid MariaDBConnector
            if not hasattr(db_connector, 'get_unprocessed_troubleshooting_reports'):
                raise ValueError("db_connector must be a valid MariaDBConnector instance")
            
            # Initialize results list
            all_results = []
            
            try:
                # Fetch troubleshooting reports
                reports_df = db_connector.get_files_by_error_code(error_code_id)
                
                # Process each report
                for _, row in reports_df.iterrows():
                    try:
                        result = self.process_from_db(row=row, include_images=include_images)
                        all_results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing report: {e}")
                        # Continue with next report
                
                return all_results
                
            except Exception as e:
                logging.error(f"Error fetching reports from database: {e}")
                return []
                
        else:
            raise ValueError("Must provide either a row to process or both db_connector and error_code_id")