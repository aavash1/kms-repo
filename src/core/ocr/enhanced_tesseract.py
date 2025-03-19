# src/core/ocr/enhanced_tesseract.py

import logging
import numpy as np
from PIL import Image
import re

# Import your existing TesseractOCR wrapper
from .tesseract_wrapper import TesseractOCR
from .image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

class EnhancedTesseractOCR:
    """
    Enhanced Tesseract OCR wrapper that applies advanced image preprocessing 
    before using the existing TesseractOCR class for text extraction.
    """
    
    def __init__(self):
        """Initialize with the existing TesseractOCR wrapper"""
        self.tesseract = TesseractOCR()
        self.preprocessor = ImagePreprocessor()
    
    def extract_text(self, image, preprocess_method="adaptive", config=''):
        """
        Extract text from an image with enhanced preprocessing.
        
        Args:
            image: PIL Image, numpy array, or file path
            preprocess_method: Image preprocessing method to use
            config: Additional Tesseract configuration
            
        Returns:
            str: Extracted text
        """
        try:
            # Handle file path
            if isinstance(image, str):
                try:
                    image = Image.open(image)
                except Exception as e:
                    logger.error(f"Failed to open image from path: {e}")
                    return ""
            
            # Apply preprocessing
            preprocessed_image = self.preprocessor.preprocess_for_ocr(image, preprocess_method)
            
            # Extract text using existing TesseractOCR
            text = self.tesseract.extract_text(preprocessed_image, config)
            
            # Clean up the extracted text
            cleaned_text = self._clean_ocr_text(text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {e}")
            return ""
    
    def get_ocr_data(self, image, preprocess_method="adaptive", config=''):
        """
        Get detailed OCR data with enhanced preprocessing.
        
        Args:
            image: PIL Image, numpy array, or file path
            preprocess_method: Image preprocessing method to use
            config: Additional Tesseract configuration
            
        Returns:
            dict: Tesseract OCR data
        """
        try:
            # Handle file path
            if isinstance(image, str):
                try:
                    image = Image.open(image)
                except Exception as e:
                    logger.error(f"Failed to open image from path: {e}")
                    return {}
            
            # Apply preprocessing
            preprocessed_image = self.preprocessor.preprocess_for_ocr(image, preprocess_method)
            
            # Get OCR data using existing TesseractOCR
            ocr_data = self.tesseract.get_ocr_data(preprocessed_image, config)
            
            return ocr_data
            
        except Exception as e:
            logger.error(f"Enhanced OCR data extraction failed: {e}")
            return {}
    
    def extract_text_from_memory(self, file_content, preprocess_method="adaptive", config=''):
        """
        Extract text from image content in memory.
        
        Args:
            file_content: Raw bytes of the image content
            preprocess_method: Image preprocessing method to use
            config: Additional Tesseract configuration
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(file_content))
            
            # Use the extract_text method
            return self.extract_text(image, preprocess_method, config)
            
        except Exception as e:
            logger.error(f"Failed to extract text from memory: {e}")
            return ""
    
    def _clean_ocr_text(self, text):
        """
        Clean up OCR text for better readability.
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines that are just spaces or punctuation
        lines = []
        for line in text.split("\n"):
            # Skip lines that are just spaces, dots, or dashes
            if line.strip() and not all(c in ' .-=_|' for c in line):
                lines.append(line)
        
        # Join with newlines
        text = "\n".join(lines)
        
        # Fix common OCR errors
        replacements = {
            '|': 'I',
            '1l': 'II',
            '0': 'O',
            '《': '<',
            '》': '>',
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def compare_preprocessing_methods(self, image, config=''):
        """
        Compare different preprocessing methods on the same image.
        
        Args:
            image: PIL Image, numpy array, or file path
            config: Additional Tesseract configuration
            
        Returns:
            dict: Dictionary with results from different preprocessing methods
        """
        methods = ["none", "otsu", "adaptive", "advanced"]
        results = {}
        
        for method in methods:
            results[method] = self.extract_text(image, method, config)
        
        return results

# Add missing import for BytesIO
from io import BytesIO