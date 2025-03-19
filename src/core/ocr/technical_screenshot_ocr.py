# src/core/ocr/technical_screenshot_ocr.py

import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalScreenshotOCR:
    """
    Specialized OCR for technical screenshots with tables, logs, and structured data.
    """
    
    def __init__(self):
        """Initialize the OCR processor"""
        # Set Tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Configure OCR parameters
        self.languages = "eng"  # English is sufficient for most technical screenshots
        self.config_options = {
            "table": "--psm 6 --oem 3 -c preserve_interword_spaces=1",  # Table mode
            "log": "--psm 6 --oem 3 -c preserve_interword_spaces=1",    # Log file mode
            "detail": "--psm 4 --oem 3"                                 # Detailed view mode
        }
    
    def preprocess_image(self, image, mode='table'):
        """
        Apply specialized preprocessing based on image content type
        
        Args:
            image: numpy array or PIL Image
            mode: 'table', 'log', or 'detail'
            
        Returns:
            PIL Image: Preprocessed image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image.copy()
            
        # Convert to grayscale if needed
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np_image
            
        if mode == 'table':
            # For table data: high contrast, sharpen, remove noise
            # Apply bilateral filter to preserve edges while removing noise
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply unsharp mask to sharpen text
            gaussian = cv2.GaussianBlur(thresh, (0, 0), 3)
            sharpened = cv2.addWeighted(thresh, 1.5, gaussian, -0.5, 0)
            
            # Scale the image to make text more readable
            scaled = cv2.resize(sharpened, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            
            return Image.fromarray(scaled)
            
        elif mode == 'log':
            # For log files: enhance text, reduce background
            # Increase contrast
            alpha = 1.3  # Contrast control
            beta = 10    # Brightness control
            contrast_adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(contrast_adjusted)
            
            # Apply thresholding
            _, thresh = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
            
            # Scale the image
            scaled = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            
            return Image.fromarray(scaled)
            
        else:  # detail mode
            # For detailed views: balance between detail and noise
            # Apply bilateral filter
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
            
            # Apply unsharp mask to improve details
            gaussian = cv2.GaussianBlur(blurred, (0, 0), 3)
            sharpened = cv2.addWeighted(blurred, 1.5, gaussian, -0.5, 0)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return Image.fromarray(thresh)
    
    def process_image(self, image_path, mode='table', output_path=None):
        """
        Process an image and extract text
        
        Args:
            image_path: Path to the image file
            mode: 'table', 'log', or 'detail'
            output_path: Path to save the processed image (optional)
            
        Returns:
            str: Extracted text
        """
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Preprocess the image
            preprocessed = self.preprocess_image(image, mode)
            
            # Save preprocessed image if output path is provided
            if output_path:
                preprocessed.save(output_path)
                logger.info(f"Preprocessed image saved to {output_path}")
            
            # Extract text using Tesseract
            config = self.config_options.get(mode, self.config_options['table'])
            text = pytesseract.image_to_string(preprocessed, lang=self.languages, config=config)
            
            # Clean up the text
            cleaned_text = self._clean_text(text, mode)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return ""
    
    def process_image_bytes(self, image_bytes, mode='table'):
        """
        Process image from bytes and extract text
        
        Args:
            image_bytes: Raw image bytes
            mode: 'table', 'log', or 'detail'
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert bytes to Image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocess the image
            preprocessed = self.preprocess_image(img, mode)
            
            # Extract text using Tesseract
            config = self.config_options.get(mode, self.config_options['table'])
            text = pytesseract.image_to_string(preprocessed, lang=self.languages, config=config)
            
            # Clean up the text
            cleaned_text = self._clean_text(text, mode)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return ""
    
    def _clean_text(self, text, mode='table'):
        """
        Clean up OCR text based on content type
        
        Args:
            text: Raw OCR text
            mode: 'table', 'log', or 'detail'
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove excess whitespace
        text = " ".join(text.split())
        
        if mode == 'table':
            # For table data, preserve structure
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Skip lines that are just separators
                if all(c in '-+|' for c in line.strip()):
                    continue
                    
                filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
            
        elif mode == 'log':
            # For log files, ensure timestamps and messages are preserved
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
            
        else:  # detail mode
            # For detailed views, general cleanup
            return text.strip()