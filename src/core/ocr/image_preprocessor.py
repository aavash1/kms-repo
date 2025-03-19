# src/core/ocr/image_preprocessor.py

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Enhanced image preprocessing class for improving OCR results.
    This class applies various image processing techniques to make text more readable for OCR.
    """
    
    @staticmethod
    def preprocess_for_ocr(image, method="adaptive"):
        """
        Apply image preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image or numpy array
            method: Preprocessing method to use ("adaptive", "otsu", "advanced", or "none")
            
        Returns:
            PIL.Image: Preprocessed image ready for OCR
        """
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            np_image = np.array(image)
            # Convert RGB to BGR (OpenCV format)
            if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        else:
            np_image = image.copy()
        
        # Apply the selected preprocessing method
        if method == "none":
            # No preprocessing, just convert to grayscale
            processed_img = ImagePreprocessor._to_grayscale(np_image)
        elif method == "otsu":
            # Basic preprocessing with Otsu thresholding
            processed_img = ImagePreprocessor._preprocess_otsu(np_image)
        elif method == "advanced":
            # Advanced preprocessing with multiple techniques
            processed_img = ImagePreprocessor._preprocess_advanced(np_image)
        else:  # "adaptive" (default)
            # Adaptive thresholding for varying lighting conditions
            processed_img = ImagePreprocessor._preprocess_adaptive(np_image)
        
        # Convert back to PIL Image for Tesseract
        return Image.fromarray(processed_img)
    
    @staticmethod
    def _to_grayscale(image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def _preprocess_otsu(image):
        """Basic preprocessing with Otsu thresholding"""
        # Convert to grayscale
        gray = ImagePreprocessor._to_grayscale(image)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return thresh
    
    @staticmethod
    def _preprocess_adaptive(image):
        """Adaptive thresholding for varying lighting conditions"""
        # Convert to grayscale
        gray = ImagePreprocessor._to_grayscale(image)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    @staticmethod
    def _preprocess_advanced(image):
        """Advanced preprocessing with multiple techniques"""
        # Convert to grayscale
        gray = ImagePreprocessor._to_grayscale(image)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Dilate to make text more pronounced
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated
    
    @staticmethod
    def apply_all_methods(image):
        """
        Apply all preprocessing methods and return a dictionary of results.
        Useful for comparing different methods.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Dictionary with preprocessed images using different methods
        """
        methods = ["none", "otsu", "adaptive", "advanced"]
        results = {}
        
        for method in methods:
            results[method] = ImagePreprocessor.preprocess_for_ocr(image, method)
        
        return results