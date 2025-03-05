# src/core/utils/post_processing.py
import regex as re
import cv2
import numpy as np
from PIL import Image

def fix_hyphenation(text):
    return re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

def preserve_line_breaks(text):
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines

def clean_extracted_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'(?<![가-힣])\s{2,}(?![가-힣])', ' ', text)
    
    # Fix common OCR mistakes
    replacements = {
        r'0l|Ol': '01',           # Fix zero-one confusion
        r'["""]': '"',            # Normalize quotes
        r'[\''']': "'",            # Normalize apostrophes
        r'1l': '11',              # Fix one-one confusion
        r'(?<!\d)1(?!\d)': 'I',   # Single 1 to I when not part of number
        r'rn|m': 'm',             # Fix rn/m confusion
        r'(?<=\d)O|o(?=\d)': '0'  # Fix O/0 confusion in numbers
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text.strip()


def preprocess_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected PIL.Image input")
    # Convert PIL image to NumPy for OpenCV processing
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better contrast
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morphological operations to enhance text regions
    kernel = np.ones((1, 1), np.uint8)
    processed_img = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(processed_img)

