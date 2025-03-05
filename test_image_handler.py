# test_image_handler.py

from src.core.file_handlers.image_handler import ImageHandler
import sys
from pathlib import Path
import os
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
# You can specify an image path via command line argument for convenience.
# Usage: python test_image_handler.py path/to/image.jpg
if __name__ == "__main__":
    # Default image path (change this to your test image)
    sample_dir = Path(PROJECT_ROOT) / "sample_data"
    image_path = sample_dir /"report.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Initialize the handler (supporting Korean and English by default)
    handler = ImageHandler(languages=['ko', 'en'])
    
    # Test with Tesseract
    try:
        print("Running Tesseract OCR...")
        tesseract_results = handler.process_image(image_path, engine="tesseract")
        aligned_text = handler.reconstruct_aligned_text(tesseract_results)

        # print("Tesseract Extracted Text:")
        # for res in tesseract_results:
        #     print(res['text'])
        print (aligned_text)
        print("\n")
    except Exception as e:
        print(f"Tesseract OCR failed: {e}")

# For EasyOCR OCR results
try:
    print("Running EasyOCR OCR...")
    easyocr_results = handler.process_image(image_path, engine="easyocr")
    aligned_text = handler.reconstruct_aligned_text(easyocr_results)
    print("\n")
    # print("EasyOCR Extracted Text:")
    # for res in easyocr_results:
    #     print(res['text'])
    print (aligned_text)
except Exception as e:
    print(f"EasyOCR OCR failed: {e}")
