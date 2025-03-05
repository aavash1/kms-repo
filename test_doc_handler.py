#/test_doc_handler.py
import os
import sys
import time
import warnings
import logging
import re
from pathlib import Path

# Disable all logging and warnings.
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Optionally, disable PDFMiner's logging if used inside your handler.
logging.getLogger("pdfminer").setLevel(logging.CRITICAL)

# Ensure the project root directory is in the path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import your AdvancedDocHandler.
from src.core.file_handlers.doc_handler import AdvancedDocHandler

def clean_extracted_text(text):
    """
    Remove lines that consist solely of page markers (e.g. "Page 1") and trim extra whitespace.
    """
    # Remove any line that contains only "Page" followed by a number.
    cleaned = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", text)
    return cleaned.strip()

def main():
    # Define the sample file path.
    sample_dir = Path(PROJECT_ROOT) / "sample_data"
    test_file = sample_dir / "백업실패 장애보고서.doc"
    
    # Create the AdvancedDocHandler and extract the text.
    handler = AdvancedDocHandler()
    extracted_text = handler.extract_text(str(test_file))
    handler.__del__()  # Explicitly clean up temporary files, if needed.
    
    # Clean the extracted text to remove page markers.
    final_text = clean_extracted_text(extracted_text)
    status_codes = handler.get_status_codes()
    if status_codes:
                print(f"Found status codes: {status_codes}")
                print(f"Extracted Text:\n{final_text}\n")
    else:
                print("No status codes found in this file")
                
                print(f"Extracted Text:\n{final_text}\n")
    
    # Print only the extracted text.
    #print(final_text)

if __name__ == "__main__":
    main()