import torch
import time
import sys
from src.core.file_handlers.pdf_handler import PDFHandler
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
import os


# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    try:
        start_time = time.time()
        
         # Suppress model loading messages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pdf_handler = PDFHandler()
        
            # Add file existence check
            file_path = "sample_data/장애지원_내용정리.pdf"
            file_path2 = "sample_data/Status code 96 - Barcode & Media ID.pdf"
            file_path1 = "sample_data/Status code 830.pdf"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
                
            result = pdf_handler.extract_text(file_path)
            
            status_codes = pdf_handler.get_status_codes()
            if status_codes:
                print(f"Found status codes: {status_codes}")
            else:
                print("No status codes found in this file")
                
                print(f"Extracted Text:\n{result}\n")

            end_time = time.time()
            
            print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
            print("Extracted Text:\n", result)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup
        if 'pdf_handler' in locals():
            del pdf_handler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()