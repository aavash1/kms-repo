import torch
import time
import sys
import logging
import warnings
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.file_handlers.doc_handler import DocHandler, DocxHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_document(handler, file_path, doc_type):
    try:
        start_time = time.time()
        logger.info(f"\n{'='*40}\nTesting {doc_type} file: {file_path}\n{'='*40}")
        
        # Convert to Path object for better path handling
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"{doc_type} file not found: {file_path}")
            
        # Text extraction test
        text = handler.extract_text(str(file_path))
        text_time = time.time() - start_time
        
        # Table extraction test
        table_start = time.time()
        tables = handler.extract_tables(str(file_path))
        table_time = time.time() - table_start
        
        # Results analysis
        logger.info(f"\n{'='*40}\nTest Results for {file_path}\n{'='*40}")
        logger.info(f"Text extraction time: {text_time:.2f}s")
        logger.info(f"Table extraction time: {table_time:.2f}s")
        logger.info(f"Total processing time: {text_time + table_time:.2f}s")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Tables found: {len(tables)}")
        
        # Sample output
        if text:
            sample = text[:500].replace('\n', ' ') + "..." if len(text) > 500 else text
            logger.info(f"\nText Sample:\n{sample}")
            
        if tables:
            logger.info(f"\nFirst Table Sample:")
            for i, row in enumerate(tables[0][:5]):
                logger.info(f"Row {i+1}: {row}")

    except Exception as e:
        logger.error(f"Error processing {doc_type}: {str(e)}", exc_info=True)
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return True

def main():
    # Use relative path from project root
    sample_file = Path(PROJECT_ROOT) / "sample_data" / "한국투자신탁_KIM-BACKUP-SVR1_kitmcdb1_SANclient_백업실패 장애보고서_20230605.doc"
    
    # Log the absolute path being used
    logger.info(f"Testing with file: {sample_file.absolute()}")
    
    test_files = [
        (str(sample_file), DocHandler, "DOC")
    ]
    
    results = {}
    
    for file_path, handler_cls, doc_type in test_files:
        handler = handler_cls()
        success = test_document(handler, file_path, doc_type)
        results[doc_type] = "PASSED" if success else "FAILED"
        del handler
    
    logger.info("\nTest Summary:")
    for doc_type, status in results.items():
        logger.info(f"{doc_type}: {status}")
        
    if all(status == "PASSED" for status in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()