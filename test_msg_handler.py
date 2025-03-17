import time
import sys
import os
import warnings
import logging
from src.core.file_handlers.msg_handler import MSGHandler

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    try:
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            msg_handler = MSGHandler()
        
        file_path = "sample_data/RE 미래에셋증권 phisdb sanclient status 83 발생 이슈 사항 (3).msg"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MSG file not found: {file_path}")
        
        result = msg_handler.extract_text(file_path)
        status_codes = msg_handler.get_status_codes()
        
        if status_codes:
            print(f"Found status codes: {status_codes}")
        else:
            print("No status codes found in this file")
        
        print(f"Extracted Body:\n{result}\n")
        
        end_time = time.time()
        print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing MSG: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'msg_handler' in locals():
            msg_handler.cleanup()

if __name__ == "__main__":
    main()