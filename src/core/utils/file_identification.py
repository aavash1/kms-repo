#src/core/utils/file_identification.py
import magic
import logging
import os

logger = logging.getLogger(__name__)

def get_file_type1(file_path):
    mime=magic.Magic(mime=True)
    file_type=mime.from_file(file_path)
    return file_type


def get_file_type(file_path):
    """
    Identify the MIME type of a file.
    
    Parameters:
    file_path (str): Path to the file
    
    Returns:
    str: MIME type of the file or empty string if identification fails
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
            
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        logger.error(f"Error identifying file type for {file_path}: {e}")
        return ""