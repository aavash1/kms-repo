#src/core/utils/file_identification.py
import magic
import logging
import os

logger = logging.getLogger(__name__)

def get_file_type1(file_path):
    mime=magic.Magic(mime=True)
    file_type=mime.from_file(file_path)
    return file_type


def get_file_type(file_path_or_content):
    """
    Identify the MIME type of a file, either from a file path or from file content in memory.
    
    Parameters:
    file_path_or_content (str or bytes): Path to the file or the file content as bytes
    
    Returns:
    str: MIME type of the file or empty string if identification fails
    """
    try:
        mime = magic.Magic(mime=True)
        if isinstance(file_path_or_content, str):
            # Handle file path
            if not os.path.exists(file_path_or_content):
                logger.warning(f"File not found: {file_path_or_content}")
                return ""
            file_type = mime.from_file(file_path_or_content)
            logger.debug(f"Detected MIME type for file {file_path_or_content}: {file_type}")
        elif isinstance(file_path_or_content, bytes):
            # Handle file content in memory
            file_type = mime.from_buffer(file_path_or_content)
            logger.debug(f"Detected MIME type from file content: {file_type}")
        else:
            logger.error(f"Invalid input type for file identification: {type(file_path_or_content)}")
            return ""
        return file_type
    except Exception as e:
        logger.error(f"Error identifying file type: {e}", exc_info=True)
        return ""