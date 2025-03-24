#src/core/utils/file_identification.py
import magic
import logging
import os
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_file_type1(file_path):
    mime=magic.Magic(mime=True)
    file_type=mime.from_file(file_path)
    return file_type


def get_file_type(file_content: bytes, content_type: Optional[str] = None) -> Optional[str]:
    """
    Identify the simplified file type using the Content-Type header (if provided) and/or file content.

    Parameters:
        file_content (bytes): The content of the file as bytes.
        content_type (Optional[str]): The Content-Type header from the HTTP response, if available.

    Returns:
        Optional[str]: Simplified file type (e.g., 'pdf', 'image', 'doc', 'hwp', 'msg'), or None if identification fails.
    """
    # First, try to determine the file type from the Content-Type header
    if content_type:
        content_type_lower = content_type.lower()
        logger.debug(f"Using provided Content-Type: {content_type}")
        if 'pdf' in content_type_lower:
            return 'pdf'
        elif 'image' in content_type_lower:
            if 'png' in content_type_lower or 'jpeg' in content_type_lower or 'jpg' in content_type_lower:
                return 'image'
        elif 'text' in content_type_lower:
            return 'text'
        elif 'msword' in content_type_lower or 'officedocument' in content_type_lower:
            return 'doc'
        elif 'vnd.hancom.hwp' in content_type_lower:
            return 'hwp'
        elif 'msg' in content_type_lower or 'application/vnd.ms-outlook' in content_type_lower:
            return 'msg'
        else:
            logger.warning(f"Unsupported Content-Type: {content_type}, falling back to python-magic")

    # Fallback to python-magic if Content-Type is not provided or not recognized
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file_content)
        logger.debug(f"Identified MIME type using python-magic: {mime_type}")

        # Map MIME types to simplified file types
        mime_type_lower = mime_type.lower()
        if 'pdf' in mime_type_lower:
            return 'pdf'
        elif 'image' in mime_type_lower:
            if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                return 'image'
        elif 'text' in mime_type_lower:
            return 'text'
        elif 'msword' in mime_type_lower or 'officedocument' in mime_type_lower:
            return 'doc'
        elif 'vnd.hancom.hwp' in mime_type_lower:
            return 'hwp'
        elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
            return 'msg'
        else:
            logger.warning(f"Unsupported MIME type from python-magic: {mime_type}")
            return None
    except Exception as e:
        logger.error(f"Error identifying file type with python-magic: {e}", exc_info=True)
        return None