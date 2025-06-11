# src/core/utils/file_identification.py
from __future__ import annotations
import magic
import mimetypes
import logging
import os
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Explicitly initialize mimetypes
mimetypes.init()
# Add Excel file types if not already present
mimetypes.add_type('application/vnd.ms-excel', '.xls')
mimetypes.add_type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx')
mimetypes.add_type('application/x-hwp', '.hwp')
mimetypes.add_type('application/haansofthwp', '.hwp')
mimetypes.add_type('application/vnd.hancom.hwp', '.hwp')
mimetypes.add_type('text/plain', '.txt')
mimetypes.add_type('application/rtf', '.rtf')
mimetypes.add_type('text/rtf', '.rtf')


def _by_extension(fname: str) -> str:
    """Return MIME derived from the filename extension or a generic
    `application/octet-stream` if guessing fails."""
    mime, _ = mimetypes.guess_type(fname)
    return (mime or "application/octet-stream").lower()

def get_file_type(
    file_content: bytes | None = None,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    Identify the simplified file type using the Content-Type header, file content, or filename extension.

    Parameters:
        file_content (bytes): The content of the file as bytes.
        filename (str): The name of the file, used for extension-based fallback.
        content_type (str): The Content-Type header from the HTTP response, if available.

    Returns:
        str: Simplified file type (e.g., 'pdf', 'image', 'doc', 'hwp', 'msg', 'txt', 'rtf').
    """
    # Direct file detection based on extension - highest priority for certain types
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in ['.xlsx', '.xls']:
            logger.debug(f"Excel file detected directly by extension: {ext}")
            return 'excel'
        elif ext in ['.pptx', '.ppt']:
            logger.debug(f"PowerPoint file detected directly by extension: {ext}")
            return 'pptx'
        elif ext == '.hwp':
            logger.debug(f"HWP file detected directly by extension: {ext}")
            return 'hwp'
        elif ext == '.txt':
            logger.debug(f"TXT file detected directly by extension: {ext}")
            return 'txt'
        elif ext == '.rtf':
            logger.debug(f"RTF file detected directly by extension: {ext}")
            return 'rtf'
    
    # First, try to determine the file type from the Content-Type header
    if content_type:
        content_type_lower = content_type.lower()
        logger.debug(f"Using provided Content-Type: {content_type}")
        if 'pdf' in content_type_lower:
            return 'pdf'
        elif 'image' in content_type_lower:
            if 'png' in content_type_lower or 'jpeg' in content_type_lower or 'jpg' in content_type_lower:
                return 'image'
        elif 'text/plain' in content_type_lower:
            # Check if it's specifically a TXT file by filename
            if filename and filename.lower().endswith('.txt'):
                return 'txt'
            return 'txt'  # Default plain text to txt
        elif 'text/rtf' in content_type_lower or 'application/rtf' in content_type_lower:
            return 'rtf'
        elif 'text' in content_type_lower and 'rtf' in content_type_lower:
            return 'rtf'
        elif 'msword' in content_type_lower or 'officedocument.wordprocessingml' in content_type_lower:
            return 'doc'
        elif 'hwp' in content_type_lower or 'hancom' in content_type_lower or 'x-hwp' in content_type_lower:
            return 'hwp'
        elif 'msg' in content_type_lower or 'application/vnd.ms-outlook' in content_type_lower:
            return 'msg'
        elif 'excel' in content_type_lower or 'spreadsheetml' in content_type_lower or 'application/vnd.ms-excel' in content_type_lower:
            return 'excel'
        elif 'powerpoint' in content_type_lower or 'presentationml' in content_type_lower or 'application/vnd.ms-powerpoint' in content_type_lower:
            return 'pptx'
        else:
            logger.warning(f"Unsupported Content-Type: {content_type}, falling back to python-magic")

    # Fallback to python-magic if Content-Type is not provided or not recognized
    mime_type = None
    if file_content:
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(file_content)
            logger.debug(f"Identified MIME type using python-magic: {mime_type}")
            
            # Check for specific file types
            if mime_type == 'application/x-hwp':
                logger.info("HWP file detected using python-magic")
                return 'hwp'
            elif mime_type == 'text/plain':
                # For plain text, check filename to distinguish TXT from other text formats
                if filename:
                    ext = Path(filename).suffix.lower()
                    if ext == '.txt':
                        return 'txt'
                    elif ext == '.rtf':
                        return 'rtf'
                return 'txt'  # Default plain text to txt
            elif mime_type in ['application/rtf', 'text/rtf']:
                return 'rtf'
                
            # Log a sample of the file content for debugging octet-stream
            if mime_type == 'application/octet-stream':
                logger.debug(f"Octet-stream content sample: {file_content[:1024].hex()[:100]}...")
                
                # Try to detect RTF by content signature
                if file_content.startswith(b'{\\rtf'):
                    logger.info("RTF file detected by content signature")
                    return 'rtf'
                
                # Try to detect HWP by file signature
                if len(file_content) > 18:
                    hwp_signature = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'  # OLE2 signature
                    if file_content.startswith(hwp_signature):
                        # If the file has an OLE signature and .hwp extension, it's likely an HWP file
                        if filename and filename.lower().endswith('.hwp'):
                            logger.info("HWP file detected by OLE signature and extension")
                            return 'hwp'
        except Exception as e:
            logger.error(f"Error identifying file type with python-magic: {e}", exc_info=True)

    # Map MIME types to simplified file types
    if mime_type:
        mime_type_lower = mime_type.lower()
        if 'pdf' in mime_type_lower:
            return 'pdf'
        elif 'image' in mime_type_lower:
            if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                return 'image'
        elif 'text/plain' in mime_type_lower:
            # Check filename for specific text format
            if filename:
                ext = Path(filename).suffix.lower()
                if ext == '.txt':
                    return 'txt'
                elif ext == '.rtf':
                    return 'rtf'
            return 'txt'  # Default plain text to txt
        elif 'text/rtf' in mime_type_lower or 'application/rtf' in mime_type_lower:
            return 'rtf'
        elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
            return 'doc'
        elif 'hwp' in mime_type_lower or 'hancom' in mime_type_lower:
            return 'hwp'
        elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
            return 'msg'
        elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
            return 'excel'
        elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
            return 'pptx'
        elif 'octet-stream' in mime_type_lower and filename:
            # Special case for files that are detected as octet-stream
            ext = Path(filename).suffix.lower()
            if ext == '.hwp':
                logger.info(f"HWP file identified by extension after octet-stream detection: {filename}")
                return 'hwp'
            elif ext == '.txt':
                logger.info(f"TXT file identified by extension after octet-stream detection: {filename}")
                return 'txt'
            elif ext == '.rtf':
                logger.info(f"RTF file identified by extension after octet-stream detection: {filename}")
                return 'rtf'
                
            # Fallback to filename extension for octet-stream
            mime_type = _by_extension(filename)
            logger.debug(f"Octet-stream detected, using filename extension MIME type: {mime_type}")
            mime_type_lower = mime_type.lower()
            if 'pdf' in mime_type_lower:
                return 'pdf'
            elif 'image' in mime_type_lower:
                if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                    return 'image'
            elif 'text/plain' in mime_type_lower:
                return 'txt'
            elif 'text/rtf' in mime_type_lower or 'application/rtf' in mime_type_lower:
                return 'rtf'
            elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
                return 'doc'
            elif 'hwp' in mime_type_lower or 'x-hwp' in mime_type_lower or 'hancom' in mime_type_lower:
                return 'hwp'
            elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
                return 'msg'
            elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
                return 'excel'
            elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
                return 'pptx'

    # Final fallback: use filename extension if available
    if filename:
        ext = Path(filename).suffix.lower()
        if ext == '.hwp':
            logger.info(f"HWP file identified in final fallback by extension: {filename}")
            return 'hwp'
        elif ext == '.txt':
            logger.info(f"TXT file identified in final fallback by extension: {filename}")
            return 'txt'
        elif ext == '.rtf':
            logger.info(f"RTF file identified in final fallback by extension: {filename}")
            return 'rtf'
            
        mime_type = _by_extension(filename)
        logger.debug(f"Falling back to filename extension, identified MIME type: {mime_type}")
        mime_type_lower = mime_type.lower()
        if 'pdf' in mime_type_lower:
            return 'pdf'
        elif 'image' in mime_type_lower:
            if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                return 'image'
        elif 'text/plain' in mime_type_lower:
            return 'txt'
        elif 'text/rtf' in mime_type_lower or 'application/rtf' in mime_type_lower:
            return 'rtf'
        elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
            return 'doc'
        elif 'hwp' in mime_type_lower or 'x-hwp' in mime_type_lower or 'hancom' in mime_type_lower:
            return 'hwp'
        elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
            return 'msg'
        elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
            return 'excel'
        elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
            return 'pptx'

    # Last resort extension checks
    if filename:
        ext = Path(filename).suffix.lower()
        if ext == '.hwp':
            logger.warning(f"HWP file detected by extension as last resort: {filename}")
            return 'hwp'
        elif ext == '.txt':
            logger.warning(f"TXT file detected by extension as last resort: {filename}")
            return 'txt'
        elif ext == '.rtf':
            logger.warning(f"RTF file detected by extension as last resort: {filename}")
            return 'rtf'

    logger.warning(f"Could not determine file type for filename: {filename}, mime_type: {mime_type}")
    return 'unknown'