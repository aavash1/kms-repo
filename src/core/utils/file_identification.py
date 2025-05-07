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
        str: Simplified file type (e.g., 'pdf', 'image', 'doc', 'hwp', 'msg').
    """
    # Direct Excel file detection based on extension - highest priority
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in ['.xlsx', '.xls']:
            logger.debug(f"Excel file detected directly by extension: {ext}")
            return 'excel'
        elif ext in ['.pptx', '.ppt']:
            logger.debug(f"PowerPoint file detected directly by extension: {ext}")
            return 'pptx'
    
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
        elif 'msword' in content_type_lower or 'officedocument.wordprocessingml' in content_type_lower:
            return 'doc'
        elif 'vnd.hancom.hwp' in content_type_lower:
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
            if mime_type == 'application/octet-stream':
                logger.debug(f"Octet-stream content sample: {file_content[:1024].hex()[:100]}...")
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
        elif 'text' in mime_type_lower:
            return 'text'
        elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
            return 'doc'
        elif 'vnd.hancom.hwp' in mime_type_lower:
            return 'hwp'
        elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
            return 'msg'
        elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
            return 'excel'
        elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
            return 'pptx'
        elif 'octet-stream' in mime_type_lower and filename:
            # Fallback to filename extension for octet-stream
            mime_type = _by_extension(filename)
            logger.debug(f"Octet-stream detected, using filename extension MIME type: {mime_type}")
            mime_type_lower = mime_type.lower()
            if 'pdf' in mime_type_lower:
                return 'pdf'
            elif 'image' in mime_type_lower:
                if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                    return 'image'
            elif 'text' in mime_type_lower:
                return 'text'
            elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
                return 'doc'
            elif 'vnd.hancom.hwp' in mime_type_lower:
                return 'hwp'
            elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
                return 'msg'
            elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
                return 'excel'
            elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
                return 'pptx'

    # Final fallback: use filename extension if available
    if filename:
        mime_type = _by_extension(filename)
        logger.debug(f"Falling back to filename extension, identified MIME type: {mime_type}")
        mime_type_lower = mime_type.lower()
        if 'pdf' in mime_type_lower:
            return 'pdf'
        elif 'image' in mime_type_lower:
            if 'png' in mime_type_lower or 'jpeg' in mime_type_lower or 'jpg' in mime_type_lower:
                return 'image'
        elif 'text' in mime_type_lower:
            return 'text'
        elif 'msword' in mime_type_lower or 'officedocument.wordprocessingml' in mime_type_lower:
            return 'doc'
        elif 'vnd.hancom.hwp' in mime_type_lower:
            return 'hwp'
        elif 'msg' in mime_type_lower or 'application/vnd.ms-outlook' in mime_type_lower:
            return 'msg'
        elif 'excel' in mime_type_lower or 'spreadsheetml' in mime_type_lower or 'application/vnd.ms-excel' in mime_type_lower:
            return 'excel'
        elif 'powerpoint' in mime_type_lower or 'presentationml' in mime_type_lower or 'application/vnd.ms-powerpoint' in mime_type_lower:
            return 'pptx'

    logger.warning(f"Could not determine file type for filename: {filename}, mime_type: {mime_type}")
    return 'unknown'