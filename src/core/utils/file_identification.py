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


def get_file_type_from_extension(filename: str) -> str:
    """
    ⚡ FAST: Primary file type detection using only file extension.
    Handles edge cases like .null extensions and files without extensions.
    
    Args:
        filename (str): The filename with extension (e.g., "document.pdf", "file.null", "README")
        
    Returns:
        str: Simplified file type (e.g., 'pdf', 'image', 'doc', 'hwp', 'msg', 'txt', 'rtf', 'unknown')
    """
    if not filename or not isinstance(filename, str):
        logger.warning(f"Invalid filename provided: {filename}")
        return 'unknown'
    
    # Clean the filename - remove query parameters and get base name
    clean_filename = filename.split('?')[0]  # Remove query params
    clean_filename = os.path.basename(clean_filename)  # Get just the filename
    
    # Extract extension using pathlib (handles edge cases better than string operations)
    try:
        path_obj = Path(clean_filename)
        ext = path_obj.suffix.lower()
    except Exception as e:
        logger.error(f"Failed to extract extension from {filename}: {e}")
        return 'unknown'
    
    # 🚨 HANDLE PROBLEMATIC CASES
    if not ext:  # No extension (e.g., "README", "Dockerfile")
        logger.debug(f"No extension found for file: {filename}")
        return 'unknown'
    
    if ext == '.null':  # Handle .null files from database
        logger.warning(f"File with .null extension detected: {filename}")
        return 'unknown'
    
    if len(ext) > 10:  # Suspiciously long extension (likely corrupted)
        logger.warning(f"Suspicious long extension '{ext}' for file: {filename}")
        return 'unknown'
    
    # 🎯 EXTENSION TO FILE TYPE MAPPING
    extension_map = {
        # Documents
        '.pdf': 'pdf',
        '.doc': 'doc',
        '.docx': 'doc',
        '.hwp': 'hwp',
        '.rtf': 'rtf',
        
        # Text files
        '.txt': 'txt',
        '.md': 'txt',
        '.csv': 'txt',
        '.log': 'txt',
        '.conf': 'txt',
        '.ini': 'txt',
        '.cfg': 'txt',
        
        # Spreadsheets
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.xlsm': 'excel',
        '.xlsb': 'excel',
        
        # Presentations
        '.ppt': 'pptx',
        '.pptx': 'pptx',
        '.pps': 'pptx',
        '.ppsx': 'pptx',
        
        # Email
        '.msg': 'msg',
        '.eml': 'msg',
        
        # Images
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.gif': 'image',
        '.bmp': 'image',
        '.tiff': 'image',
        '.tif': 'image',
        '.webp': 'image',
        '.svg': 'image',
        '.ico': 'image',
        
        # Archives (treat as documents for now)
        '.zip': 'unknown',  # Don't process archives
        '.rar': 'unknown',
        '.7z': 'unknown',
        '.tar': 'unknown',
        '.gz': 'unknown',
    }
    
    file_type = extension_map.get(ext, 'unknown')
    
    if file_type == 'unknown':
        logger.debug(f"Extension '{ext}' not in supported list for file: {filename}")
    else:
        logger.debug(f"Extension '{ext}' mapped to file type: '{file_type}' for file: {filename}")
    
    return file_type


def get_file_type(
    file_content: bytes | None = None,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    🔄 FALLBACK: Content-based file type detection when extension fails.
    Only used when get_file_type_from_extension() returns 'unknown'.
    
    Parameters:
        file_content (bytes): The content of the file as bytes.
        filename (str): The name of the file, used for extension-based fallback.
        content_type (str): The Content-Type header from the HTTP response, if available.

    Returns:
        str: Simplified file type (e.g., 'pdf', 'image', 'doc', 'hwp', 'msg', 'txt', 'rtf').
    """
    logger.info(f"🔄 FALLBACK: Using content analysis for file: {filename}")
    
    # 🚀 FIRST: Try fast extension-based detection
    if filename:
        ext_type = get_file_type_from_extension(filename)
        if ext_type != 'unknown':
            logger.info(f"✅ Extension-based fallback succeeded: {ext_type}")
            return ext_type
    
    # 📊 SECOND: Try Content-Type header analysis
    if content_type:
        content_type_lower = content_type.lower()
        logger.debug(f"Analyzing Content-Type: {content_type}")
        
        # Map content types to file types
        content_type_map = {
            'pdf': 'pdf',
            'image': 'image',
            'text/plain': 'txt',
            'rtf': 'rtf',
            'msword': 'doc',
            'officedocument.wordprocessingml': 'doc',
            'hwp': 'hwp',
            'hancom': 'hwp',
            'x-hwp': 'hwp',
            'msg': 'msg',
            'application/vnd.ms-outlook': 'msg',
            'excel': 'excel',
            'spreadsheetml': 'excel',
            'application/vnd.ms-excel': 'excel',
            'powerpoint': 'pptx',
            'presentationml': 'pptx',
            'application/vnd.ms-powerpoint': 'pptx'
        }
        
        for key, file_type in content_type_map.items():
            if key in content_type_lower:
                logger.info(f"✅ Content-Type analysis succeeded: {file_type}")
                return file_type

    # 🔬 LAST RESORT: python-magic content analysis
    if file_content:
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(file_content)
            logger.debug(f"python-magic detected MIME type: {mime_type}")
            
            if not mime_type:
                logger.warning("python-magic returned empty MIME type")
                return 'unknown'
            
            mime_type_lower = mime_type.lower()
            
            # Handle specific MIME types
            mime_type_map = {
                'application/pdf': 'pdf',
                'application/x-hwp': 'hwp',
                'text/plain': 'txt',
                'application/rtf': 'rtf',
                'text/rtf': 'rtf',
                'application/msword': 'doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'doc',
                'application/vnd.ms-outlook': 'msg',
                'application/vnd.ms-excel': 'excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
                'application/vnd.ms-powerpoint': 'pptx',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx'
            }
            
            # Check exact matches first
            if mime_type_lower in mime_type_map:
                detected_type = mime_type_map[mime_type_lower]
                logger.info(f"✅ python-magic exact match: {detected_type}")
                return detected_type
            
            # Check partial matches
            partial_matches = {
                'pdf': 'pdf',
                'image': 'image',
                'text': 'txt',
                'rtf': 'rtf',
                'msword': 'doc',
                'wordprocessingml': 'doc',
                'hwp': 'hwp',
                'hancom': 'hwp',
                'outlook': 'msg',
                'excel': 'excel',
                'spreadsheetml': 'excel',
                'powerpoint': 'pptx',
                'presentationml': 'pptx'
            }
            
            for key, file_type in partial_matches.items():
                if key in mime_type_lower:
                    logger.info(f"✅ python-magic partial match '{key}': {file_type}")
                    return file_type
                    
            # Special handling for octet-stream
            if 'octet-stream' in mime_type_lower:
                logger.debug("Detected octet-stream, trying content signatures...")
                
                # RTF signature
                if file_content.startswith(b'{\\rtf'):
                    logger.info("✅ RTF detected by content signature")
                    return 'rtf'
                
                # HWP signature (OLE2)
                hwp_signature = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'
                if file_content.startswith(hwp_signature) and filename and filename.lower().endswith('.hwp'):
                    logger.info("✅ HWP detected by OLE signature + extension")
                    return 'hwp'
                
                # PDF signature
                if file_content.startswith(b'%PDF'):
                    logger.info("✅ PDF detected by content signature")
                    return 'pdf'
                    
        except Exception as e:
            logger.error(f"python-magic analysis failed: {e}")

    logger.warning(f"❌ All detection methods failed for filename: {filename}")
    return 'unknown'


def _by_extension(fname: str) -> str:
    """Return MIME derived from the filename extension or a generic
    `application/octet-stream` if guessing fails."""
    mime, _ = mimetypes.guess_type(fname)
    return (mime or "application/octet-stream").lower()


# 🎯 MAIN ENTRY POINT - Use this function in your code
def detect_file_type(filename: str, file_content: bytes = None, content_type: str = None) -> str:
    """
    🎯 MAIN ENTRY POINT: Comprehensive file type detection.
    
    Args:
        filename (str): Filename (required)
        file_content (bytes, optional): File content for fallback analysis
        content_type (str, optional): HTTP Content-Type header
        
    Returns:
        str: File type ('pdf', 'doc', 'hwp', 'txt', 'rtf', 'excel', 'pptx', 'msg', 'image', 'unknown')
        
    Usage:
        # Fast detection (recommended)
        file_type = detect_file_type("document.pdf")
        
        # With fallback
        file_type = detect_file_type("document.pdf", file_content=content, content_type="application/pdf")
    """
    # 🚀 PRIMARY: Fast extension-based detection
    file_type = get_file_type_from_extension(filename)
    
    if file_type != 'unknown':
        return file_type
    
    # 🔄 FALLBACK: Content-based detection
    if file_content or content_type:
        logger.info(f"Extension detection failed for {filename}, trying content analysis...")
        return get_file_type(file_content, filename=filename, content_type=content_type)
    
    logger.warning(f"No content available for fallback analysis: {filename}")
    return 'unknown'