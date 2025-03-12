# src/core/services/file_download.py
import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

async def download_file_from_url(url: str) -> Optional[bytes]:
    """
    Download a file directly from a URL.
    
    Args:
        url: The URL to download from
        
    Returns:
        Binary content of the file or None if download failed
    """
    try:
        if not url:
            logger.error("No URL provided for download")
            return None
            
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download file from URL: {url}, status code: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        return None