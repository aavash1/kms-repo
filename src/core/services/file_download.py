# src/core/services/file_download.py
import logging
import aiohttp
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

async def download_file_from_url(url: str, max_retries: int = 3, retry_delay: float = 2.0) -> Optional[bytes]:
    """
    Download a file directly from a URL with retry logic.

    Args:
        url: The URL to download from
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Binary content of the file or None if download failed
    """
    if not url:
        logger.error("No URL provided for download")
        return None

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        # Attempt to read the response text for more details
                        response_text = await response.text() if response.status >= 400 else ""
                        logger.error(
                            f"Failed to download file from URL: {url}, status code: {response.status}, "
                            f"details: {response_text[:500] if response_text else 'No additional details'}"
                        )
                        if attempt < max_retries - 1:  # Don't sleep after the last attempt
                            await asyncio.sleep(retry_delay)
                        continue
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Connection error downloading file from {url}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            logger.error(f"Unexpected error downloading file from {url}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
    return None