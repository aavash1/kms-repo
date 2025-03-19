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

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Check the Content-Type header to ensure it's not an HTML error page.
                        content_type = response.headers.get("Content-Type", "")
                        if content_type.startswith("text/plain"):
                            error_text = await response.text()
                            logger.error(
                                f"Downloaded content from {url} appears to be an error page: {error_text[:200]}"
                            )
                            # Retry if not on the last attempt.
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return None
                        return content
                    else:
                        try:
                            response_text = await response.text()
                        except Exception:
                            response_text = "Unable to read response text"
                        logger.error(
                            f"Failed to download file from URL: {url}, status code: {response.status}, details: {response_text[:500]}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            return None
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