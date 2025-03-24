# src/core/services/file_download.py
import logging
import aiohttp
from typing import Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)

async def download_file_from_url(url: str, max_retries: int = 3, timeout: int = 30) -> Optional[Tuple[bytes, str]]:
    retry_count = 0
    async with aiohttp.ClientSession() as session:
        while retry_count < max_retries:
            try:
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download file from {url}: HTTP {response.status}")
                        retry_count += 1
                        continue
                    content = await response.read()
                    content_type = response.headers.get("Content-Type", "application/octet-stream")
                    logger.info(f"Successfully downloaded file from {url}, content_type: {content_type}")
                    return content, content_type
            except aiohttp.ClientError as e:
                logger.error(f"Error downloading file from {url}: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Max retries reached for {url}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error downloading file from {url}: {str(e)}")
                return None
    logger.warning(f"Download failed after {max_retries} retries for {url}")
    return None