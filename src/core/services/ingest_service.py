import os
import tempfile
import json
import ollama
import hashlib
import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import List, Optional, Dict, Any
from src.core.services.file_utils import process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection, process_file_content, CHROMA_DIR, process_html_content
from src.core.services.file_download import download_file_from_url
from src.core.mariadb_db.mariadb_connector import MariaDBConnector
import logging
from bs4 import BeautifulSoup
import re
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
import boto3

from src.core.file_handlers.factory import FileHandlerFactory
from src.core.services.static_data_cache import StaticDataCache
from src.core.services.knowledge_graph import KnowledgeGraph

from src.core.file_handlers.pdf_handler import PDFHandler
from src.core.file_handlers.hwp_handler import HWPHandler
from src.core.file_handlers.doc_handler import AdvancedDocHandler
from src.core.file_handlers.msg_handler import MSGHandler
from src.core.file_handlers.image_handler import ImageHandler
from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor  # Updated import
from src.core.services.static_data_cache import static_data_cache
from src.core.utils.file_identification import get_file_type
from src.core.services.knowledge_graph import knowledge_graph

load_dotenv()

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self, db_connector: MariaDBConnector = None, model_manager=None):
        self.sample_data_dir = os.path.join(os.getcwd(), "sample_data")
        os.makedirs(self.sample_data_dir, exist_ok=True)
        self.db_connector = db_connector or MariaDBConnector()
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
        self.semaphore = asyncio.Semaphore(5)  # General concurrency limit
        self.chroma_lock = asyncio.Lock()
        self.model_manager = model_manager
        self.static_data_cache = static_data_cache or StaticDataCache()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        FileHandlerFactory.initialize(model_manager)

        # Resource limits for downloads and embeddings
        self.MAX_CONCURRENT_DOWNLOADS = 5
        self.MAX_CONCURRENT_EMBEDDINGS = 10
        self.download_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_DOWNLOADS)
        self.embedding_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_EMBEDDINGS)

        # Initialize handlers using FileHandlerFactory
        self.handlers = {
            'pdf': FileHandlerFactory.get_handler_for_extension('pdf'),
            'image': FileHandlerFactory.get_handler_for_extension('png'),  # Also handles jpg, jpeg
            'hwp': FileHandlerFactory.get_handler_for_extension('hwp'),
            'doc': FileHandlerFactory.get_handler_for_extension('doc'),  # Also handles docx
            'msg': FileHandlerFactory.get_handler_for_extension('msg'),
        }

        # Initialize specific handlers
        if model_manager:
            self.pdf_handler = PDFHandler(model_manager=model_manager)
            self.image_handler = ImageHandler(model_manager=model_manager)
            self.msg_handler = MSGHandler(model_manager=model_manager)
            self.doc_handler = AdvancedDocHandler(model_manager=model_manager)
            self.hwp_handler = HWPHandler(model_manager=model_manager)
            self.html_handler = HTMLContentHandler(model_manager=model_manager)
            self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
        else:
            self.pdf_handler = PDFHandler()
            self.image_handler = ImageHandler()
            self.msg_handler = MSGHandler()
            self.doc_handler = AdvancedDocHandler()
            self.hwp_handler = HWPHandler()
            self.html_handler = HTMLContentHandler()
            self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")

        self.os_version_map = {
            "1": "유닉스", "2": "리눅스", "3": "유닉스부트", "4": "RHEL", "5": "CentOS",
            "6": "Unix", "7": "Windows", "8": "Solaris", "9": "AIX", "10": "HP-UX", "11": "모름"
        }

    def _sanitize_filename(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return f"{hashlib.md5(filename.encode('utf-8')).hexdigest()[:10]}{ext}"

    def _is_url_expired(self, url: str) -> bool:
        from urllib.parse import parse_qs
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            amz_date = query_params.get("X-Amz-Date", [None])[0]
            expires = query_params.get("X-Amz-Expires", [None])[0]
            if not amz_date or not expires:
                logger.warning(f"Missing expiration data in URL: {url}")
                return False
            from datetime import datetime, timedelta
            date_format = "%Y%m%dT%H%M%SZ"
            expiration_time = datetime.strptime(amz_date, date_format) + timedelta(seconds=int(expires))
            return datetime.utcnow() > expiration_time
        except Exception as e:
            logger.error(f"Error checking URL expiration for {url}: {str(e)}")
            return False

    async def _process_file_content(self, file_content: bytes, logical_nm: str, error_code_id: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=200) -> Dict[str, Any]:
        temp_file = None
        temp_file_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self._sanitize_filename(logical_nm))
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name

            extraction_result = await process_file_content(
                file_content=file_content,
                filename=logical_nm,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_manager=self.model_manager
            )
            if extraction_result["status"] != "success":
                logger.warning(f"Failed to process {logical_nm}: {extraction_result['message']}")
                return extraction_result

            return extraction_result  # Already includes chunking and embedding with batching from file_utils.py
        except Exception as e:
            logger.error(f"Error processing file content for {logical_nm}: {e}")
            return {"logical_nm": logical_nm, "status": "error", "message": str(e)}
        finally:
            if temp_file and not temp_file.closed:
                temp_file.close()
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")

    async def process_uploaded_files_optimized(self, files: List[UploadFile], status_code: str, metadata: Optional[str] = None, chunk_size=1000, chunk_overlap=200, model_manager=None):
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized."}

        temp_files = []
        try:
            all_chunks = []
            chunk_metadata = []
            chunk_ids = []

            for file in files:
                if file.size > self.MAX_FILE_SIZE:
                    results.append({"filename": file.filename, "status": "error", "message": f"File exceeds maximum size of {self.MAX_FILE_SIZE} bytes"})
                    continue

                temp_file_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix=self._sanitize_filename(file.filename)) as temp_file:
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)
                    bytes_written = 0
                    while chunk := await file.read(1024 * 1024):
                        temp_file.write(chunk)
                        bytes_written += len(chunk)
                    logger.debug(f"Wrote {bytes_written} bytes to {temp_file_path}")

                extraction_result = process_file(temp_file_path, chunk_size, chunk_overlap, model_manager=self.model_manager)
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({"filename": file.filename, "status": "error", "message": "No text was extracted from the file."})
                    continue

                metadata_dict = json.loads(metadata) if metadata else {}
                base_metadata = {"filename": file.filename, "status_code": status_code, **metadata_dict}

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file.filename}_chunk_{i}"
                    if chunk_id in set(chromadb_collection.get()["ids"]):
                        logger.warning(f"Skipping duplicate embedding ID: {chunk_id}")
                        continue
                    all_chunks.append(chunk)
                    chunk_metadata.append({**base_metadata, "chunk_index": i, "chunk_count": len(chunks)})
                    chunk_ids.append(chunk_id)

            if all_chunks:
                batch_size = 20
                processed_chunks = 0
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]
                    batch_meta = chunk_metadata[i:i + batch_size]

                    async with self.embedding_semaphore:  # Limit concurrent embeddings
                        embed_tasks = [self._embed_text(chunk) for chunk in batch_chunks]
                        embeddings = await asyncio.gather(*embed_tasks)

                    async with self.chroma_lock:
                        valid_embeddings = [emb for emb in embeddings if emb is not None]
                        valid_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
                        if valid_embeddings:
                            chromadb_collection.add(
                                ids=[batch_ids[idx] for idx in valid_indices],
                                embeddings=valid_embeddings,
                                documents=[batch_chunks[idx] for idx in valid_indices],
                                metadatas=[batch_meta[idx] for idx in valid_indices]
                            )
                            processed_chunks += len(valid_embeddings)
                            logger.debug(f"Processed batch of {len(valid_embeddings)} chunks, total: {processed_chunks}")

                for file in files:
                    file_chunks = [cid for cid in chunk_ids if cid.startswith(file.filename)]
                    if file_chunks:
                        results.append({
                            "filename": file.filename,
                            "status": "success",
                            "message": f"File processed successfully with {len(file_chunks)} chunks.",
                            "chunk_count": len(file_chunks)
                        })

            successful = len([r for r in results if r["status"] == "success"])
            return {
                "status": "completed",
                "total_files": len(files),
                "successful": successful,
                "failed": len(files) - successful,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error in process_uploaded_files_optimized: {e}")
            return {"status": "error", "message": str(e), "results": results}
        finally:
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        try:
            embed_response = await asyncio.to_thread(ollama.embed, model="mxbai-embed-large", input=text)  # Run in thread since ollama.embed is sync
            embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
            if not embedding or len(embedding) != 1024:
                logger.error(f"Invalid embedding generated for text: {text[:30]}...")
                return None
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    async def process_text_content(self, text_content: str, status_code: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200):
        try:
            if not text_content.strip():
                return {"content_id": f"content_{status_code}", "status": "error", "message": "No text content provided."}
            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                return {"content_id": f"content_{status_code}", "status": "error", "message": "ChromaDB collection is not initialized."}
            cleaned_text = clean_extracted_text(text_content)
            from src.core.utils.text_chunking import chunk_text
            chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
            if not chunks:
                return {"content_id": f"content_{status_code}", "status": "error", "message": "Failed to chunk text content."}
            base_metadata = {"content_type": "text", "status_code": status_code, **(metadata or {})}
            batch_size = 20
            chunk_ids = []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = [f"content_{status_code}_{i + j}" for j in range(len(batch_chunks))]
                batch_meta = [{**base_metadata, "chunk_index": i + j, "chunk_count": len(chunks)} for j in range(len(batch_chunks))]

                async with self.embedding_semaphore:
                    embed_tasks = [self._embed_text(chunk) for chunk in batch_chunks]
                    embeddings = await asyncio.gather(*embed_tasks)

                async with self.chroma_lock:
                    valid_embeddings = [emb for emb in embeddings if emb is not None]
                    valid_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
                    if valid_embeddings:
                        chromadb_collection.add(
                            ids=[batch_ids[idx] for idx in valid_indices],
                            embeddings=valid_embeddings,
                            documents=[batch_chunks[idx] for idx in valid_indices],
                            metadatas=[batch_meta[idx] for idx in valid_indices]
                        )
                        chunk_ids.extend([batch_ids[idx] for idx in valid_indices])

            return {"content_id": f"content_{status_code}", "status": "success", "message": f"Text content processed with {len(chunk_ids)} chunks.", "chunk_count": len(chunk_ids)}
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return {"content_id": f"content_{status_code}", "status": "error", "message": str(e)}

    async def process_embedded_images(self, content: str, error_code_id: str, metadata: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        results = []
        skipped_images = []
        img_urls = re.findall(r'<img[^>]+src=["\'](.*?)["\']', content)
        if not img_urls:
            return results, skipped_images
        chromadb_collection = get_chromadb_collection()
        processed_ids = set(chromadb_collection.get()["ids"]) if chromadb_collection else set()

        async def process_image_url(url):
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            df = self.db_connector.fetch_dataframe(
                "SELECT file_id, logical_nm, url FROM attachment_files WHERE url = ? AND delete_yn = 'N'",
                (url,)
            )
            if df.empty:
                df = self.db_connector.fetch_dataframe(
                    "SELECT file_id, logical_nm, url FROM attachment_files WHERE url LIKE ? AND delete_yn = 'N'",
                    (f"{base_url}%",)
                )
            if df.empty:
                logger.warning(f"Image URL {url} not found in attachment_files table.")
                skipped_images.append({"url": url, "reason": "Not found in attachment_files"})
                return {"url": url, "status": "error", "message": "Image not found in attachment_files"}

            for _, row in df.iterrows():
                file_id, logical_nm, file_url = row["file_id"], row["logical_nm"], row["url"]
                chunk_id_base = f"{logical_nm}_chunk_0"
                if chunk_id_base in processed_ids:
                    logger.info(f"Skipping duplicate image: {logical_nm}")
                    continue
                if self._is_url_expired(file_url):
                    logger.warning(f"Skipping expired URL for image {logical_nm}: {file_url}")
                    new_url = await self._refresh_presigned_url(file_url, logical_nm, row.get("physical_nm", ""))
                    if new_url:
                        file_url = new_url
                    else:
                        return {"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL expired and could not be refreshed"}

                async with self.download_semaphore:  # Limit concurrent downloads
                    download_result = await download_file_from_url(file_url)
                
                # Handle tuple return value
                if isinstance(download_result, tuple):
                    file_content, content_type = download_result
                else:
                    file_content = download_result
                    content_type = None

                if not file_content:
                    logger.warning(f"Failed to download image from {file_url}")
                    return {"logical_nm": logical_nm, "status": "error", "message": "Failed to download image after retries"}

                image_handler = ImageHandler(model_manager=self.model_manager)
                text = await image_handler.extract_text_from_memory(file_content)  # Now file_content is bytes
                if not text or not text.strip():
                    logger.warning(f"No text extracted from image {logical_nm}")
                    return {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from image"}

                image_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": file_url, "error_code_id": error_code_id, **metadata}
                return await self._process_file_content(file_content, logical_nm, error_code_id, image_metadata)

        tasks = [process_image_url(url) for url in img_urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r], skipped_images

    async def _refresh_presigned_url(self, old_url: str, logical_nm: str, physical_nm: str) -> Optional[str]:
        try:
            s3_url = os.getenv("S3_URL")
            s3_access_key = os.getenv("S3_ACCESS_KEY")
            s3_secret_key = os.getenv("S3_SECRET_KEY")
            s3_bucket_name = os.getenv("S3_BUCKET_NAME")
            s3_region = os.getenv("S3_REGION")

            parsed_url = urlparse(old_url)
            key = parsed_url.path.lstrip('/')
            s3_client = boto3.client(
                's3', endpoint_url=s3_url, aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key, region_name=s3_region
            )
            new_url = s3_client.generate_presigned_url(
                ClientMethod='get_object', Params={'Bucket': s3_bucket_name, 'Key': key}, ExpiresIn=3600
            )
            logger.info(f"Refreshed URL for {logical_nm} from {old_url} to {new_url}")
            return new_url
        except Exception as e:
            logger.error(f"Failed to refresh URL for {logical_nm}: {e}")
            return None

    async def process_mariadb_troubleshooting_data(self, html_handler=None, vision_extractor=None) -> Dict[str, Any]:
        try:
            # Use the instance's html_handler and vision_extractor if not provided
            html_handler = html_handler or self.html_handler
            vision_extractor = vision_extractor or self.vision_extractor

            if not html_handler or not vision_extractor:
                raise ValueError("HTML handler and vision extractor must be provided")

            if not self.db_connector.is_connection_active():
                self.db_connector.connect()
                logger.debug("Re-established MariaDB connection.")

            reports_df = self.db_connector.get_unprocessed_troubleshooting_reports()
            if reports_df.empty:
                logger.info("No unprocessed troubleshooting reports found")
                return {"status": "success", "message": "No reports to process", "processed_count": 0, "details": []}

            grouped_reports = reports_df.groupby('error_code_nm')
            results = []
            processed_count = 0
            processed_urls = set()
            processed_report_ids = set()  # Track processed report IDs

            for error_code_nm, group in grouped_reports:
                logger.info(f"Processing reports for error_code_nm: {error_code_nm}")
                for _, row in group.iterrows():
                    html_content = row.get('content', '')
                    report_id = row.get('resolve_id', 'unknown')
                    url = row.get('url', None)
                    logical_nm = row.get('logical_nm', None)

                    metadata = {
                        "error_code_nm": str(error_code_nm),
                        "client_name": row.get('client_name', ''),
                        "os_version": row.get('os_version', ''),
                        "resolve_id": report_id
                    }

                    # Skip if report_id has already been processed
                    if report_id in processed_report_ids:
                        logger.info(f"Skipping duplicate report_id: {report_id}")
                        continue
                    processed_report_ids.add(report_id)

                    html_result = await process_html_content(
                        html_content=html_content,
                        metadata=metadata,
                        html_handler=html_handler,
                        vision_extractor=vision_extractor
                    )
                    results.append(html_result)

                    if html_result["status"] == "success":
                        processed_count += 1

                    if url and logical_nm and url not in processed_urls:
                        processed_urls.add(url)
                        logger.info(f"Processing attachment for resolve ID: {report_id}: {logical_nm}")
                        try:
                            async with self.download_semaphore:  # Limit downloads
                                download_result = await download_file_from_url(url)
                            
                            # Handle tuple return value
                            if isinstance(download_result, tuple):
                                file_content, content_type = download_result
                            else:
                                file_content = download_result
                                content_type = None

                            if not file_content:
                                logger.warning(f"Failed to download file from {url}")
                                results.append({"report_id": report_id, "logical_nm": logical_nm, "status": "error", "message": f"Failed to download file from {url}"})
                                continue

                            file_extension = logical_nm.lower().split('.')[-1]
                            if file_extension in ['pdf', 'png', 'jpg', 'jpeg']:
                                # Use process_file_content for consistency
                                file_metadata = {
                                    "error_code_nm": str(error_code_nm),
                                    "client_name": row.get('client_name', ''),
                                    "os_version": row.get('os_version', ''),
                                    "resolve_id": report_id,
                                    "logical_nm": logical_nm,
                                    "url": url
                                }
                                result = await process_file_content(
                                    file_content=file_content,
                                    filename=logical_nm,
                                    metadata=file_metadata,
                                    model_manager=self.model_manager
                                )
                            else:
                                result = {"report_id": report_id, "logical_nm": logical_nm, "status": "warning", "message": f"Unsupported file type: {file_extension}"}

                            results.append(result)
                            if result["status"] == "success":
                                processed_count += 1
                        except Exception as e:
                            logger.error(f"Error processing attachment for report {report_id}: {str(e)}", exc_info=True)
                            results.append({"report_id": report_id, "logical_nm": logical_nm, "status": "error", "message": f"Error processing attachment: {str(e)}"})
                    else:
                        logger.warning(f"No valid attachment URL or duplicate URL for report {report_id}")

            self.db_connector.close()
            logger.info(f"Processed {processed_count} troubleshooting reports")

            from src.core.services.file_utils import set_globals, get_chromadb_collection, get_rag_chain, get_global_prompt, get_workflow, get_memory
            from langchain_ollama import OllamaEmbeddings
            import chromadb
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain.vectorstores import Chroma

            chroma_coll = get_chromadb_collection()
            persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            vector_store = Chroma(
                client=persistent_client,
                embedding_function=embeddings,
                collection_name="netbackup_docs",
                collection_metadata={"hnsw:space": "cosine"}
            )
            set_globals(chroma_coll=chroma_coll, rag=get_rag_chain(), vect_store=vector_store, prompt=get_global_prompt(), workflow=get_workflow(), memory=get_memory())
            logger.debug("Updated global state after ingestion")

            return {
                "status": "success",
                "message": f"Processed {processed_count} out of {len(reports_df)} reports",
                "processed_count": processed_count,
                "details": results
            }
        except Exception as e:
            logger.error(f"Error in process_mariadb_troubleshooting_data: {e}", exc_info=True)
            self.db_connector.close()
            raise
    
    async def process_mariadb_troubleshooting_data11(self, html_handler=None, vision_extractor=None) -> Dict[str, Any]:
        """
        Process unprocessed troubleshooting reports from MariaDB, grouped by error_code_id, and embed into ChromaDB.
        
        Args:
            html_handler: HTMLContentHandler instance
            vision_extractor: GraniteVisionExtractor instance
        
        Returns:
            Dict with processing results
        """
        try:
            # Use the instance's html_handler and vision_extractor if not provided
            html_handler = html_handler or self.html_handler
            vision_extractor = vision_extractor or self.vision_extractor

            if not html_handler or not vision_extractor:
                raise ValueError("HTML handler and vision extractor must be provided")

            self.db_connector.connect()
            reports_df = self.db_connector.get_unprocessed_troubleshooting_reports()
            if reports_df.empty:
                logger.info("No unprocessed troubleshooting reports found")
                return {"status": "success", "message": "No reports to process", "processed_count": 0}

            # Group by error_code_id
            grouped_reports = reports_df.groupby('error_code_id')
            results = []
            processed_count = 0
            processed_urls = set()

            for error_code_id, group in grouped_reports:
                logger.info(f"Processing reports for error_code_id: {error_code_id}")
                for _, row in group.iterrows():
                    html_content = row.get('content', '')
                    report_id = row.get('resolve_id', 'unknown')
                    url = row.get('url', None)
                    logical_nm = row.get('logical_nm', None)

                    # Use specific metadata
                    metadata={
                        "error_code_id": str(error_code_id),
                        "client_name": row.get('client_name', ''),
                        "os_version": row.get('os_version', '')
                    }

                    # Process HTML content
                    html_result = await process_html_content(
                        html_content=html_content,
                        metadata=metadata,
                        html_handler=html_handler,
                        vision_extractor=vision_extractor
                    )
                    results.append(html_result)

                    # If HTML processing succeeded, mark as processed and increment count
                    if html_result["status"] == "success":
                        processed_count += 1
                        #self.db_connector.mark_report_as_processed(error_code_id)
                        # Continue to next row if HTML content was processed successfully
                        # Note: We won't skip here to ensure attachments are processed

                    if url and logical_nm and url not in processed_urls:
                        processed_urls.add(url)  # Avoid reprocessing the same URL
                        logger.info(f"Processing attachment from URL for report {report_id}: {logical_nm}")
                        try:
                            # Download the file content
                            async with self.download_semaphore:
                                download_result = await download_file_from_url(url)
                            
                            # Handle tuple return value
                            if isinstance(download_result, tuple):
                                file_content, content_type = download_result
                            else:
                                file_content = download_result
                                content_type = None

                            if not file_content:
                                logger.warning(f"Failed to download file from {url} for report {report_id}")
                                results.append({
                                    "report_id": report_id,
                                    "logical_nm": logical_nm,
                                    "status": "error",
                                    "message": f"Failed to download file from {url}"
                                })
                                continue

                            # Determine file type and process accordingly
                            file_extension = logical_nm.lower().split('.')[-1]
                            if file_extension in ['pdf']:
                                # Use a temporary file approach since we have already initialized handlers with model_manager
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{logical_nm}") as temp_file:
                                    temp_file.write(file_content)
                                    temp_file.flush()
                                    temp_file_path = temp_file.name
                                
                                try:
                                    # Use the already initialized PDF handler
                                    text, _ = await self.pdf_handler.extract_text(temp_file_path)
                                    if text and text.strip():
                                        result = await self.process_text_content(text, str(error_code_id), metadata)
                                    else:
                                        result = {
                                            "report_id": report_id,
                                            "logical_nm": logical_nm,
                                            "status": "warning",
                                            "message": "No text extracted from PDF"
                                        }
                                finally:
                                    if os.path.exists(temp_file_path):
                                        os.unlink(temp_file_path)

                            elif file_extension in ['png', 'jpg', 'jpeg']:
                                # Use the already initialized image handler
                                text = await self.image_handler.extract_text_from_memory(file_content)
                                if text and text.strip():
                                    result = await self.process_text_content(text, str(error_code_id), metadata)
                                else:
                                    result = {
                                        "report_id": report_id,
                                        "logical_nm": logical_nm,
                                        "status": "warning",
                                        "message": "No text extracted from image"
                                    }
                            else:
                                result = {
                                    "report_id": report_id,
                                    "logical_nm": logical_nm,
                                    "status": "warning",
                                    "message": f"Unsupported file type: {file_extension}"
                                }

                            results.append(result)

                            if result["status"] == "success":
                                processed_count += 1
                                #self.db_connector.mark_report_as_processed(report_id)
                            elif result["status"] == "error":
                                logger.error(f"Failed to process attachment {logical_nm} for report {report_id}: {result['message']}")

                        except Exception as e:
                            logger.error(f"Error processing attachment for report {report_id}: {str(e)}")
                            results.append({
                                "report_id": report_id,
                                "logical_nm": logical_nm,
                                "status": "error",
                                "message": f"Error processing attachment: {str(e)}"
                            })
                    else:
                        logger.warning(f"No valid attachment URL or duplicate URL found for report {report_id}")

            self.db_connector.close()
            logger.info(f"Processed {processed_count} troubleshooting reports")
            return {
                "status": "success",
                "message": f"Processed {processed_count} out of {len(reports_df)} reports",
                "processed_count": processed_count,
                "details": results
            }

        except Exception as e:
            logger.error(f"Error in process_mariadb_troubleshooting_data: {e}", exc_info=True)
            self.db_connector.close()
            raise

    async def process_troubleshooting_report(self, logical_names: List[str], error_code_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        client_name, os_version = metadata.get("client_name"), metadata.get("os_version")
        logger.info(f"Fetching metadata for error_code_id: {error_code_id}")
        file_df = self.db_connector.get_files_by_error_code(error_code_id, logical_names)
        if file_df.empty:
            logger.warning(f"No file metadata found for error_code_id: {error_code_id} with logical_names: {logical_names}")
            return {"status": "warning", "message": f"No files or content found for error_code_id: {error_code_id}", "results": []}
        unique_contents = list(set(file_df["content"].dropna().tolist()))
        file_df = file_df.drop(columns=["content"])
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            raise RuntimeError("ChromaDB collection is not initialized")
        processed_ids = set(chromadb_collection.get()["ids"])
        content_metadata = {"error_code_id": error_code_id, "client_name": client_name, "os_version": os_version}
        for idx, content in enumerate(unique_contents):
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:10]
            content_id = f"content_{error_code_id}_{content_hash}_{idx}"
            if content_id in processed_ids:
                logger.info(f"Skipping duplicate content: {content_id}")
                continue
            image_results, _ = await self.process_embedded_images(content, error_code_id, content_metadata)
            results.extend(image_results)
            cleaned_content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
            if not cleaned_content.strip():
                logger.warning(f"Content for error_code_id {error_code_id} is empty after cleaning: {content[:100]}...")
                results.append({"content_id": f"content_{error_code_id}", "status": "error", "message": "No text content provided."})
                continue
            content_result = await self.process_text_content(cleaned_content, error_code_id, content_metadata)
            results.append(content_result)
        for _, row in file_df.iterrows():
            file_id, logical_nm, url = row.get("file_id"), row.get("logical_nm"), row.get("url")
            if not url:
                results.append({"logical_nm": logical_nm, "status": "error", "message": "No URL found for file"})
                continue
            chunk_id_base = f"{logical_nm}_chunk_0"
            if chunk_id_base in processed_ids:
                logger.info(f"Skipping duplicate file: {logical_nm}")
                continue
            if self._is_url_expired(url):
                results.append({"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL has expired"})
                continue
            logger.info(f"Downloading file: {logical_nm} from {url}")
            async with self.download_semaphore:
                download_result = await download_file_from_url(url)
            if isinstance(download_result, tuple):
                file_content, content_type = download_result
            else:
                file_content = download_result
                content_type = None
            if not file_content:
                results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file"})
                continue
            file_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": url, "error_code_id": error_code_id, "client_name": client_name, "os_version": os_version}
            file_result = await self.process_files_by_logical_names([logical_nm], error_code_id, file_metadata)
            results.extend(file_result)
        successful = sum(1 for r in results if r["status"] == "success")
        return {"status": "completed", "total_processed": len(results), "successful": successful, "failed": len(results) - successful, "error_code_id": error_code_id, "results": results}

    async def process_files_by_logical_names(self, logical_names: List[str], error_code: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200) -> List[Dict[str, Any]]:
        try:
            results = []
            file_df = self.db_connector.get_files_by_logical_names(logical_names)
            if file_df.empty:
                logger.warning(f"No files found for logical names: {logical_names}")
                return results
            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                raise RuntimeError("ChromaDB collection is not initialized")

            for _, row in file_df.iterrows():
                file_id, logical_nm, url = row.get("file_id"), row.get("logical_nm"), row.get("url")
                if not url:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No URL found for file"})
                    continue
                if self._is_url_expired(url):
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL has expired"})
                    continue

                logger.info(f"Downloading file: {logical_nm} from {url}")
                async with self.download_semaphore:  # Limit concurrent downloads
                    download_result = await download_file_from_url(url)
                if isinstance(download_result, tuple):
                    file_content, content_type = download_result
                else:
                    file_content = download_result
                    content_type = None
                if not file_content:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file"})
                    continue

                extraction_result = await process_file_content(
                    file_content=file_content,
                    filename=logical_nm,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model_manager=self.model_manager
                )
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No text extracted from the file or unsupported type"})
                    continue

                file_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": url, "error_code_id": error_code, **(metadata or {})}
                chunk_ids = []
                batch_size = 20  # Consistent with other methods

                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_ids = [f"{logical_nm}_chunk_{i + j}" for j in range(len(batch_chunks))]
                    batch_meta = [{**file_metadata, "chunk_index": i + j, "chunk_count": len(chunks)} for j in range(len(batch_chunks))]

                    async with self.embedding_semaphore:  # Limit concurrent embeddings
                        embed_tasks = [self._embed_text(chunk) for chunk in batch_chunks]
                        embeddings = await asyncio.gather(*embed_tasks)

                    async with self.chroma_lock:
                        valid_embeddings = [emb for emb in embeddings if emb is not None]
                        valid_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
                        if valid_embeddings:
                            chromadb_collection.add(
                                ids=[batch_ids[idx] for idx in valid_indices],
                                embeddings=valid_embeddings,
                                documents=[batch_chunks[idx] for idx in valid_indices],
                                metadatas=[batch_meta[idx] for idx in valid_indices]
                            )
                            chunk_ids.extend([batch_ids[idx] for idx in valid_indices])

                if chunk_ids:
                    results.append({
                        "logical_nm": logical_nm,
                        "status": "success",
                        "message": f"Processed {len(chunk_ids)} chunks",
                        "chunk_count": len(chunk_ids)
                    })

            return results
        except Exception as e:
            logger.error(f"Error processing files by logical names: {e}")
            raise

    async def process_direct_uploads_with_urls(self, resolve_data: str, file_urls: List[str]) -> Dict[str, Any]:
        try:
            # Parse resolve_data
            resolve_data_dict = json.loads(resolve_data)
            error_code_id = resolve_data_dict.get("errorCodeId", "unknown")
            client_nm = resolve_data_dict.get("clientNm", "")
            os_version_id = resolve_data_dict.get("osVersionId", "")
            content = resolve_data_dict.get("content", "")
            resolve_id = resolve_data_dict.get("resolveId", "unknown")

            metadata = {
                "error_code_nm": str(error_code_id),
                "client_name": client_nm,
                "os_version": os_version_id,
                "resolve_id": resolve_id
            }

            # Check for duplicates
            chromadb_collection = get_chromadb_collection()
            existing_ids = set(chromadb_collection.get()["ids"])
            report_id = f"resolve_{resolve_id}"
            if any(id_.startswith(report_id) for id_ in existing_ids):
                logger.info(f"Skipping duplicate report for resolve_id: {resolve_id}")
                return {
                    "status": "success",
                    "message": "Report already processed",
                    "processed_count": 0,
                    "details": []
                }

            results = []
            processed_count = 0

            # Process HTML content using the imported process_html_content function
            if content:
                html_result = await process_html_content(
                    html_content=content,
                    metadata=metadata,
                    html_handler=self.html_handler,
                    vision_extractor=self.vision_extractor
                )
                results.append(html_result)
                if html_result["status"] == "success":
                    processed_count += 1

            # Process file URLs
            for file_url in file_urls:
                logical_nm = file_url.split("/")[-1]
                logger.info(f"Processing file URL for report {resolve_id}: {logical_nm}")
                try:
                    async with self.download_semaphore:
                        download_result = await download_file_from_url(file_url)
                    
                    if isinstance(download_result, tuple):
                        file_content, content_type = download_result
                    else:
                        file_content = download_result
                        content_type = None

                    if not file_content:
                        results.append({
                            "report_id": resolve_id,
                            "logical_nm": logical_nm,
                            "status": "error",
                            "message": f"Failed to download file from {file_url}"
                        })
                        continue

                    file_metadata = {
                        "error_code_nm": str(error_code_id),
                        "client_name": client_nm,
                        "os_version": os_version_id,
                        "resolve_id": resolve_id,
                        "logical_nm": logical_nm,
                        "url": file_url
                    }
                    result = await process_file_content(
                        file_content=file_content,
                        filename=logical_nm,
                        metadata=file_metadata,
                        model_manager=self.model_manager
                    )
                    results.append(result)
                    if result["status"] == "success":
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing file URL {file_url} for report {resolve_id}: {str(e)}", exc_info=True)
                    results.append({
                        "report_id": resolve_id,
                        "logical_nm": logical_nm,
                        "status": "error",
                        "message": f"Error processing file URL: {str(e)}"
                    })

            return {
                "status": "success",
                "message": f"Processed {processed_count} items",
                "processed_count": processed_count,
                "details": results
            }
        except Exception as e:
            logger.error(f"Error in process_direct_uploads_with_urls: {e}", exc_info=True)
            raise

    def extract_content_sections(self, content: str) -> Dict[str, str]:
        """
        Extract key sections (Problem, Error Message, Solution, etc.) from the content string.
        """
        sections = {
            "problem": "",
            "error_message": "",
            "solution": "",
            "troubleshooting": "",
            "resolution": ""
        }
        
        # Simple keyword-based extraction (can be improved with NLP if needed)
        content_lower = content.lower()
        if "problem" in content_lower:
            start = content_lower.find("problem")
            end = content_lower.find("error message") if "error message" in content_lower else len(content)
            sections["problem"] = content[start:end].strip()
        
        if "error message" in content_lower:
            start = content_lower.find("error message")
            end = content_lower.find("solution") if "solution" in content_lower else len(content)
            sections["error_message"] = content[start:end].strip()
        
        if "solution" in content_lower:
            start = content_lower.find("solution")
            end = content_lower.find("troubleshooting") if "troubleshooting" in content_lower else len(content)
            sections["solution"] = content[start:end].strip()
        
        if "troubleshooting" in content_lower:
            start = content_lower.find("troubleshooting")
            end = content_lower.find("resolution") if "resolution" in content_lower else len(content)
            sections["troubleshooting"] = content[start:end].strip()
        
        if "resolution" in content_lower:
            start = content_lower.find("resolution")
            sections["resolution"] = content[start:].strip()

        return sections

    def generate_recommended_action(self, content_sections: Dict[str, str], client_name: str, error_code_nm: str) -> str:
        """
        Generate a recommended action based on the extracted content sections.
        """
        recommended_action = f"For client {client_name} with error code {error_code_nm}:\n\n"

        if content_sections["problem"]:
            recommended_action += f"**Problem Identified**: {content_sections['problem']}\n\n"
        
        if content_sections["error_message"]:
            recommended_action += f"**Error Message**: {content_sections['error_message']}\n\n"
        
        if content_sections["solution"]:
            recommended_action += f"**Solution Overview**: {content_sections['solution']}\n\n"
        
        if content_sections["troubleshooting"]:
            recommended_action += f"**Troubleshooting Steps**: {content_sections['troubleshooting']}\n\n"
        
        if content_sections["resolution"]:
            recommended_action += f"**Resolution Steps**: {content_sections['resolution']}\n\n"
        
        if not any(content_sections.values()):
            recommended_action += "No specific resolution found in the content. Please review the error code details and consult the NetBackup documentation for further assistance."

        return recommended_action

    async def process_direct_uploads_with_urls2(self, resolve_data: str, file_urls: List[str]) -> Dict[str, Any]:
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized.", "results": []}

        try:
            data = json.loads(resolve_data)
            error_code_id = str(data.get("errorCodeId", ""))
            client_name = data.get("clientNm", "")
            os_version_id = str(data.get("osVersionId", "11")) if data.get("osVersionId") is not None else "11"
            content = BeautifulSoup(data.get("content", ""), "html.parser").get_text(separator=" ", strip=True)
            os_version_name = self.os_version_map.get(os_version_id, "Unknown")
            metadata = {
                "error_code_id": error_code_id,
                "client_name": client_name,
                "os_version": os_version_name,
                "content": content
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse resolve_data: {e}")
            return {"status": "failed", "total_processed": 0, "successful": 0, "failed": 1, "results": [{"status": "error", "message": f"Invalid JSON: {e}"}]}

        # Process text content if present
        if content.strip():
            content_result = await self.process_text_content(content, error_code_id, metadata)
            results.append(content_result)

        # Process each S3 URL
        if file_urls:
            for url in file_urls:
                logical_nm = url.split("/")[-1]  # Extract filename from URL for simplicity
                if self._is_url_expired(url):
                    logger.warning(f"Skipping expired URL: {url}")
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL has expired"})
                    continue

                logger.info(f"Downloading file from URL: {url}")
                async with self.download_semaphore:  # Limit concurrent downloads
                    download_result = await download_file_from_url(url)
                if isinstance(download_result, tuple):
                    file_content, content_type = download_result
                else:
                    file_content = download_result
                    content_type = None
                if not file_content:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file from URL"})
                    continue

                file_metadata = {
                    "logical_nm": logical_nm,
                    "url": url,
                    "error_code_id": error_code_id,
                    "client_name": client_name,
                    "os_version": os_version_name
                }
                file_result = await self._process_file_content(file_content, logical_nm, error_code_id, file_metadata)
                results.append(file_result)

        successful = len([r for r in results if r["status"] == "success"])
        return {
            "status": "completed" if successful > 0 else "failed",
            "total_processed": len(file_urls) + (1 if content.strip() else 0),
            "successful": successful,
            "failed": len(results) - successful,
            "error_code_id": error_code_id,
            "results": results
        }
    
    async def process_direct_uploads(self, resolve_data: str, files: List[UploadFile]) -> Dict[str, Any]:
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized.", "results": []}

        try:
            data = json.loads(resolve_data)
            error_code_id = str(data.get("errorCodeId", ""))
            client_name = data.get("clientNm", "")
            os_version_id = str(data.get("osVersionId", "11")) if data.get("osVersionId") is not None else "11"
            content = BeautifulSoup(data.get("content", ""), "html.parser").get_text(separator=" ", strip=True)
            os_version_name = self.os_version_map.get(os_version_id, "Unknown")
            metadata = {
                "error_code_id": error_code_id,
                "client_name": client_name,
                "os_version": os_version_name,
                "content": content
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse resolve_data: {e}")
            return {"status": "failed", "total_processed": 0, "successful": 0, "failed": 1, "results": [{"status": "error", "message": f"Invalid JSON: {e}"}]}

        if content.strip():
            content_result = await self.process_text_content(content, error_code_id, metadata)
            results.append(content_result)

        if files:
            file_results = await self.process_uploaded_files_optimized(files, error_code_id, json.dumps(metadata))
            results.extend(file_results["results"])

        successful = len([r for r in results if r["status"] == "success"])
        return {
            "status": "completed" if successful > 0 else "failed",
            "total_processed": len(files) + (1 if content.strip() else 0),
            "successful": successful,
            "failed": len(results) - successful,
            "error_code_id": error_code_id,
            "results": results
        }