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
from src.core.services.file_download import download_file_from_url

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
        self.MAX_FILE_SIZE = 100 * 1024 * 1024
        self.semaphore = asyncio.Semaphore(5)
        self.chroma_lock = asyncio.Lock()
        self.model_manager = model_manager
        self.static_data_cache = static_data_cache or StaticDataCache()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        FileHandlerFactory.initialize(model_manager)

        self.handlers = {
            'pdf': FileHandlerFactory.get_handler_for_extension('pdf'),
            'image': FileHandlerFactory.get_handler_for_extension('png'),  # Also handles jpg, jpeg
            'hwp': FileHandlerFactory.get_handler_for_extension('hwp'),
            'doc': FileHandlerFactory.get_handler_for_extension('doc'),  # Also handles docx
            'msg': FileHandlerFactory.get_handler_for_extension('msg'),
        }

        # Use ModelManager for handlers
        if model_manager:
            self.pdf_handler = PDFHandler(model_manager=model_manager)
            self.image_handler = ImageHandler(model_manager=model_manager)
            self.msg_handler = MSGHandler(model_manager=model_manager)
            self.doc_handler = AdvancedDocHandler(model_manager=model_manager)
            self.hwp_handler = HWPHandler(model_manager=model_manager)
        else:
            # Fallback to handlers without model_manager
            self.pdf_handler = PDFHandler()
            self.image_handler = ImageHandler()
            self.msg_handler = MSGHandler()
            self.doc_handler = AdvancedDocHandler()
            self.hwp_handler = HWPHandler()


        # Define OS version mapping
        self.os_version_map = {
            "1": "유닉스",
            "2": "리눅스",
            "3": "유닉스부트",
            "4": "RHEL",
            "5": "CentOS",
            "6": "Unix",
            "7": "Windows",
            "8": "Solaris",
            "9": "AIX",
            "10": "HP-UX",
            "11": "모름"
        }

    def _sanitize_filename(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return f"{hashlib.md5(filename.encode('utf-8')).hexdigest()[:10]}{ext}"

    def _is_url_expired(self, url: str) -> bool:
        """
        Check if a pre-signed URL has expired based on X-Amz-Date and X-Amz-Expires.

        Args:
            url: The pre-signed URL to check.

        Returns:
            bool: True if the URL is expired, False otherwise.
        """
        from urllib.parse import urlparse, parse_qs
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
            current_time = datetime.utcnow()
            return current_time > expiration_time
        except Exception as e:
            logger.error(f"Error checking URL expiration for {url}: {str(e)}")
            return False

    async def _process_file_content(self, file_content: bytes, logical_nm: str, error_code_id: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=200) -> Dict[str, Any]:
        """
        Process raw file content, extract text, chunk it, and embed into ChromaDB.

        Args:
            file_content: Raw bytes of the file content.
            logical_nm: Logical name of the file.
            error_code_id: Error code ID associated with the file.
            metadata: Metadata to associate with the file chunks.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.

        Returns:
            Dict with processing status and details.
        """
        temp_file = None
        temp_file_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self._sanitize_filename(logical_nm))
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name

            extraction_result = process_file_content(temp_file_path, logical_nm, chunk_size, chunk_overlap)
            chunks = extraction_result.get("chunks", [])
            if not chunks:
                logger.warning(f"No chunks extracted for {logical_nm}: {extraction_result.get('message', 'Unknown reason')}")
                return {"logical_nm": logical_nm, "status": "error", "message": extraction_result.get("message", "No text extracted from the file")}

            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                return {"logical_nm": logical_nm, "status": "error", "message": "ChromaDB collection is not initialized"}

            chunk_ids = []
            async with self.chroma_lock:
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{logical_nm}_chunk_{i}"
                    if chunk_id in set(chromadb_collection.get()["ids"]):
                        logger.warning(f"Skipping duplicate embedding ID: {chunk_id}")
                        continue
                    chunk_metadata = {**metadata, "chunk_index": i, "chunk_count": len(chunks)}
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                    if embedding:
                        chromadb_collection.add(ids=[chunk_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                        chunk_ids.append(chunk_id)

            return {
                "logical_nm": logical_nm,
                "status": "success",
                "message": f"Processed {len(chunk_ids)} chunks",
                "chunk_count": len(chunk_ids)
            }
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
        
        async def process_file_task(file):
            async with self.semaphore:
                temp_file_path = None
                try:
                    if file.size > self.MAX_FILE_SIZE:
                        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds maximum size of {self.MAX_FILE_SIZE} bytes")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=self._sanitize_filename(file.filename)) as temp_file:
                        temp_file_path = temp_file.name
                        bytes_written = 0
                        while chunk := await file.read(1024 * 1024):  # 1MB chunks
                            temp_file.write(chunk)
                            bytes_written += len(chunk)
                        logger.debug(f"Wrote {bytes_written} bytes to {temp_file_path}, expected {file.size} bytes")
                        if bytes_written != file.size:
                            logger.error(f"File size mismatch for {file.filename}: wrote {bytes_written}, expected {file.size}")
                    
                    extraction_result = process_file(temp_file_path, chunk_size, chunk_overlap, model_manager=self.model_manager)
                    chunks = extraction_result.get("chunks", [])
                    if not chunks:
                        return {"filename": file.filename, "status": "error", "message": "No text was extracted from the file."}
                    
                    metadata_dict = json.loads(metadata) if metadata else {}
                    base_metadata = {"filename": file.filename, "status_code": status_code, **metadata_dict}
                    chunk_ids = []
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{file.filename}_chunk_{i}"
                        if chunk_id in set(chromadb_collection.get()["ids"]):
                            logger.warning(f"Skipping duplicate embedding ID: {chunk_id}")
                            continue
                        chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                        embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                        embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                        if embedding:
                            async with self.chroma_lock:
                                chromadb_collection.add(ids=[chunk_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                            chunk_ids.append(chunk_id)
                    return {
                        "filename": file.filename,
                        "status": "success",
                        "message": f"File processed successfully with {len(chunk_ids)} chunks.",
                        "chunk_count": len(chunk_ids)
                    }
                except HTTPException as he:
                    logger.error(f"HTTP Exception for file '{file.filename}': {he.detail}")
                    return {"filename": file.filename, "status": "error", "message": he.detail}
                except Exception as e:
                    logger.error(f"Error processing file '{file.filename}': {e}")
                    return {"filename": file.filename, "status": "error", "message": str(e)}
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                        except Exception as e:
                            logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")

        tasks = [process_file_task(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in file processing: {result}")
            elif result:
                final_results.append(result)

        successful = len([r for r in final_results if r["status"] == "success"])
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "results": final_results
        }

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
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                content_id = f"content_{status_code}_{i}"
                if content_id in set(chromadb_collection.get()["ids"]):
                    logger.warning(f"Skipping duplicate embedding ID: {content_id}")
                    continue
                chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                if embedding:
                    chromadb_collection.add(ids=[content_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                    chunk_ids.append(content_id)
            return {"content_id": f"content_{status_code}", "status": "success", "message": f"Text content processed successfully with {len(chunk_ids)} chunks.", "chunk_count": len(chunk_ids)}
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return {"content_id": f"content_{status_code}", "status": "error", "message": str(e)}

    async def process_embedded_images(self, content: str, error_code_id: str, metadata: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        results = []
        skipped_images = []  # Track skipped images
        img_urls = re.findall(r'<img[^>]+src=["\'](.*?)["\']', content)
        if not img_urls:
            return results, skipped_images
        chromadb_collection = get_chromadb_collection()
        processed_ids = set(chromadb_collection.get()["ids"]) if chromadb_collection else set()
        for url in img_urls:
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
                skipped_images.append({"url": url, "reason": "Not found in attachment_files"})  # Add to skipped images
                results.append({"url": url, "status": "error", "message": "Image not found in attachment_files"})
                continue
            for _, row in df.iterrows():
                file_id, logical_nm, file_url = row["file_id"], row["logical_nm"], row["url"]
                chunk_id_base = f"{logical_nm}_chunk_0"
                if chunk_id_base in processed_ids:
                    logger.info(f"Skipping duplicate image: {logical_nm}")
                    continue
                if self._is_url_expired(file_url):
                    logger.warning(f"Skipping expired URL for image {logical_nm}: {file_url}")
                    # Attempt to refresh URL
                    new_url = await self._refresh_presigned_url(file_url, logical_nm, row.get("physical_nm", ""))
                    if new_url:
                        file_url = new_url
                    else:
                        results.append({"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL expired and could not be refreshed"})
                        continue
                # Download and process the image
                file_content = await download_file_from_url(file_url)
                if not file_content:
                    logger.warning(f"Failed to download image from {file_url}")
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download image after retries"})
                    continue
                # Use ImageHandler to process the image content
                from src.core.file_handlers.image_handler import ImageHandler
                image_handler = ImageHandler()
                text = image_handler.extract_text_from_memory(file_content)
                if not text or not text.strip():
                    logger.warning(f"No text extracted from image {logical_nm}")
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No text extracted from image"})
                    continue
                image_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": file_url, "error_code_id": error_code_id, **metadata}
                result = await self._process_file_content(file_content, logical_nm, error_code_id, image_metadata)
                results.append(result)
        return results, skipped_images

    async def _refresh_presigned_url(self, old_url: str, logical_nm: str, physical_nm: str) -> Optional[str]:
        """Refresh an expired pre-signed URL by generating a new one using AWS S3."""
        try:
            # Get S3 configuration from environment variables
            s3_url = os.getenv("S3_URL")
            s3_access_key = os.getenv("S3_ACCESS_KEY")
            s3_secret_key = os.getenv("S3_SECRET_KEY")
            s3_bucket_name = os.getenv("S3_BUCKET_NAME")
            s3_region = os.getenv("S3_REGION")

            # Extract bucket and key from the URL
            parsed_url = urlparse(old_url)
            bucket_name = parsed_url.netloc.split('.')[0]  # Assumes bucket is first part of netloc
            key = parsed_url.path.lstrip('/')

            # Initialize S3 client with custom endpoint and credentials
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_url,  # Custom endpoint
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name=s3_region
            )

            # Generate a new pre-signed URL (valid for 1 hour)
            new_url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': s3_bucket_name, 'Key': key},
            ExpiresIn=3600
        )
            logger.info(f"Refreshed URL for {logical_nm} from {old_url} to {new_url}")
            return new_url
        except Exception as e:
            logger.error(f"Failed to refresh URL for {logical_nm}: {e}")
        return None
    
    async def process_mariadb_troubleshooting_data(self, html_handler=None, vision_extractor=None) -> Dict[str, Any]:
        """
        Process unprocessed troubleshooting reports from MariaDB, grouped by error_code_id, and embed into ChromaDB.
        
        Args:
            html_handler: HTMLContentHandler instance
            vision_extractor: GraniteVisionExtractor instance
        
        Returns:
            Dict with processing results
        """
        try:
            if not html_handler or not vision_extractor:
                raise ValueError("HTML handler and vision extractor must be provided")

            if not self.db_connector.is_connection_active():
                self.db_connector.connect()
                logger.debug("Re-established MariaDB connection for processing troubleshooting data.")
            
            reports_df = self.db_connector.get_unprocessed_troubleshooting_reports()
            if reports_df.empty:
                logger.info("No unprocessed troubleshooting reports found")
                return {"status": "success", "message": "No reports to process", "processed_count": 0}

            # Group by error_code_id
            grouped_reports = reports_df.groupby('error_code_nm')
            results = []
            processed_count = 0
            processed_urls = set()

            for error_code_nm, group in grouped_reports:
                logger.info(f"Processing reports for error_code_nm: {error_code_nm}")
                for _, row in group.iterrows():
                    html_content = row.get('content', '')
                    report_id = row.get('resolve_id', 'unknown')
                    url = row.get('url', None)
                    logical_nm = row.get('logical_nm', None)

                    # Use specific metadata
                    metadata = {
                        "error_code_nm": str(error_code_nm),
                        "client_name": row.get('client_name', ''),
                        "os_version": row.get('os_version', '')
                    }

                    # Process HTML content and embedded images
                    html_result = process_html_content(
                        html_content=html_content,
                        metadata=metadata,
                        html_handler=html_handler,
                        vision_extractor=vision_extractor
                    )
                    results.append(html_result)

                    # If HTML processing succeeded, mark as processed and move to next row
                    if html_result["status"] == "success":
                        processed_count += 1
                        #self.db_connector.mark_report_as_processed(error_code_id)
                        continue

                    if url and logical_nm and url not in processed_urls:
                        processed_urls.add(url)  # Avoid reprocessing the same URL
                        logger.info(f"Processing attachment from URL for report {report_id}: {logical_nm}")
                        try:
                            # Download the file content
                            file_content = await download_file_from_url(url)
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
                                    # Extract text (now returns a single string)
                                    text = self.pdf_handler.extract_text(temp_file_path)
                                    if text and text.strip():
                                        result = await self.process_text_content(text, str(error_code_nm), metadata)
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
                                        logger.debug(f"Cleaned up temporary file: {temp_file_path}")

                            elif file_extension in ['png', 'jpg', 'jpeg']:
                                # Use the already initialized image handler
                                text = self.image_handler.extract_text_from_memory(file_content)
                                if text and text.strip():
                                    result = await self.process_text_content(text, str(error_code_nm), metadata)
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
                            logger.error(f"Error processing attachment for report {report_id}: {str(e)}", exc_info=True)
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

            # Update global state after ingestion
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

            set_globals(
                chroma_coll=chroma_coll,
                rag=get_rag_chain(),
                vect_store=vector_store,  # Use Chroma object instead of Collection
                prompt=get_global_prompt(),
                workflow=get_workflow(),
                memory=get_memory()
            )
            logger.debug("Updated global state after ingestion with Chroma vector store")



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
                        "error_code_id":str(error_code_id),
                        "client_name":row.get('client_name',''),
                        "os_version":row.get('os_version','')
                    }

                    # Process HTML content and embedded images
                    html_result = process_html_content(
                        html_content=html_content,
                        metadata=metadata,
                        html_handler=html_handler,
                        vision_extractor=vision_extractor
                    )
                    results.append(html_result)

                    # If HTML processing succeeded, mark as processed and move to next row
                    if html_result["status"] == "success":
                        processed_count += 1
                        #self.db_connector.mark_report_as_processed(error_code_id)
                        continue

                    if url and logical_nm and url not in processed_urls:
                        processed_urls.add(url)  # Avoid reprocessing the same URL
                        logger.info(f"Processing attachment from URL for report {report_id}: {logical_nm}")
                        try:
                            # Download the file content
                            file_content = await download_file_from_url(url)
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
                                    text, _ = self.pdf_handler.extract_text(temp_file_path)
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
                                text = self.image_handler.extract_text_from_memory(file_content)
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
            image_results = await self.process_embedded_images(content, error_code_id, content_metadata)
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
            file_content = await download_file_from_url(url)
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
                file_content = await download_file_from_url(url)
                if not file_content:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file"})
                    continue
                #extraction_result = process_file_content(file_content, logical_nm, chunk_size, chunk_overlap)
                extraction_result = process_file_content(
                    file_content=file_content,
                    filename=logical_nm,
                    metadata=file_metadata,  # Make sure you're passing metadata here
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model_manager=self.model_manager  # Add this line
                    )
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No text extracted from the file or unsupported type"})
                    continue
                file_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": url, "error_code_id": error_code, **(metadata or {})}
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{logical_nm}_chunk_{i}"
                    if chunk_id in set(chromadb_collection.get()["ids"]):
                        logger.warning(f"Skipping duplicate embedding ID: {chunk_id}")
                        continue
                    chunk_metadata = {**file_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                    if embedding:
                        chromadb_collection.add(ids=[chunk_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                        chunk_ids.append(chunk_id)
                results.append({"logical_nm": logical_nm, "status": "success", "message": f"Processed {len(chunk_ids)} chunks", "chunk_count": len(chunk_ids)})
            return results
        except Exception as e:
            logger.error(f"Error processing files by logical names: {e}")
            raise

    
    async def process_direct_uploads_with_urls(self, resolve_data: str, file_urls: List[str]) -> Dict[str, Any]:
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized.", "results": []}

        try:
            # Log the raw resolve_data for debugging
            logger.debug(f"Received resolve_data: {resolve_data}")

            # Parse the resolve_data JSON
            data = json.loads(resolve_data)
            error_code_nm = str(data.get("errorCodeNm", ""))
            client_name = data.get("clientNm", "")

            # Handle osVersionId, allowing for null values
            os_version_id_raw = data.get("osVersionId")
            if os_version_id_raw is None:
                os_version_id = "11"  # Default to "11" if osVersionId is null or missing
                logger.debug("osVersionId is null or missing, defaulting to '11'")
            else:
                os_version_id = str(os_version_id_raw)

            # Validate os_version_id against os_version_map
            os_version_name = self.os_version_map.get(os_version_id, "Unknown")
            if os_version_name == "Unknown" and os_version_id != "11":
                logger.warning(f"Invalid osVersionId '{os_version_id}' provided, defaulting os_version_name to 'Unknown'")

            # Use the raw content directly for recommended_action
            content = data.get("content", "")
            resolve_id = str(data.get("resolveId", ""))

            # Retrieve static error code information
            error_code_info = self.static_data_cache.get_error_code_info(error_code_nm)
            if not error_code_info:
                logger.error(f"Error code NM {error_code_nm} not found in static cache.")
                return {"status": "failed", "total_processed": 0, "successful": 0, "failed": 1, "results": [{"status": "error", "message": f"Error code NM {error_code_nm} not found"}]}

            # Log the retrieved error_code_info for debugging
            logger.debug(f"Retrieved error_code_info for error_code_nm {error_code_nm}: {error_code_info}")

            # Create metadata dictionary to store alongside the content
            metadata = {
                "error_code_id": error_code_info["error_code_id"],
                "error_code_nm": error_code_nm,
                "explanation_en": error_code_info["explanation_en"],
                "message_en": error_code_info["message_en"],
                "recom_action_en": error_code_info["recom_action_en"],
                "client_name": client_name,
                "os_version": os_version_name,
                "content": content,
                "resolve_id": resolve_id
            }

            # Process the content (for indexing or other purposes, but not for recommended_action)
            cleaned_content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
            if cleaned_content.strip():
                content_result = await self.process_text_content(cleaned_content, error_code_nm, metadata)
                results.append(content_result)
            else:
                results.append({"status": "error", "message": "No content provided to process"})

            if file_urls:
                for url in file_urls:
                    logical_nm = url.split("/")[-1]
                    if self._is_url_expired(url):
                        logger.warning(f"Skipping expired URL: {url}")
                        results.append({"logical_nm": logical_nm, "status": "error", "message": "Pre-signed URL has expired"})
                        continue

                    logger.info(f"Downloading file from URL: {url}")
                    download_result = await download_file_from_url(url)
                    if download_result is None:
                        results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file from URL"})
                        continue

                    file_content, content_type = download_result  # Unpack only if download_result is not None

                    file_type = get_file_type(file_content, content_type)
                    if not file_type:
                        results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to identify file type"})
                        continue

                    file_metadata = {
                        "logical_nm": logical_nm,
                        "url": url,
                        "error_code_id": error_code_info["error_code_id"],
                        "error_code_nm": error_code_nm,
                        "explanation_en": error_code_info["explanation_en"],
                        "message_en": error_code_info["message_en"],
                        "recom_action_en": error_code_info["recom_action_en"],
                        "client_name": client_name,
                        "os_version": os_version_name,
                        "resolve_id": resolve_id,
                        "file_type": file_type
                    }

                    # Process the file based on its type
                    handler = self.handlers.get(file_type)
                    if not handler:
                        result = await self._process_file_content(file_content, logical_nm, error_code_nm, file_metadata)
                        results.append(result)
                        continue

                    if file_type == "image":
                        text = handler.extract_text_from_memory(file_content)
                        if text and text.strip():
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from image"}
                        results.append(result)
                    elif file_type == "pdf":
                        text = handler.extract_text_from_memory(file_content)
                        logger.debug(f"Extracted text from PDF: {text!r}")
                        # Remove any non-printable characters and normalize whitespace
                        if text:
                            text = "".join(c for c in text if c.isprintable() or c.isspace())
                            text = " ".join(text.split())  # Normalize whitespace
                        if text and text.strip():
                            logger.info(f"Successfully extracted text from PDF: {logical_nm}, length: {len(text)}")
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            logger.warning(f"No text extracted from PDF: {logical_nm}")
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from PDF"}
                        results.append(result)
                    elif file_type == "hwp":
                        text = handler.extract_text_from_memory(file_content)
                        if text and text.strip():
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from HWP"}
                        results.append(result)
                    elif file_type == "doc":
                        text = handler.extract_text_from_memory(file_content)
                        if text and text.strip():
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from DOC/DOCX"}
                        results.append(result)
                    elif file_type == "msg":
                        text = handler.extract_text_from_memory(file_content)
                        if text and text.strip():
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from MSG"}
                        results.append(result)
                    elif file_type == "text":
                        text = file_content.decode('utf-8', errors='ignore')
                        if text.strip():
                            result = await self.process_text_content(text, error_code_nm, file_metadata)
                        else:
                            result = {"logical_nm": logical_nm, "status": "error", "message": "No text extracted from text file"}
                        results.append(result)
                    else:
                        result = await self._process_file_content(file_content, logical_nm, error_code_nm, file_metadata)
                        results.append(result)

            # Add to knowledge graph, ensuring metadata is stored for chat-based answering
            await asyncio.to_thread(self.knowledge_graph.add_resolve, data, file_urls)

            # Log the data being added to the knowledge graph for debugging
            logger.debug(f"Adding to knowledge graph: resolve_data={data}, file_urls={file_urls}")

            successful = len([r for r in results if r["status"] == "success"])

            # Use the raw content directly as recommended_action, without any client-specific prefix
            recommended_action = content

            # Construct the response
            response = {
                "status": "completed" if successful > 0 else "failed",
                "total_processed": len(file_urls) + (1 if cleaned_content.strip() else 0),
                "successful": successful,
                "failed": len(results) - successful,
                "error_code_id": error_code_info["error_code_id"],
                "error_code_nm": error_code_nm,
                "client_name": client_name,  # Include client_name for context
                "explanation_en": error_code_info["explanation_en"],
                "message_en": error_code_info["message_en"],
                "recommended_action": recommended_action,
                "results": results
            }

            return response

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse resolve_data: {e}. Received data: {resolve_data}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in resolve_data: {str(e)}. Received data: {resolve_data}")
        except Exception as e:
            logger.error(f"Error in process_direct_uploads_with_urls: {e}")
            raise
        
    ##New methods:def extract_content_sections(self, content: str) -> Dict[str, str]:
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
                file_content = await download_file_from_url(url)
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