import os
import tempfile
import json
import ollama
import hashlib
import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import List, Optional, Dict, Any, Tuple
from src.core.services.file_utils import (process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection, process_file_content, CHROMA_DIR, process_html_content, get_personal_vector_store, get_vector_store)
from src.core.services.file_download import download_file_from_url
from src.core.services.file_utils import (CHROMA_DIR, set_globals, get_vector_store, get_rag_chain, get_global_prompt, get_workflow, get_memory)
from src.core.mariadb_db.mariadb_connector import MariaDBConnector
from src.core.postgresqldb_db.postgresql_connector import PostgreSQLConnector
import logging
from bs4 import BeautifulSoup
import re
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
import boto3
import datetime
import math
from langchain_ollama import OllamaEmbeddings

from src.core.file_handlers.factory import FileHandlerFactory
from src.core.services.static_data_cache import StaticDataCache
from src.core.services.knowledge_graph import KnowledgeGraph

from src.core.file_handlers.pdf_handler import PDFHandler
from src.core.file_handlers.hwp_handler import HWPHandler
from src.core.file_handlers.doc_handler import AdvancedDocHandler
from src.core.file_handlers.msg_handler import MSGHandler
from src.core.file_handlers.image_handler import ImageHandler
from src.core.file_handlers.excel_handler import ExcelHandler  # New import
from src.core.file_handlers.pptx_handler import PPTXHandler  # New import
from src.core.file_handlers.txt_handler import TXTHandler
from src.core.file_handlers.rtf_handler import RTFHandler
from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor  # Updated import
from src.core.services.static_data_cache import static_data_cache
from src.core.utils.file_identification import get_file_type
from src.core.services.knowledge_graph import knowledge_graph

from src.core.services.query_service import QueryService

load_dotenv()

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self, db_connector: None = None, model_manager=None):
        self.sample_data_dir = os.path.join(os.getcwd(), "sample_data")
        os.makedirs(self.sample_data_dir, exist_ok=True)
        
        self.db_connector = db_connector
        self._valid_document_types_cache = None
        self._cache_expiry = None
        self.use_postgresql = (hasattr(db_connector, 'execute_query') and isinstance(db_connector, PostgreSQLConnector))

        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
        self.semaphore = asyncio.Semaphore(5)  # General concurrency limit
        self.chroma_lock = asyncio.Lock()
        self.model_manager = model_manager
        self.static_data_cache = static_data_cache or StaticDataCache()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.personal_vector_store = get_personal_vector_store()  # ▲ NEW
        self.kb_vector_store       = get_vector_store()
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
            'excel': FileHandlerFactory.get_handler_for_extension('xlsx'),  # Added excel
            'pptx': FileHandlerFactory.get_handler_for_extension('pptx'),  # Added pptx
            'txt':FileHandlerFactory.get_handler_for_extension('txt'),
            'rtf':FileHandlerFactory.get_handler_for_extension('rtf'),
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
            self.excel_handler = ExcelHandler(model_manager=model_manager)  # New handler
            self.pptx_handler = PPTXHandler(model_manager=model_manager)
            self.txt_handler = TXTHandler(model_manager=model_manager)
            self.rtf_handler = RTFHandler(model_manager=model_manager)
            
        else:
            self.pdf_handler = PDFHandler()
            self.image_handler = ImageHandler()
            self.msg_handler = MSGHandler()
            self.doc_handler = AdvancedDocHandler()
            self.hwp_handler = HWPHandler()
            self.html_handler = HTMLContentHandler()
            self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
            self.excel_handler = ExcelHandler()  # New handler
            self.pptx_handler = PPTXHandler()  # New handler
            self.txt_handler = TXTHandler()
            self.rtf_handler = RTFHandler()
            



        self.os_version_map = {
            "1": "유닉스", "2": "리눅스", "3": "유닉스부트", "4": "RHEL", "5": "CentOS",
            "6": "Unix", "7": "Windows", "8": "Solaris", "9": "AIX", "10": "HP-UX", "11": "모름"
        }



        # Load valid document types from environment variable
        valid_document_types_str = os.getenv("VALID_DOCUMENT_TYPES", "troubleshooting,contract,memo,wbs,rnr,proposal,presentation")
        self.valid_document_types = set(valid_document_types_str.split(",")) if valid_document_types_str else set()
        if not self.valid_document_types:
            logger.warning("No valid document types specified in VALID_DOCUMENT_TYPES environment variable. Using default set.")
            self.valid_document_types = {"troubleshooting", "contract", "memo", "wbs", "rnr", "proposal", "presentation"}
        logger.info(f"Loaded valid document types: {self.valid_document_types}")

    async def get_valid_document_types(self) -> List[str]:
        """
        Fetch valid document types from PostgreSQL category table.
        Implements caching to avoid frequent DB calls.
        """
        import time
        
        # Check cache first (cache for 1 hour)
        now = time.time()
        if (self._valid_document_types_cache and 
            self._cache_expiry and 
            now < self._cache_expiry):
            return self._valid_document_types_cache
        
        try:
            if self.db_connector:
                # Query the category table
                query = "SELECT DISTINCT category_nm FROM category WHERE category_nm IS NOT NULL"
                result = self.db_connector.fetch_all(query)
                
                # Extract category names
                valid_types = [row['category_nm'] for row in result if row['category_nm']]
                
                # Update cache
                self._valid_document_types_cache = valid_types
                self._cache_expiry = now + 3600  # Cache for 1 hour
                
                logger.info(f"Loaded {len(valid_types)} document types from database: {valid_types}")
                return valid_types
            else:
                # Fallback to environment variable
                env_types = os.getenv("VALID_DOCUMENT_TYPES", "troubleshooting,contract,memo,wbs,rnr,proposal,presentation")
                fallback_types = [t.strip() for t in env_types.split(",")]
                logger.warning("No database connector available, using fallback document types")
                return fallback_types
                
        except Exception as e:
            logger.error(f"Error fetching document types from database: {e}")
            # Fallback to environment variable
            env_types = os.getenv("VALID_DOCUMENT_TYPES", "troubleshooting,contract,memo,wbs,rnr,proposal,presentation")
            fallback_types = [t.strip() for t in env_types.split(",")]
            logger.warning(f"Using fallback document types due to error: {fallback_types}")
            return fallback_types

    
    def _sanitize_filename(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return f"{hashlib.md5(filename.encode('utf-8')).hexdigest()[:10]}{ext}"

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace, invalid characters, and metadata-like strings."""
        if not text:
            return ""
        # Replace problematic Unicode characters
        text = text.replace('—', '-').replace('•', '*')
        # Remove metadata-like headers
        if text.startswith("=== Page") or text.startswith("Image "):
            return ""
        # Remove excessive newlines and spaces
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
        return text
    
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

    async def _embed_text(self, text: str, metadata: Dict = None) -> Tuple[Optional[List[float]], Dict]:
        """Generate embeddings for text with retry logic and better error handling."""
        if metadata is None:
            metadata = {}
            
        # Enhanced text cleaning with length validation
        cleaned_text = self._clean_text_for_embedding(text, aggressive=False)
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            logger.warning(f"Text too short or empty after cleaning: {cleaned_text[:100]}...")
            return None, metadata

        try:
            # Retry logic with backoff
            max_attempts = 3
            backoff_factor = 0.5
            
            for attempt in range(max_attempts):
                try:
                    # Verify Ollama service is available
                    try:
                        await asyncio.to_thread(ollama.list)  # Basic health check
                    except Exception as e:
                        logger.error(f"Ollama service unavailable: {str(e)}")
                        raise HTTPException(
                            status_code=503,
                            detail="Embedding service unavailable"
                        )

                    # Generate embedding with timeout
                    embed_response = await asyncio.wait_for(
                        asyncio.to_thread(
                            ollama.embed,
                            model="mxbai-embed-large",
                            input=cleaned_text,
                            options={"temperature": 0.0}
                        ),
                        timeout=30.0
                    )
                    
                    logger.debug(f"Raw embedding response type: {type(embed_response)}")
                    logger.debug(f"Raw embedding response keys: {embed_response.keys() if hasattr(embed_response, 'keys') else 'No keys'}")
                    
                    # FIXED: Handle new Ollama response format correctly
                    embedding = None
                    if hasattr(embed_response, 'embeddings'):
                        # New format: response is an object with embeddings attribute
                        embeddings = embed_response.embeddings
                        if embeddings and len(embeddings) > 0:
                            embedding = embeddings[0]  # Get first (and usually only) embedding
                    elif isinstance(embed_response, dict):
                        # Handle dict format
                        if 'embeddings' in embed_response:
                            embeddings = embed_response['embeddings']
                            if embeddings and len(embeddings) > 0:
                                embedding = embeddings[0]  # Get first embedding from list
                        elif 'embedding' in embed_response:
                            # Fallback to old format
                            embedding = embed_response['embedding']
                    
                    if embedding is None:
                        logger.error(f"No embedding found in response: {embed_response}")
                        if attempt == max_attempts - 1:
                            return None, metadata
                        continue
                    
                    # Validate embedding dimensions
                    if not isinstance(embedding, list):
                        logger.error(f"Embedding is not a list: {type(embedding)}")
                        if attempt == max_attempts - 1:
                            return None, metadata
                        continue
                        
                    if len(embedding) < 100:  # Reasonable minimum for embedding dimensions
                        logger.error(f"Invalid embedding dimensions: {len(embedding)}")
                        if attempt == max_attempts - 1:
                            return None, metadata
                        continue
                    
                    logger.debug(f"Successfully generated embedding with {len(embedding)} dimensions")
                    
                    # Normalize embedding
                    import numpy as np
                    embedding_array = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    return embedding_array.tolist(), metadata
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Embedding timeout on attempt {attempt + 1}")
                    if attempt == max_attempts - 1:
                        return None, metadata
                        
                except Exception as e:
                    logger.error(f"Embedding error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_attempts - 1:
                        return None, metadata
                    
                # Exponential backoff
                await asyncio.sleep(backoff_factor * (attempt + 1))
                
            return None, metadata
            
        except Exception as e:
            logger.error(f"Unexpected embedding error: {str(e)}")
            return None, metadata


    def _clean_text_for_embedding(self, text: str, aggressive: bool = False) -> str:
        """Clean text by removing problematic characters and formatting that might cause embedding issues.
        
        Args:
            text: The text to clean
            aggressive: Whether to apply more aggressive cleaning (for retry attempts)
        
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
            
        # Basic cleaning (always applied)
        # Replace problematic Unicode characters
        text = text.replace('—', '-').replace('•', '*').replace('…', '...').replace('"', '"').replace('"', '"')
        
        # Remove PDF-specific metadata markers
        text = re.sub(r'===\s*Page\s+\d+\s*===', ' ', text)
        text = re.sub(r'===\s*IMAGE TEXTS\s*===', ' ', text)
        text = re.sub(r'Image\s+\d+:', ' ', text)
        
        # Remove non-printable characters except newlines
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        
        # Convert multiple newlines to a single space
        text = re.sub(r'\n+', ' ', text)
        
        # Remove excessive whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text.strip())
        
        if aggressive:
            # More aggressive cleaning for retry attempts
            # Remove any technical command-line formats that often cause issues
            text = re.sub(r'-\w+\s+[\w\-_]+', ' ', text)  # Command-line options like -h hostname
            text = re.sub(r'\[\w+[\]\:]', ' ', text)      # Brackets with content like [option]
            text = re.sub(r'\/\w+', ' ', text)            # File paths or URIs like /path/file
            text = re.sub(r'\|\s*\w+', ' ', text)         # Pipe characters with commands
            text = re.sub(r'\w+\.\w+\.\w+', ' ', text)    # Remove complex IDs or version numbers
            
            # Remove parenthesized content which often causes issues
            text = re.sub(r'\([^)]*\)', ' ', text)
            
            # Remove common format specifications
            text = re.sub(r'mm\/dd\/yyyy', 'date', text)  # Date formats
            text = re.sub(r'hh\:mm\:ss', 'time', text)    # Time formats
            
            # Remove any remaining special characters that might cause issues
            text = re.sub(r'[^\w\s\.,;:\?\!]', ' ', text)
            
            # Normalize whitespace again after aggressive cleaning
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Limit length for very long texts to ensure quality
            if len(text) > 6000:
                text = text[:6000]
        
        return text.strip()


    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure metadata values are valid types for ChromaDB (str, int, float, bool).
        Replace None values with empty strings.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ",".join(str(item) for item in value)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    async def process_uploaded_files_optimized(self, files: List[UploadFile], status_code: str, metadata: Optional[str] = None, *, scope: str = "kb", chunk_size=1000, chunk_overlap=200, model_manager=None):
        # Set the target vector store based on scope
        target_store = self.personal_vector_store if scope == "chat" else self.kb_vector_store
        results = []
        
        # Use the appropriate ChromaDB collection based on scope
        if scope == "chat":
            # Get or create personal vector store for chat-specific uploads
            personal_store = get_personal_vector_store()
            if hasattr(personal_store, '_collection'):
                chromadb_collection = personal_store._collection
            else:
                logger.error("Personal vector store does not have ._collection attribute")
                # Fallback, but log warning
                chromadb_collection = get_chromadb_collection()
                logger.warning("Fallback to main collection for chat scope - this is likely incorrect!")
        else:
            # In KB scope, use the main KB collection
            chromadb_collection = get_chromadb_collection()

        temp_files = []
        try:
            all_chunks = []
            chunk_metadata = []
            chunk_ids = []

            for file in files:
                if not file.filename:
                    results.append({"filename": "unknown", "status": "error", "message": "File must have a valid filename"})
                    continue

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

                extraction_result = await process_file(
                    temp_file_path,
                    chunk_size,
                    chunk_overlap,
                    filename=file.filename,
                    model_manager=self.model_manager or model_manager
                )
                logger.debug(f"Extraction result for {file.filename}: {extraction_result}")

                chunks = extraction_result.get("chunks", [])
                status = extraction_result.get("status", "error")
                message = extraction_result.get("message", "Unknown error")

                if status != "success" or not chunks:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": message or "No text was extracted from the file.",
                        "chunk_count": 0,
                        "id": None
                    })
                    continue

                # Parse and sanitize metadata
                metadata_dict = json.loads(metadata) if metadata else {}
                # Add scope to metadata to track where chunks belong
                metadata_dict["scope"] = scope
                base_metadata = {"filename": file.filename, "status_code": status_code, **metadata_dict}
                # Sanitize the metadata for ChromaDB compatibility
                base_metadata = self.sanitize_metadata(base_metadata)

                # Check for existing IDs (this check needs to be adapted based on collection type)
                existing_ids = []
                try:
                    if hasattr(chromadb_collection, 'get'):
                        # Native ChromaDB collection
                        existing_ids = chromadb_collection.get()["ids"]
                    elif hasattr(target_store, 'get'):
                        # Langchain Chroma
                        existing_ids = target_store.get()["ids"]
                    elif hasattr(target_store, '_collection'):
                        # Access underlying collection
                        existing_ids = target_store._collection.get()["ids"]
                except Exception as e:
                    logger.warning(f"Failed to get existing IDs: {e}")
                    existing_ids = []
                
                existing_ids_set = set(existing_ids)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file.filename}_chunk_{i}"
                    
                    if chunk_id in existing_ids_set:
                        logger.warning(f"Skipping duplicate embedding ID: {chunk_id}")
                        continue
                    
                    all_chunks.append(chunk)
                    chunk_metadata.append({**base_metadata, "chunk_index": i, "chunk_count": len(chunks)})
                    chunk_ids.append(chunk_id)

                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"File processed successfully with {len(chunks)} chunks.",
                    "chunk_count": len(chunks),
                    "id": file.filename
                })

            if all_chunks:
                batch_size = 20
                processed_chunks = 0
                logger.info(f"Beginning embedding process for {len(all_chunks)} chunks...")
                
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]
                    batch_meta = chunk_metadata[i:i + batch_size]

                    logger.debug(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                    
                    async with self.embedding_semaphore:
                        embed_tasks = [self._embed_text(chunk, batch_meta[idx]) for idx, chunk in enumerate(batch_chunks)]
                        embeddings_results = await asyncio.gather(*embed_tasks)
                        
                        # Log the success rate of embedding generation
                        successful_embeds = [result for result in embeddings_results if result[0] is not None]
                        logger.info(f"Batch {i//batch_size + 1}: Generated {len(successful_embeds)}/{len(batch_chunks)} embeddings successfully")
                        
                        embeddings = [result[0] for result in embeddings_results if result[0] is not None]
                        valid_indices = [idx for idx, result in enumerate(embeddings_results) if result[0] is not None]
                        
                        # Check if we have any valid embeddings to store
                        if not embeddings:
                            logger.warning(f"No valid embeddings in batch {i//batch_size + 1}, skipping ChromaDB storage")
                            continue

                    # Use a separate try/except block for ChromaDB operations
                    try:
                        async with self.chroma_lock:
                            # Ensure all metadata values are properly sanitized before adding to ChromaDB
                            sanitized_batch_meta = [self.sanitize_metadata(meta) for meta in [batch_meta[idx] for idx in valid_indices]]
                            
                            # Add the embeddings to the appropriate collection based on its type
                            logger.info(f"Adding {len(embeddings)} embeddings to ChromaDB ({scope} scope)...")
                            
                            # Check what type of collection we have and use the appropriate method
                            if hasattr(chromadb_collection, 'add'):
                                # Native ChromaDB collection
                                chromadb_collection.add(
                                    ids=[batch_ids[idx] for idx in valid_indices],
                                    embeddings=embeddings,
                                    documents=[batch_chunks[idx] for idx in valid_indices],
                                    metadatas=sanitized_batch_meta
                                )
                            elif hasattr(target_store, 'add_embeddings'):
                                # Langchain Chroma with add_embeddings method
                                target_store.add_embeddings(
                                    texts=[batch_chunks[idx] for idx in valid_indices],
                                    embeddings=embeddings,
                                    metadatas=sanitized_batch_meta,
                                    ids=[batch_ids[idx] for idx in valid_indices]
                                )
                            elif hasattr(target_store, 'add_texts'):
                                # Fallback to add_texts which will re-embed, but at least it will work
                                target_store.add_texts(
                                    texts=[batch_chunks[idx] for idx in valid_indices],
                                    metadatas=sanitized_batch_meta,
                                    ids=[batch_ids[idx] for idx in valid_indices]
                                )
                            else:
                                logger.error(f"Unknown collection type, can't add embeddings")
                                raise ValueError("Unknown collection type")
                                
                            processed_chunks += len(embeddings)
                            logger.info(f"Successfully stored batch in ChromaDB, total stored: {processed_chunks}")
                    except Exception as chroma_error:
                        logger.error(f"ChromaDB storage error: {chroma_error}", exc_info=True)

                    # Perform a final verification to confirm data was actually stored
                    try:
                        final_count = 0
                        if hasattr(chromadb_collection, 'get'):
                            final_count = len(chromadb_collection.get()["ids"])
                        elif hasattr(target_store, 'get'):
                            final_count = len(target_store.get()["ids"])
                        elif hasattr(target_store, '_collection'):
                            final_count = len(target_store._collection.get()["ids"])
                        
                        logger.info(f"Final ChromaDB collection count: {final_count} entries")
                    except Exception as e:
                        logger.error(f"Failed to verify final ChromaDB status: {e}")

                successful = len([r for r in results if r["status"] == "success"])
                return {
                    "status": "completed",
                    "total_files": len(files),
                    "successful": successful,
                    "failed": len(files) - successful,
                    "results": results,
                    "chunks_stored": processed_chunks
                }
            
            else:
                return {
                    "status": "warning",
                    "message": "No chunks were extracted from the files.",
                    "total_files": len(files),
                    "successful": 0,
                    "failed": len(files),
                    "results": results
                }
        except Exception as e:
            logger.error(f"Error in process_uploaded_files_optimized: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "results": results}
        finally:
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")

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
        """
        Process document data and S3 file URLs with enhanced multi-file support.
        Each file gets its own logical_nm in metadata.
        """
        try:
            logger.info(f"Raw resolve_data: {resolve_data}")
            
            # Parse resolve_data
            try:
                resolve_data_dict = json.loads(resolve_data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in resolve_data: {e}")
                return {
                    "status": "error",
                    "message": f"Invalid JSON in resolve_data: {str(e)}",
                    "processed_count": 0,
                    "details": []
                }

            # Extract core fields
            document_id = resolve_data_dict.get("document_id", "unknown")
            document_type = resolve_data_dict.get("document_type", "unknown")
            tags = resolve_data_dict.get("tags", [])
            content = resolve_data_dict.get("content", "")
            custom_metadata = resolve_data_dict.get("custom_metadata", {})
            member_id = custom_metadata.get("member_id", "unknown")
            logical_names = custom_metadata.get("logical_names", [])

            # Validate required fields
            if not document_id or document_id == "unknown":
                logger.error("Missing or invalid document_id")
                return {
                    "status": "error",
                    "message": "Missing or invalid document_id",
                    "processed_count": 0,
                    "details": []
                }

            if not document_type or document_type == "unknown":
                logger.error("Missing or invalid document_type")
                return {
                    "status": "error",
                    "message": "Missing or invalid document_type",
                    "processed_count": 0,
                    "details": []
                }

            # Validate document_type against database
            valid_types = await self.get_valid_document_types()
            if document_type not in valid_types:
                logger.error(f"Invalid document_type: {document_type}")
                return {
                    "status": "error",
                    "message": f"Invalid document_type: {document_type}. Must be one of {valid_types}",
                    "processed_count": 0,
                    "details": []
                }

            # Convert lists/complex structures to strings for ChromaDB compatibility
            tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)

            # Process custom_metadata to ensure ChromaDB compatibility
            processed_custom_metadata = {}
            for key, value in custom_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    processed_custom_metadata[key] = value
                elif isinstance(value, list):
                    processed_custom_metadata[key] = ",".join(str(item) for item in value)
                elif isinstance(value, dict):
                    processed_custom_metadata[key] = json.dumps(value)
                else:
                    processed_custom_metadata[key] = str(value)

            # Build base metadata (without file-specific fields)
            base_metadata = {
                "document_id": document_id,
                "document_type": document_type,
                "tags_str": tags_str,
                "source": f"{document_type}_documents",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "member_id": member_id,
                "uploaded_by": member_id,
            }
            # Add processed custom metadata
            base_metadata.update(processed_custom_metadata)

            # Check for duplicates
            chromadb_collection = get_chromadb_collection()
            existing_ids = set(chromadb_collection.get()["ids"])
            report_id = f"{document_type}_{document_id}_{member_id}"
            
            if any(id_.startswith(report_id) for id_ in existing_ids):
                logger.info(f"Skipping duplicate document for {document_type} with ID {document_id}")
                return {
                    "status": "success",
                    "message": "Document already processed",
                    "processed_count": 0,
                    "details": []
                }

            results = []
            processed_count = 0
            all_chunks = []
            chunk_metadata = []
            chunk_ids = []

            # Process HTML content (if any)
            if content:
                html_result = await process_html_content(
                    html_content=content,
                    metadata=base_metadata,
                    html_handler=self.html_handler,
                    vision_extractor=self.vision_extractor
                )
                results.append(html_result)
                if html_result["status"] == "success":
                    processed_count += 1

            # Process file URLs with individual logical_nm tracking
            if file_urls:
                if logical_names and len(logical_names) == len(file_urls):
                    # NEW: Use provided logical names (from backend)
                    all_logical_names = logical_names
                    logger.info(f"Using backend-provided logical_names: {all_logical_names}")
                else:
                    # FALLBACK: Extract from URLs (for chat uploads & backward compatibility)
                    all_logical_names = [file_url.split("/")[-1] for file_url in file_urls]
                    logger.info(f"Extracting logical_names from URLs: {all_logical_names}")
                
                # Add file list to base metadata for reference
                base_metadata["file_count"] = len(file_urls)
                base_metadata["all_files"] = ",".join(all_logical_names)
                
                for file_index, file_url in enumerate(file_urls):
                    logical_nm = all_logical_names[file_index]
                    logger.info(f"Processing file {file_index + 1}/{len(file_urls)} for document {document_id}: {logical_nm}")
                    
                    try:
                        # Check if URL is expired
                        if self._is_url_expired(file_url):
                            logger.warning(f"Skipping expired URL: {file_url}")
                            results.append({
                                "document_id": document_id,
                                "logical_nm": logical_nm,
                                "file_index": file_index,
                                "status": "error",
                                "message": "Pre-signed URL has expired"
                            })
                            continue

                        # Download file content
                        async with self.download_semaphore:
                            download_result = await download_file_from_url(file_url)

                        if isinstance(download_result, tuple):
                            file_content, content_type = download_result
                        else:
                            file_content = download_result
                            content_type = None

                        if not file_content:
                            results.append({
                                "document_id": document_id,
                                "logical_nm": logical_nm,
                                "file_index": file_index,
                                "status": "error",
                                "message": f"Failed to download file from {file_url}"
                            })
                            continue

                        # Build file-specific metadata
                        file_metadata = base_metadata.copy()
                        file_metadata["logical_nm"] = logical_nm
                        file_metadata["url"] = file_url
                        file_metadata["file_index"] = file_index  # Track which file this is
                        file_metadata["is_multi_file"] = len(file_urls) > 1
                        
                        # Add content type if available
                        if content_type:
                            file_metadata["content_type"] = content_type

                        # Process file content
                        result = await process_file_content(
                            file_content=file_content,
                            filename=logical_nm,
                            metadata=file_metadata,
                            model_manager=self.model_manager
                        )
                        
                        # Add file_index to result for tracking
                        result["file_index"] = file_index
                        result["logical_nm"] = logical_nm
                        results.append(result)
                        
                        # Only proceed if processing was successful
                        if result["status"] == "success":
                            processed_count += 1
                            
                            # Extract chunks for vector storage
                            extracted_chunks = result.get("chunks", [])
                            if extracted_chunks:
                                for i, chunk in enumerate(extracted_chunks):
                                    # Include file_index in chunk_id for uniqueness
                                    chunk_id = f"{document_type}_{document_id}_{member_id}_file{file_index}_{logical_nm}_chunk_{i}"
                                    if chunk_id in existing_ids:
                                        logger.warning(f"Skipping duplicate chunk ID: {chunk_id}")
                                        continue
                                        
                                    all_chunks.append(chunk)
                                    chunk_meta = file_metadata.copy()
                                    chunk_meta["chunk_index"] = i
                                    chunk_meta["chunk_count"] = len(extracted_chunks)
                                    chunk_meta["global_chunk_id"] = len(all_chunks)  # Global chunk counter
                                    chunk_metadata.append(chunk_meta)
                                    chunk_ids.append(chunk_id)

                    except Exception as e:
                        logger.error(f"Error processing file URL {file_url} for document {document_id}: {str(e)}", exc_info=True)
                        results.append({
                            "document_id": document_id,
                            "logical_nm": logical_nm,
                            "file_index": file_index,
                            "status": "error",
                            "message": f"Error processing file URL: {str(e)}"
                        })

            # Store all extracted chunks in ChromaDB (same batching logic)
            chunks_stored = 0
            if all_chunks:
                logger.info(f"Beginning embedding process for {len(all_chunks)} chunks from {len(file_urls)} files...")
                batch_size = 20
                
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]
                    batch_meta = chunk_metadata[i:i + batch_size]
                    
                    logger.debug(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                    
                    async with self.embedding_semaphore:
                        embed_tasks = [self._embed_text(chunk, batch_meta[idx]) for idx, chunk in enumerate(batch_chunks)]
                        embeddings_results = await asyncio.gather(*embed_tasks, return_exceptions=True)
                        
                        # Handle both successful results and exceptions
                        successful_embeds = []
                        for result in embeddings_results:
                            if isinstance(result, Exception):
                                logger.error(f"Embedding task failed with exception: {result}")
                                continue
                            elif result is not None and len(result) == 2 and result[0] is not None:
                                successful_embeds.append(result)
                        
                        logger.info(f"Batch {i//batch_size + 1}: Generated {len(successful_embeds)}/{len(batch_chunks)} embeddings successfully")
                        
                        # Extract embeddings and valid indices
                        embeddings = []
                        valid_indices = []
                        for idx, result in enumerate(embeddings_results):
                            if (not isinstance(result, Exception) and 
                                result is not None and 
                                len(result) == 2 and 
                                result[0] is not None):
                                embeddings.append(result[0])
                                valid_indices.append(idx)
                        
                        if not embeddings:
                            logger.warning(f"No valid embeddings in batch {i//batch_size + 1}, skipping ChromaDB storage")
                            continue
                    
                    try:
                        async with self.chroma_lock:
                            # Verify metadata is valid before adding to ChromaDB
                            valid_metadatas = []
                            for idx in valid_indices:
                                meta = {}
                                for k, v in batch_meta[idx].items():
                                    if isinstance(v, (str, int, float, bool)):
                                        meta[k] = v
                                    elif isinstance(v, list):
                                        meta[k] = ",".join(str(item) for item in v)
                                    elif isinstance(v, dict):
                                        meta[k] = json.dumps(v)
                                    else:
                                        meta[k] = str(v)
                                valid_metadatas.append(meta)
                            
                            # Add embeddings to ChromaDB
                            logger.info(f"Adding {len(embeddings)} embeddings to ChromaDB...")
                            
                            chromadb_collection.add(
                                ids=[batch_ids[idx] for idx in valid_indices],
                                embeddings=embeddings,
                                documents=[batch_chunks[idx] for idx in valid_indices],
                                metadatas=valid_metadatas
                            )
                            chunks_stored += len(embeddings)
                            logger.info(f"Successfully stored batch in ChromaDB, total stored: {chunks_stored}")
                    except Exception as chroma_error:
                        logger.error(f"ChromaDB storage error: {chroma_error}", exc_info=True)
                
                # Final verification
                try:
                    final_count = len(chromadb_collection.get()["ids"])
                    logger.info(f"Final ChromaDB collection count: {final_count} entries")
                except Exception as e:
                    logger.error(f"Failed to verify final ChromaDB status: {e}")

            return {
                "status": "success" if processed_count > 0 else "failed",
                "message": f"Processed {processed_count} items from {len(file_urls)} files, stored {chunks_stored} chunks in vector database",
                "processed_count": processed_count,
                "chunks_stored": chunks_stored,
                "file_count": len(file_urls),
                "all_files": all_logical_names,
                "details": results
            }

        except Exception as e:
            logger.error(f"Error in process_direct_uploads_with_urls: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing uploads with URLs: {str(e)}",
                "processed_count": 0,
                "details": []
            }

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
    
    async def _store_document_metadata_postgres(self, metadata: Dict[str, Any], processed_count: int, chunks_stored: int):
        """
        Store document metadata in PostgreSQL database.
        """
        try:
            if not self.use_postgresql:
                logger.warning("PostgreSQL not available, skipping metadata storage")
                return

            # Create metadata storage query - adapt this to your actual table structure
            query = """
                INSERT INTO document_metadata 
                (document_id, document_type, tags, source, created_at, custom_metadata, processed_count, chunks_stored)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, document_type) 
                DO UPDATE SET 
                    tags = EXCLUDED.tags,
                    custom_metadata = EXCLUDED.custom_metadata,
                    processed_count = EXCLUDED.processed_count,
                    chunks_stored = EXCLUDED.chunks_stored,
                    updated_at = NOW()
            """
            
            # Prepare values
            values = (
                metadata.get("document_id"),
                metadata.get("document_type"),
                metadata.get("tags_str"),
                metadata.get("source"),
                metadata.get("created_at"),
                json.dumps({k: v for k, v in metadata.items() if k not in ["document_id", "document_type", "tags_str", "source", "created_at"]}),
                processed_count,
                chunks_stored
            )
            
            # Execute query using your PostgreSQL connector's execute_query method
            self.db_connector.execute_query(query, values, fetch=False)
            logger.info(f"Stored metadata in PostgreSQL for document {metadata.get('document_id')}")
            
        except Exception as e:
            logger.error(f"Failed to store document metadata in PostgreSQL: {e}")
            # Don't raise exception as this is optional functionality

    async def delete_by_logical_name(self, logical_nm: str, deleted_by: str) -> Dict[str, Any]:
        """
        Delete documents from ChromaDB collection based on logical_nm.
        
        Args:
            logical_nm: The logical file name to delete from ChromaDB
            deleted_by: Member ID who performed the deletion (for audit trail)
        
        Returns:
            Dict with deletion results and metadata
        """
        try:
            logger.info(f"Starting deletion process for logical_nm: {logical_nm}")
            
            # Get ChromaDB collection
            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                logger.error("ChromaDB collection not initialized")
                return {
                    "status": "error",
                    "message": "ChromaDB collection not initialized",
                    "deleted_count": 0,
                    "logical_nm": logical_nm,
                    "deleted_by": deleted_by
                }
            
            # Build where conditions for ChromaDB query
            where_conditions = {"logical_nm": {"$eq": logical_nm}}
            
            # Step 1: Find matching chunks before deletion
            try:
                existing_data = chromadb_collection.get(
                    where=where_conditions,
                    include=["metadatas"]
                )
                matching_ids = existing_data.get("ids", [])
                matching_metadatas = existing_data.get("metadatas", [])
                
                if not matching_ids:
                    logger.warning(f"No chunks found for deletion with logical_nm: {logical_nm}")
                    return {
                        "status": "warning",
                        "message": f"No chunks found to delete for logical_nm: {logical_nm}",
                        "deleted_count": 0,
                        "logical_nm": logical_nm,
                        "deleted_by": deleted_by,
                        "affected_documents": []
                    }
                
                # Extract document_ids and other metadata for reporting
                document_ids = list(set(
                    meta.get("document_id", "unknown") 
                    for meta in matching_metadatas 
                    if meta.get("document_id")
                ))
                
                member_ids = list(set(
                    meta.get("member_id", "unknown") 
                    for meta in matching_metadatas 
                    if meta.get("member_id")
                ))
                
                # Get additional metadata for detailed reporting
                file_info = {
                    "content_types": list(set(
                        meta.get("content_type", "unknown") 
                        for meta in matching_metadatas 
                        if meta.get("content_type")
                    )),
                    "file_indices": list(set(
                        meta.get("file_index") 
                        for meta in matching_metadatas 
                        if meta.get("file_index") is not None
                    )),
                    "document_types": list(set(
                        meta.get("document_type", "unknown") 
                        for meta in matching_metadatas 
                        if meta.get("document_type")
                    ))
                }
                
                logger.info(f"Found {len(matching_ids)} chunks to delete for logical_nm: {logical_nm}")
                logger.info(f"Affected document_ids: {document_ids}")
                logger.info(f"Affected member_ids: {member_ids}")
                
            except Exception as query_error:
                logger.error(f"Error querying ChromaDB for logical_nm {logical_nm}: {query_error}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Failed to query ChromaDB: {str(query_error)}",
                    "deleted_count": 0,
                    "logical_nm": logical_nm,
                    "deleted_by": deleted_by
                }
            
            # Step 2: Perform deletion
            try:
                logger.info(f"Deleting {len(matching_ids)} chunks with logical_nm: {logical_nm}")
                chromadb_collection.delete(where=where_conditions)
                logger.info(f"ChromaDB delete operation completed for logical_nm: {logical_nm}")
                
            except Exception as delete_error:
                logger.error(f"ChromaDB delete operation failed: {delete_error}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Failed to delete from ChromaDB: {str(delete_error)}",
                    "deleted_count": 0,
                    "logical_nm": logical_nm,
                    "deleted_by": deleted_by,
                    "affected_documents": document_ids
                }
            
            # Step 3: Verify deletion was successful
            try:
                verification_data = chromadb_collection.get(where=where_conditions)
                remaining_chunks = len(verification_data.get("ids", []))
                
                if remaining_chunks > 0:
                    logger.error(f"Deletion incomplete: {remaining_chunks} chunks still exist after deletion")
                    return {
                        "status": "partial_failure",
                        "message": f"Partially deleted logical_nm '{logical_nm}': {remaining_chunks} chunks remain",
                        "deleted_count": len(matching_ids) - remaining_chunks,
                        "remaining_count": remaining_chunks,
                        "logical_nm": logical_nm,
                        "affected_documents": document_ids,
                        "affected_members": member_ids,
                        "file_info": file_info,
                        "deleted_by": deleted_by
                    }
                
            except Exception as verify_error:
                logger.warning(f"Could not verify deletion: {verify_error}")
                # Don't fail the operation if verification fails
            
            # Step 4: Log final collection status and return success
            try:
                final_count = chromadb_collection.count()
                logger.info(f"Deletion successful. ChromaDB collection now has {final_count} total chunks")
            except Exception as e:
                logger.warning(f"Could not verify final collection count: {e}")
                final_count = "unknown"
            
            return {
                "status": "success",
                "message": f"Successfully deleted logical_nm '{logical_nm}' ({len(matching_ids)} chunks)",
                "deleted_count": len(matching_ids),
                "logical_nm": logical_nm,
                "affected_documents": document_ids,
                "affected_members": member_ids,
                "file_info": file_info,
                "collection_count_after": final_count,
                "deleted_by": deleted_by,
                "deleted_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in delete_by_logical_name business logic: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error during deletion: {str(e)}",
                "deleted_count": 0,
                "logical_nm": logical_nm,
                "deleted_by": deleted_by
            }

    async def update_documents(self, update_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update ChromaDB collection based on update mode and parameters.
        
        Args:
            update_request: Dictionary containing:
                - update_mode: "add", "replace-by-logical-name", or "replace-by-document-id"
                - resolve_data: Optional JSON string with document metadata
                - file_urls: List of S3 URLs for new/replacement files
                - logical_names: List of logical names matching file_urls
                - target_logical_nm: Target logical name for replacement (if applicable)
                - target_document_id: Target document ID for replacement (if applicable)
                - updated_by: Member ID who performed the update
        
        Returns:
            Dict with update results and metadata
        """
        try:
            # Extract parameters from request
            update_mode = update_request["update_mode"]
            resolve_data = update_request.get("resolve_data")
            file_urls = update_request["file_urls"]
            logical_names = update_request.get("logical_names", [])
            target_logical_nm = update_request.get("target_logical_nm")
            target_document_id = update_request.get("target_document_id")
            updated_by = update_request["updated_by"]
            
            logger.info(f"Starting update process with mode: {update_mode}, files: {len(file_urls)}")
            
            update_results = []
            
            # Step 1: Handle replacement modes (delete existing first)
            if update_mode in ["replace-by-logical-name", "replace-by-document-id"]:
                logger.info(f"Replace mode: deleting existing content first")
                
                # Get ChromaDB collection
                chromadb_collection = get_chromadb_collection()
                if not chromadb_collection:
                    return {
                        "status": "error",
                        "message": "ChromaDB collection not initialized",
                        "update_mode": update_mode,
                        "updated_by": updated_by
                    }
                
                # Build where conditions based on mode
                if update_mode == "replace-by-logical-name":
                    where_conditions = {"logical_nm": {"$eq": target_logical_nm}}
                    logger.info(f"Deleting existing chunks with logical_nm: {target_logical_nm}")
                else:  # replace-by-document-id
                    where_conditions = {"document_id": {"$eq": target_document_id}}
                    logger.info(f"Deleting existing chunks with document_id: {target_document_id}")
                
                try:
                    # Query existing data before deletion
                    existing_data = chromadb_collection.get(
                        where=where_conditions,
                        include=["metadatas"]
                    )
                    existing_count = len(existing_data.get("ids", []))
                    existing_metadatas = existing_data.get("metadatas", [])
                    
                    if existing_count > 0:
                        # Extract information about what's being deleted
                        deleted_document_ids = list(set(
                            meta.get("document_id", "unknown") 
                            for meta in existing_metadatas 
                            if meta.get("document_id")
                        ))
                        
                        # Perform deletion
                        chromadb_collection.delete(where=where_conditions)
                        logger.info(f"Deleted {existing_count} existing chunks")
                        
                        update_results.append({
                            "operation": f"delete_existing_{update_mode}",
                            "status": "success",
                            "message": f"Deleted {existing_count} existing chunks",
                            "deleted_count": existing_count,
                            "deleted_document_ids": deleted_document_ids,
                            "target_logical_nm": target_logical_nm,
                            "target_document_id": target_document_id
                        })
                    else:
                        logger.info(f"No existing chunks found for {update_mode}")
                        update_results.append({
                            "operation": f"delete_existing_{update_mode}",
                            "status": "info",
                            "message": "No existing chunks to delete",
                            "deleted_count": 0,
                            "target_logical_nm": target_logical_nm,
                            "target_document_id": target_document_id
                        })
                        
                except Exception as delete_error:
                    logger.error(f"Error deleting existing chunks: {delete_error}", exc_info=True)
                    return {
                        "status": "error",
                        "message": f"Failed to delete existing chunks: {str(delete_error)}",
                        "update_mode": update_mode,
                        "updated_by": updated_by,
                        "details": update_results
                    }
            
            # Step 2: Process new files
            logger.info(f"Processing {len(file_urls)} new files")
            
            # Parse or create resolve_data
            if resolve_data:
                try:
                    resolve_data_dict = json.loads(resolve_data)
                except json.JSONDecodeError as e:
                    return {
                        "status": "error",
                        "message": f"Invalid JSON in resolve_data: {str(e)}",
                        "update_mode": update_mode,
                        "updated_by": updated_by
                    }
            else:
                # Auto-generate minimal resolve_data
                resolve_data_dict = await self._generate_default_resolve_data(
                    update_mode, target_document_id, updated_by, len(file_urls)
                )
                logger.info(f"Auto-generated resolve_data with document_id: {resolve_data_dict['document_id']}")
            
            # Validate document_type if provided
            document_type = resolve_data_dict.get("document_type")
            if document_type:
                valid_types = await self.get_valid_document_types()
                if document_type not in valid_types:
                    return {
                        "status": "error",
                        "message": f"Invalid document_type: {document_type}. Valid types: {valid_types}",
                        "update_mode": update_mode,
                        "updated_by": updated_by
                    }
            
            # Inject metadata for tracking
            if "custom_metadata" not in resolve_data_dict:
                resolve_data_dict["custom_metadata"] = {}
            
            resolve_data_dict["custom_metadata"]["updated_by"] = updated_by
            resolve_data_dict["custom_metadata"]["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            resolve_data_dict["custom_metadata"]["operation_type"] = update_mode
            resolve_data_dict["custom_metadata"]["target_logical_nm"] = target_logical_nm
            resolve_data_dict["custom_metadata"]["target_document_id"] = target_document_id
            
            if logical_names:
                resolve_data_dict["custom_metadata"]["logical_names"] = logical_names
            
            # Convert back to JSON string
            enhanced_resolve_data = json.dumps(resolve_data_dict)
            
            # Process new files using existing ingestion service
            try:
                ingest_result = await self.process_direct_uploads_with_urls(enhanced_resolve_data, file_urls)
                
                update_results.append({
                    "operation": "add_new_files",
                    "status": ingest_result.get("status"),
                    "message": ingest_result.get("message"),
                    "processed_count": ingest_result.get("processed_count", 0),
                    "chunks_stored": ingest_result.get("chunks_stored", 0),
                    "file_count": ingest_result.get("file_count", 0),
                    "document_id": resolve_data_dict.get("document_id"),
                    "details": ingest_result.get("details", [])
                })
                
            except Exception as ingest_error:
                logger.error(f"Error during file ingestion: {ingest_error}", exc_info=True)
                update_results.append({
                    "operation": "add_new_files",
                    "status": "error",
                    "message": f"Failed to ingest new files: {str(ingest_error)}",
                    "processed_count": 0,
                    "chunks_stored": 0,
                    "file_count": 0
                })
            
            # Step 3: Compile final response
            total_processed = sum(result.get("processed_count", 0) for result in update_results)
            total_chunks = sum(result.get("chunks_stored", 0) for result in update_results)
            total_deleted = sum(result.get("deleted_count", 0) for result in update_results)
            
            # Determine overall status
            operation_statuses = [result.get("status") for result in update_results]
            if "error" in operation_statuses:
                overall_status = "partial_failure" if any(s == "success" for s in operation_statuses) else "error"
            elif "warning" in operation_statuses:
                overall_status = "warning"
            else:
                overall_status = "success"
            
            return {
                "status": overall_status,
                "message": f"Update completed using {update_mode} mode",
                "update_mode": update_mode,
                "target_logical_nm": target_logical_nm,
                "target_document_id": target_document_id,
                "updated_by": updated_by,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "summary": {
                    "files_processed": total_processed,
                    "chunks_stored": total_chunks,
                    "chunks_deleted": total_deleted,
                    "operations_performed": len(update_results)
                },
                "details": update_results
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in update_documents business logic: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error during update: {str(e)}",
                "update_mode": update_request.get("update_mode", "unknown"),
                "updated_by": update_request.get("updated_by", "unknown")
            }

    async def refresh_chromadb_collection(self, refreshed_by: str) -> Dict[str, Any]:
        """
        Refresh ChromaDB collection and all related services.
        
        Args:
            refreshed_by: Member ID who performed the refresh (for audit trail)
        
        Returns:
            Dict with refresh results and status information
        """
        try:
            logger.info(f"Starting ChromaDB refresh process initiated by: {refreshed_by}")
            import chromadb
            refresh_operations = []
            start_time = datetime.datetime.now(datetime.timezone.utc)
            
            # Step 1: Reconnect to existing ChromaDB collection
            try:
                persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
                
                try:
                    # Get existing collection (don't delete)
                    chroma_coll = persistent_client.get_collection("netbackup_docs")
                    current_count = chroma_coll.count()
                    logger.info(f"Reconnected to existing ChromaDB collection with {current_count} documents")
                    
                    refresh_operations.append({
                        "operation": "reconnect_chromadb_collection",
                        "status": "success",
                        "message": f"Reconnected to existing collection with {current_count} documents",
                        "collection_count": current_count
                    })
                    
                except Exception as get_error:
                    logger.warning(f"Could not get existing collection, creating new one: {get_error}")
                    chroma_coll = persistent_client.get_or_create_collection(
                        name="netbackup_docs",
                        metadata={"hnsw:space": "cosine"}
                    )
                    current_count = chroma_coll.count()
                    logger.info(f"Created new ChromaDB collection with {current_count} documents")
                    
                    refresh_operations.append({
                        "operation": "create_chromadb_collection",
                        "status": "warning",
                        "message": f"Created new collection (existing not found) with {current_count} documents",
                        "collection_count": current_count,
                        "details": str(get_error)
                    })
                    
            except Exception as client_error:
                logger.error(f"Failed to initialize ChromaDB client: {client_error}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Failed to initialize ChromaDB client: {str(client_error)}",
                    "refreshed_by": refreshed_by,
                    "operations": refresh_operations
                }
            
            # Step 2: Recreate vector store with fresh connection
            try:
                embeddings = OllamaEmbeddings(model="mxbai-embed-large")
                
                try:
                    from langchain_chroma import Chroma
                except ImportError:
                    from langchain.vectorstores import Chroma

                vector_store = Chroma(
                    client=persistent_client,
                    embedding_function=embeddings,
                    collection_name="netbackup_docs",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                
                refresh_operations.append({
                    "operation": "recreate_vector_store",
                    "status": "success",
                    "message": "Successfully recreated vector store with fresh connection"
                })
                
            except Exception as vector_error:
                logger.error(f"Failed to recreate vector store: {vector_error}", exc_info=True)
                refresh_operations.append({
                    "operation": "recreate_vector_store",
                    "status": "error",
                    "message": f"Failed to recreate vector store: {str(vector_error)}"
                })
                vector_store = None
            
            # Step 3: Update global state
            try:
                refresh_success = set_globals(
                    chroma_coll=chroma_coll,
                    rag=get_rag_chain(),
                    vect_store=vector_store,
                    prompt=get_global_prompt(),
                    workflow=get_workflow(),
                    memory=get_memory()
                )
                
                if refresh_success:
                    refresh_operations.append({
                        "operation": "update_global_state",
                        "status": "success",
                        "message": "Successfully updated global state"
                    })
                    logger.info("Successfully updated global state after refresh")
                else:
                    refresh_operations.append({
                        "operation": "update_global_state",
                        "status": "error",
                        "message": "Failed to update global state"
                    })
                    logger.error("Failed to update global state")
                    
            except Exception as globals_error:
                logger.error(f"Error updating global state: {globals_error}", exc_info=True)
                refresh_operations.append({
                    "operation": "update_global_state",
                    "status": "error",
                    "message": f"Error updating global state: {str(globals_error)}"
                })
            
            # Step 5: Verify the refresh worked
            verification_results = []
            try:
                # Test ChromaDB access
                test_count = get_chromadb_collection().count()
                verification_results.append({
                    "component": "chromadb_collection",
                    "status": "success",
                    "message": f"ChromaDB collection accessible with {test_count} documents",
                    "count": test_count
                })
                logger.info(f"Verification: ChromaDB collection accessible with {test_count} documents")
                
            except Exception as verify_chroma_error:
                verification_results.append({
                    "component": "chromadb_collection",
                    "status": "error",
                    "message": f"ChromaDB verification failed: {str(verify_chroma_error)}"
                })
                logger.error(f"ChromaDB verification failed: {verify_chroma_error}")
            
            try:
                # Test vector store access
                if vector_store:
                    verification_results.append({
                        "component": "vector_store",
                        "status": "success",
                        "message": "Vector store initialized successfully"
                    })
                    logger.info("Verification: Vector store initialized successfully")
                else:
                    verification_results.append({
                        "component": "vector_store",
                        "status": "error",
                        "message": "Vector store is None after refresh"
                    })
                    
            except Exception as verify_vector_error:
                verification_results.append({
                    "component": "vector_store",
                    "status": "error",
                    "message": f"Vector store verification failed: {str(verify_vector_error)}"
                })
            
            # Step 6: Determine overall status and compile response
            end_time = datetime.datetime.now(datetime.timezone.utc)
            refresh_duration = (end_time - start_time).total_seconds()
            
            # Analyze operation results
            operation_statuses = [op.get("status") for op in refresh_operations]
            verification_statuses = [vr.get("status") for vr in verification_results]
            
            if "error" in operation_statuses or "error" in verification_statuses:
                if any(s == "success" for s in operation_statuses):
                    overall_status = "partial_success"
                else:
                    overall_status = "error"
            elif "warning" in operation_statuses or "warning" in verification_statuses:
                overall_status = "warning"
            else:
                overall_status = "success"
            
            success_count = len([op for op in refresh_operations if op.get("status") == "success"])
            
            return {
                "status": overall_status,
                "message": f"ChromaDB refresh completed with {success_count}/{len(refresh_operations)} operations successful",
                "collection_count": current_count,
                "refreshed_by": refreshed_by,
                "refresh_duration_seconds": round(refresh_duration, 2),
                "started_at": start_time.isoformat(),
                "completed_at": end_time.isoformat(),
                "operations_summary": {
                    "total_operations": len(refresh_operations),
                    "successful_operations": len([op for op in refresh_operations if op.get("status") == "success"]),
                    "failed_operations": len([op for op in refresh_operations if op.get("status") == "error"]),
                    "warning_operations": len([op for op in refresh_operations if op.get("status") == "warning"])
                },
                "operations": refresh_operations,
                "verification_results": verification_results
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in refresh_chromadb_collection business logic: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Unexpected error during refresh: {str(e)}",
                "refreshed_by": refreshed_by,
                "operations": refresh_operations if 'refresh_operations' in locals() else []
            }
    
    
    
    async def process_direct_uploads(self, resolve_data: str, files: List[UploadFile]) -> Dict[str, Any]:
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized.", "results": []}

        try:
            # Parse resolve_data
            try:
                resolve_data_dict = json.loads(resolve_data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in resolve_data: {e}")
                return {
                    "status": "error",
                    "message": f"Invalid JSON in resolve_data: {str(e)}",
                    "processed_count": 0,
                    "details": []
                }

            # Extract core fields
            document_id = resolve_data_dict.get("document_id", "unknown")
            document_type = resolve_data_dict.get("document_type", "unknown")
            tags = resolve_data_dict.get("tags", [])
            content = BeautifulSoup(resolve_data_dict.get("content", ""), "html.parser").get_text(separator=" ", strip=True)
            custom_metadata = resolve_data_dict.get("custom_metadata", {})

            # Validate required fields
            if not document_id or document_id == "unknown":
                logger.error("Missing or invalid document_id")
                return {
                    "status": "error",
                    "message": "Missing or invalid document_id",
                    "processed_count": 0,
                    "details": []
                }

            if not document_type or document_type == "unknown":
                logger.error("Missing or invalid document_type")
                return {
                    "status": "error",
                    "message": "Missing or invalid document_type",
                    "processed_count": 0,
                    "details": []
                }

            # Validate document_type
            if document_type not in self.valid_document_types:
                logger.error(f"Invalid document_type: {document_type}")
                return {
                    "status": "error",
                    "message": f"Invalid document_type: {document_type}. Must be one of {self.valid_document_types}",
                    "processed_count": 0,
                    "details": []
                }

            # Convert any lists or complex structures in metadata to strings for ChromaDB compatibility
            tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)

            # Process custom_metadata to ensure all values are compatible with ChromaDB
            processed_custom_metadata = {}
            for key, value in custom_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    processed_custom_metadata[key] = value
                elif isinstance(value, list):
                    processed_custom_metadata[key] = ",".join(str(item) for item in value)
                elif isinstance(value, dict):
                    processed_custom_metadata[key] = json.dumps(value)
                else:
                    processed_custom_metadata[key] = str(value)

            # Build base metadata
            metadata = {
                "document_id": document_id,
                "document_type": document_type,
                "tags_str": tags_str,  # Store as string instead of list
                "source": f"{document_type}_documents",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), #datetime.datetime.utcnow().isoformat(),
            }
            # Add processed custom metadata
            metadata.update(processed_custom_metadata)

            # Check for duplicates
            existing_ids = set(chromadb_collection.get()["ids"])
            report_id = f"{document_type}_{document_id}"
            if any(id_.startswith(report_id) for id_ in existing_ids):
                logger.info(f"Skipping duplicate document for {document_type} with ID {document_id}")
                return {
                    "status": "success",
                    "message": "Document already processed",
                    "processed_count": 0,
                    "details": []
                }

            processed_count = 0
            all_chunks = []
            chunk_metadata = []
            chunk_ids = []

            # Process text content if present
            if content.strip():
                html_result = await process_html_content(
                    html_content=content,
                    metadata=metadata,
                    html_handler=self.html_handler,
                    vision_extractor=self.vision_extractor
                )
                results.append(html_result)
                if html_result["status"] == "success":
                    processed_count += 1

            # Process uploaded files
            if files:
                for file in files:
                    logical_nm = file.filename
                    logger.info(f"Processing uploaded file for document {document_id}: {logical_nm}")

                    if file.size > self.MAX_FILE_SIZE:
                        results.append({
                            "document_id": document_id,
                            "logical_nm": logical_nm,
                            "status": "error",
                            "message": f"File exceeds maximum size of {self.MAX_FILE_SIZE} bytes"
                        })
                        continue

                    try:
                        # Read file content
                        file_content = await file.read()
                        if not file_content:
                            results.append({
                                "document_id": document_id,
                                "logical_nm": logical_nm,
                                "status": "error",
                                "message": "Empty file content"
                            })
                            continue

                        # Build file-specific metadata
                        file_metadata = metadata.copy()
                        file_metadata["logical_nm"] = logical_nm
                        
                        # Process file content
                        result = await process_file_content(
                            file_content=file_content,
                            filename=logical_nm,
                            metadata=file_metadata,
                            model_manager=self.model_manager
                        )
                        
                        # Handle the result
                        results.append(result)
                        
                        # Only proceed if processing was successful
                        if result["status"] == "success":
                            processed_count += 1
                            
                            # Extract chunks for vector storage
                            extracted_chunks = result.get("chunks", [])
                            if extracted_chunks:
                                # Store each chunk for later batch embedding
                                for i, chunk in enumerate(extracted_chunks):
                                    chunk_id = f"{document_type}_{document_id}_{logical_nm}_chunk_{i}"
                                    if chunk_id in existing_ids:
                                        logger.warning(f"Skipping duplicate chunk ID: {chunk_id}")
                                        continue
                                        
                                    all_chunks.append(chunk)
                                    chunk_meta = file_metadata.copy()
                                    chunk_meta["chunk_index"] = i
                                    chunk_meta["chunk_count"] = len(extracted_chunks)
                                    chunk_metadata.append(chunk_meta)
                                    chunk_ids.append(chunk_id)

                    except Exception as e:
                        logger.error(f"Error processing file {logical_nm} for document {document_id}: {str(e)}", exc_info=True)
                        results.append({
                            "document_id": document_id,
                            "logical_nm": logical_nm,
                            "status": "error",
                            "message": f"Error processing file: {str(e)}"
                        })
            else:
                logger.info(f"No files to process for document {document_id}")

            # Store all extracted chunks in ChromaDB
            chunks_stored = 0
            if all_chunks:
                logger.info(f"Beginning embedding process for {len(all_chunks)} chunks...")
                batch_size = 20
                
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]
                    batch_meta = chunk_metadata[i:i + batch_size]
                    
                    logger.debug(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                    
                    async with self.embedding_semaphore:
                        embed_tasks = [self._embed_text(chunk, batch_meta[idx]) for idx, chunk in enumerate(batch_chunks)]
                        embeddings_results = await asyncio.gather(*embed_tasks, return_exceptions=True)
                        
                        # Handle both successful results and exceptions
                        successful_embeds = []
                        for result in embeddings_results:
                            if isinstance(result, Exception):
                                logger.error(f"Embedding task failed with exception: {result}")
                                continue
                            elif result is not None and len(result) == 2 and result[0] is not None:
                                successful_embeds.append(result)
                        
                        logger.info(f"Batch {i//batch_size + 1}: Generated {len(successful_embeds)}/{len(batch_chunks)} embeddings successfully")
                        
                        # Extract embeddings and valid indices
                        embeddings = []
                        valid_indices = []
                        for idx, result in enumerate(embeddings_results):
                            if (not isinstance(result, Exception) and 
                                result is not None and 
                                len(result) == 2 and 
                                result[0] is not None):
                                embeddings.append(result[0])
                                valid_indices.append(idx)
                        
                        # Check if we have any valid embeddings to store
                        if not embeddings:
                            logger.warning(f"No valid embeddings in batch {i//batch_size + 1}, skipping ChromaDB storage")
                            continue
                    
                    try:
                        async with self.chroma_lock:
                            # Verify metadata is valid before adding to ChromaDB
                            valid_metadatas = []
                            for idx in valid_indices:
                                # Final validation of metadata items to ensure ChromaDB compatibility
                                meta = {}
                                for k, v in batch_meta[idx].items():
                                    if isinstance(v, (str, int, float, bool)):
                                        meta[k] = v
                                    elif isinstance(v, list):
                                        meta[k] = ",".join(str(item) for item in v)
                                    elif isinstance(v, dict):
                                        meta[k] = json.dumps(v)
                                    else:
                                        meta[k] = str(v)
                                valid_metadatas.append(meta)
                            
                            # Add the embeddings to ChromaDB
                            logger.info(f"Adding {len(embeddings)} embeddings to ChromaDB...")
                            
                            # Log one metadata example for debugging
                            if valid_metadatas:
                                logger.debug(f"Sample metadata: {valid_metadatas[0]}")
                                
                            chromadb_collection.add(
                                ids=[batch_ids[idx] for idx in valid_indices],
                                embeddings=embeddings,
                                documents=[batch_chunks[idx] for idx in valid_indices],
                                metadatas=valid_metadatas
                            )
                            chunks_stored += len(embeddings)
                            logger.info(f"Successfully stored batch in ChromaDB, total stored: {chunks_stored}")
                    except Exception as chroma_error:
                        logger.error(f"ChromaDB storage error: {chroma_error}", exc_info=True)
                
                # Final verification
                try:
                    final_count = len(chromadb_collection.get()["ids"])
                    logger.info(f"Final ChromaDB collection count: {final_count} entries")
                except Exception as e:
                    logger.error(f"Failed to verify final ChromaDB status: {e}")

            return {
                "status": "success" if processed_count > 0 else "failed",
                "message": f"Processed {processed_count} items, stored {chunks_stored} chunks in vector database",
                "processed_count": processed_count,
                "chunks_stored": chunks_stored,
                "details": results
            }

        except Exception as e:
            logger.error(f"Error in process_direct_uploads: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing uploads: {str(e)}",
                "processed_count": 0,
                "details": results
            }