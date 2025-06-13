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
from sentence_transformers import SentenceTransformer


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
    def __init__(self, db_connector: None = None, model_manager=None,lazy_init=False):
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
        
        self.embedding_model = None
        self._init_embedding_model()

        self.kb_vector_store       = get_vector_store()
        FileHandlerFactory.initialize(model_manager)

        # Resource limits for downloads and embeddings
        self.MAX_CONCURRENT_DOWNLOADS = 5
        self.MAX_CONCURRENT_EMBEDDINGS = 10
        self.download_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_DOWNLOADS)
        self.embedding_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_EMBEDDINGS)
        self.handlers = {}
        if not lazy_init:
            logger.info("IngestService initialized with eager loading mode (handlers loaded on-demand)")
        else:
            logger.info("IngestService initialized with lazy loading mode")
            



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

    def _init_embedding_model(self):
        """Initialize sentence-transformers model for embeddings"""
        try:
            
            from src.core.models.model_manager import ModelManager
            
            # Try to use passed model_manager first, then singleton
            if self.model_manager:
                self.embedding_model = self.model_manager.get_embedding_model()
                logger.info(f"IngestService using passed ModelManager's embedding model")
            else:
                # Fallback to ModelManager singleton
                model_manager = ModelManager()  # Get singleton instance
                self.embedding_model = model_manager.get_embedding_model()
                logger.info(f"IngestService using ModelManager singleton's embedding model")
                
        except Exception as e:
            logger.error(f"Failed to get embedding model from ModelManager: {e}")
            # Last resort fallback - create new instance (only if ModelManager fails)
            try:
                from dotenv import load_dotenv
                from sentence_transformers import SentenceTransformer
                load_dotenv()
                
                model_path = os.getenv('EMBEDDING_MODEL_PATH')
                model_name = os.getenv('EMBEDDING_MODEL_NAME', 'mixedbread-ai/mxbai-embed-large-v1')
                
                if model_path and os.path.exists(model_path):
                    self.embedding_model = SentenceTransformer(model_path, device='cuda')
                    logger.warning(f"IngestService fallback: loaded embedding model from cache: {model_path}")
                else:
                    cache_dir = os.path.dirname(model_path) if model_path else r"C:\AI_Models\local_cache"
                    self.embedding_model = SentenceTransformer(
                        model_name, cache_folder=cache_dir, device='cuda'
                    )
                    logger.warning(f"IngestService fallback: loaded embedding model: {model_name}")
            except Exception as fallback_error:
                logger.error(f"All embedding model initialization methods failed: {fallback_error}")
                self.embedding_model = None

    def _initialize_handlers_on_demand(self):
        """Initialize handlers only when first needed (lazy loading)."""
        if not hasattr(self, '_handlers_initialized'):
            logger.info("Initializing handlers on first use...")
            
            # Initialize specific handlers that are directly referenced
            if self.model_manager:
                self.html_handler = HTMLContentHandler(model_manager=self.model_manager)
                self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
            else:
                self.html_handler = HTMLContentHandler()
                self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
            
            self._handlers_initialized = True
            logger.info("Handlers initialized successfully")

    # def _initialize_file_handlers(self):
    #     """Initialize all file handlers - extracted to separate method"""
    #     FileHandlerFactory.initialize(self.model_manager)
        
    #     # Initialize handlers using FileHandlerFactory
    #     self.handlers = {
    #         'pdf': FileHandlerFactory.get_handler_for_extension('pdf'),
    #         'image': FileHandlerFactory.get_handler_for_extension('png'),
    #         'hwp': FileHandlerFactory.get_handler_for_extension('hwp'),
    #         'doc': FileHandlerFactory.get_handler_for_extension('doc'),
    #         'msg': FileHandlerFactory.get_handler_for_extension('msg'),
    #         'excel': FileHandlerFactory.get_handler_for_extension('xlsx'),
    #         'pptx': FileHandlerFactory.get_handler_for_extension('pptx'),
    #         'txt': FileHandlerFactory.get_handler_for_extension('txt'),
    #         'rtf': FileHandlerFactory.get_handler_for_extension('rtf'),
    #     }

    #     # Initialize specific handlers
    #     if self.model_manager:
    #         self.pdf_handler = PDFHandler(model_manager=self.model_manager)
    #         self.image_handler = ImageHandler(model_manager=self.model_manager)
    #         self.msg_handler = MSGHandler(model_manager=self.model_manager)
    #         self.doc_handler = AdvancedDocHandler(model_manager=self.model_manager)
    #         self.hwp_handler = HWPHandler(model_manager=self.model_manager)
    #         self.html_handler = HTMLContentHandler(model_manager=self.model_manager)
    #         self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
    #         self.excel_handler = ExcelHandler(model_manager=self.model_manager)
    #         self.pptx_handler = PPTXHandler(model_manager=self.model_manager)
    #         self.txt_handler = TXTHandler(model_manager=self.model_manager)
    #         self.rtf_handler = RTFHandler(model_manager=self.model_manager)
    #     else:
    #         self.pdf_handler = PDFHandler()
    #         self.image_handler = ImageHandler()
    #         self.msg_handler = MSGHandler()
    #         self.doc_handler = AdvancedDocHandler()
    #         self.hwp_handler = HWPHandler()
    #         self.html_handler = HTMLContentHandler()
    #         self.vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
    #         self.excel_handler = ExcelHandler()
    #         self.pptx_handler = PPTXHandler()
    #         self.txt_handler = TXTHandler()
    #         self.rtf_handler = RTFHandler()
    
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
        """Generate embeddings using sentence-transformers with GPU acceleration."""
        if metadata is None:
            metadata = {}
            
        # Enhanced text cleaning
        cleaned_text = self._clean_text_for_embedding(text, aggressive=False)
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            logger.warning(f"Text too short or empty after cleaning")
            return None, metadata

        try:
            if not self.embedding_model:
                logger.error("Embedding model not initialized")
                return None, metadata

            # Generate embedding using sentence-transformers (this will use GPU)
            embedding = await asyncio.to_thread(
                self.embedding_model.encode,
                cleaned_text,
                convert_to_tensor=False,  # Return as list
                normalize_embeddings=True  # Normalize for better similarity
            )
            
            # Convert to list if it's a tensor/numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy().tolist()
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions using GPU")
            return embedding, metadata
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
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

    def _validate_physical_filename(self, physical_nm: str) -> bool:
        """
        Validate physical filename to handle database edge cases.
        
        Args:
            physical_nm (str): The physical filename to validate
            
        Returns:
            bool: True if filename is valid for processing, False otherwise
        """
        if not physical_nm or not isinstance(physical_nm, str):
            logger.warning(f"Invalid physical_nm: {physical_nm}")
            return False
        
        # Handle problematic cases from your database
        if physical_nm.endswith('.null'):
            logger.warning(f"File with .null extension detected: {physical_nm}")
            return False
        
        # Check for files without extensions
        if '.' not in physical_nm:
            logger.warning(f"File without extension detected: {physical_nm}")
            return False
        
        # Check for empty or very short filenames
        if len(physical_nm.strip()) < 3:
            logger.warning(f"Suspiciously short filename: {physical_nm}")
            return False
        
        # Check for files with only special characters
        import re
        if not re.search(r'[a-zA-Z0-9]', physical_nm):
            logger.warning(f"Filename contains no alphanumeric characters: {physical_nm}")
            return False
        
        return True
    
    async def process_direct_uploads_with_urls(self, input_data: str, file_urls: List[str]) -> Dict[str, Any]:
        """
        Process document data and S3 file URLs with enhanced multi-file support.
        Each file gets its own logical_nm in metadata.
        """
        try:
            logger.info(f"Raw input_data: {input_data}")
            
            # Parse resolve_data
            try:
                input_data_dict = json.loads(input_data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in input_data: {e}")
                return {
                    "status": "error",
                    "message": f"Invalid JSON in input_data: {str(e)}",
                    "processed_count": 0,
                    "details": []
                }

            # Extract core fields
            document_id = input_data_dict.get("document_id", "unknown")
            document_type = input_data_dict.get("document_type", "unknown")
            tags = input_data_dict.get("tags", [])
            content = input_data_dict.get("content", "")
            custom_metadata = input_data_dict.get("custom_metadata", {})
            member_id = custom_metadata.get("member_id", "unknown")
            physical_nm  = custom_metadata.get("physical_nm", [])

            all_physical_names = []

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

            if not hasattr(self, '_cached_handlers'):
                self._cached_handlers = {}
                logger.info("Initializing handler cache for first time")

            # Initialize handlers on demand
            self._initialize_handlers_on_demand()

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
                # Debug: Log what we extracted from custom_metadata
                logger.info(f"Extracted physical_nm  from custom_metadata: {physical_nm }")
                logger.info(f"File URLs count: {len(file_urls)}, logical_nm count: {len(physical_nm )}")
                
                if physical_nm and len(physical_nm ) == len(file_urls):
                    # Use clean logical names from frontend (via custom_metadata)
                    all_physical_names = physical_nm 
                    logger.info(f"✅ Using clean physical_nm from backend server: {all_physical_names}")
                else:
                    # FALLBACK: Clean UUID prefixes from S3 URLs
                    logger.warning(f"⚠️  Physical names mismatch or missing. physical_nm: {physical_nm}, file_urls: {len(file_urls)}")
                    all_physical_names = []
                    for file_url in file_urls:
                        # Remove query parameters and extract filename
                        clean_url = file_url.split("?")[0]  
                        filename = clean_url.split("/")[-1]
                        
                        # Clean UUID prefixes if present
                        if filename.count('-') >= 4:  # Likely has UUID prefix
                            # Check if it starts with UUID pattern
                            import re
                            uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                            if re.match(uuid_pattern, filename):
                                # Try to find actual filename after UUID
                                parts = filename.split('-', 5)  # Split on first 5 dashes
                                if len(parts) > 5 and '.' in parts[5]:
                                    filename = parts[5]  # Use everything after 5th dash
                                # If no good filename found, use last part
                                elif '.' in filename:
                                    # Keep the UUID filename if it has extension
                                    pass
                                else:
                                    filename = "unknown_file"
                        
                        all_physical_names.append(filename)
                    
                    logger.info(f"🔧 Cleaned logical_nm from S3 URLs: {all_physical_names}")
                
                # Add file list to base metadata for reference
                base_metadata["file_count"] = len(file_urls)
                base_metadata["all_files"] = ",".join(all_physical_names)
                
                for file_index, file_url in enumerate(file_urls):
                    physical_nm = all_physical_names[file_index]
                    logger.info(f"Processing file {file_index + 1}/{len(file_urls)} for document {document_id}: {physical_nm}")
                    
                    try:
                        # 🚀 STEP 1: IMMEDIATE EXTENSION VALIDATION (microseconds - no I/O)
                        from src.core.utils.file_identification import detect_file_type
                        
                        logger.debug(f"🔍 Step 1: Validating extension for {physical_nm}")
                        file_type = detect_file_type(physical_nm)
                        
                        if file_type == 'unknown':
                            logger.error(f"❌ Unsupported file type detected from extension: {physical_nm}")
                            results.append({
                                "document_id": document_id,
                                "physical_nm": physical_nm,
                                "file_index": file_index,
                                "status": "error",
                                "message": f"Unsupported file type: {physical_nm}. Supported types: pdf, doc, hwp, txt, rtf, excel, pptx, msg, image"
                            })
                            continue  # Skip download and processing entirely
                        
                        logger.info(f"✅ Step 1 Complete: {physical_nm} → {file_type} (extension-based)")
                        
                        # 🚨 STEP 2: ADDITIONAL VALIDATION (optional database-specific checks)
                        if not self._validate_physical_filename(physical_nm):
                            logger.error(f"❌ Filename validation failed: {physical_nm}")
                            results.append({
                                "document_id": document_id,
                                "physical_nm": physical_nm,
                                "file_index": file_index,
                                "status": "error",
                                "message": f"Invalid filename: {physical_nm}"
                            })
                            continue
                        
                        # ⏰ STEP 3: URL EXPIRATION CHECK (network validation - milliseconds)
                        if self._is_url_expired(file_url):
                            logger.warning(f"❌ Expired URL detected: {file_url}")
                            results.append({
                                "document_id": document_id,
                                "physical_nm": physical_nm,
                                "file_index": file_index,
                                "status": "error",
                                "message": "Pre-signed URL has expired"
                            })
                            continue
                        
                        logger.debug(f"✅ Step 3 Complete: URL validation passed for {physical_nm}")
                        
                        # 📥 STEP 4: DOWNLOAD FILE (network I/O - seconds)
                        logger.debug(f"📥 Step 4: Downloading {physical_nm}...")
                        async with self.download_semaphore:
                            download_result = await download_file_from_url(file_url)

                        if isinstance(download_result, tuple):
                            file_content, content_type = download_result
                        else:
                            file_content = download_result
                            content_type = None

                        if not file_content:
                            logger.error(f"❌ Download failed for {physical_nm}")
                            results.append({
                                "document_id": document_id,
                                "physical_nm": physical_nm,
                                "file_index": file_index,
                                "status": "error", 
                                "message": f"Failed to download file from {file_url}"
                            })
                            continue
                        
                        logger.info(f"✅ Step 4 Complete: Downloaded {len(file_content)} bytes for {physical_nm}")
                        
                        # 💾 STEP 5: WRITE TO TEMP FILE (disk I/O - milliseconds)
                        logger.debug(f"💾 Step 5: Creating temp file for {physical_nm}...")
                        
                        # Build file-specific metadata
                        file_metadata = base_metadata.copy()
                        file_metadata["physical_nm"] = physical_nm
                        file_metadata["url"] = file_url
                        file_metadata["file_index"] = file_index
                        file_metadata["is_multi_file"] = len(file_urls) > 1
                        file_metadata["detected_file_type"] = file_type  # Add early detection result
                        file_metadata["file_type_detection"] = "extension_validated"
                        
                        # Add content type if available
                        if content_type:
                            file_metadata["content_type"] = content_type
                        
                        # 🎯 STEP 6: HANDLER INSTANTIATION + PROCESSING (CPU + disk I/O)
                        logger.debug(f"🎯 Step 6: Processing {physical_nm} with {file_type} handler...")
                        
                        result = await process_file_content(
                            file_content=file_content,
                            filename=physical_nm,
                            metadata=file_metadata,
                            model_manager=self.model_manager,
                        )
                        
                        logger.info(f"✅ Step 6 Complete: Processed {physical_nm} → {result['status']}")
                        
                        # Add file tracking info to result
                        result["file_index"] = file_index
                        result["physical_nm"] = physical_nm
                        results.append(result)
                        
                        # Handle different result statuses
                        if result["status"] == "success":
                            processed_count += 1
                            
                            # Extract chunks for vector storage
                            extracted_chunks = result.get("chunks", [])
                            if extracted_chunks:
                                for i, chunk in enumerate(extracted_chunks):
                                    chunk_id = f"{document_type}_{document_id}_{member_id}_file{file_index}_{physical_nm}_chunk_{i}"
                                    if chunk_id in existing_ids:
                                        logger.warning(f"Skipping duplicate chunk ID: {chunk_id}")
                                        continue
                                        
                                    all_chunks.append(chunk)
                                    chunk_meta = file_metadata.copy()
                                    chunk_meta["chunk_index"] = i
                                    chunk_meta["chunk_count"] = len(extracted_chunks)
                                    chunk_meta["global_chunk_id"] = len(all_chunks)
                                    chunk_metadata.append(chunk_meta)
                                    chunk_ids.append(chunk_id)
                                    
                        elif result["status"] == "error":
                            logger.error(f"❌ File processing failed for {physical_nm}: {result.get('message', 'Unknown error')}")
                            
                        elif result["status"] == "warning":
                            logger.warning(f"⚠️ File processing warning for {physical_nm}: {result.get('message', 'Unknown warning')}")

                    except Exception as e:
                        logger.error(f"❌ Error processing file URL {file_url} for document {document_id}: {str(e)}", exc_info=True)
                        results.append({
                            "document_id": document_id,
                            "physical_nm": physical_nm,
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
                "all_files": all_physical_names,
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

    
    async def cleanup_duplicate_entries(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up duplicate entries in ChromaDB collection.
        """
        try:
            chromadb_collection = get_chromadb_collection()
            all_data = chromadb_collection.get()
            
            if not all_data or not all_data.get("ids"):
                return {"status": "success", "message": "No data to clean"}
            
            ids = all_data["ids"]
            metadatas = all_data.get("metadatas", [])
            
            # Group by logical_nm to find duplicates
            logical_nm_groups = {}
            for i, metadata in enumerate(metadatas):
                logical_nm = metadata.get("logical_nm", "unknown")
                if logical_nm not in logical_nm_groups:
                    logical_nm_groups[logical_nm] = []
                logical_nm_groups[logical_nm].append(i)
            
            # Find excessive duplicates (more than 5 copies)
            duplicates_to_remove = []
            for logical_nm, indices in logical_nm_groups.items():
                if len(indices) > 5:  # Keep first 2, remove rest
                    duplicates_to_remove.extend(indices[2:])
            
            if not duplicates_to_remove:
                return {"status": "success", "message": "No excessive duplicates found"}
            
            duplicate_ids = [ids[i] for i in duplicates_to_remove]
            
            if dry_run:
                return {
                    "status": "dry_run",
                    "message": f"Would delete {len(duplicate_ids)} duplicate entries",
                    "duplicate_count": len(duplicate_ids),
                    "sample_logical_names": list(set([metadatas[i].get("logical_nm", "unknown") for i in duplicates_to_remove[:10]]))
                }
            else:
                # Actually delete duplicates
                chromadb_collection.delete(ids=duplicate_ids)
                return {
                    "status": "success", 
                    "message": f"Deleted {len(duplicate_ids)} duplicate entries",
                    "deleted_count": len(duplicate_ids)
                }
                
        except Exception as e:
            logger.error(f"Error in cleanup_duplicate_entries: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    
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

    async def delete_by_physical_name(self, physical_nm: str, member_id: str) -> Dict[str, Any]:
        """Delete documents by physical name with member isolation."""
        try:
            chromadb_collection = get_chromadb_collection()
            
            logger.info(f"Attempting to delete documents with physical_nm: '{physical_nm}' for member: {member_id}")
            
            # Use $and operator for compound conditions
            results = chromadb_collection.get(
                where={
                    "$and": [
                        {"physical_nm": {"$eq": physical_nm}},
                        {"member_id": {"$eq": member_id}}
                    ]
                }
            )
            
            if not results["ids"]:
                logger.warning(f"No documents found with physical_nm: '{physical_nm}' for member: {member_id}")
                return {
                    "status": "not_found",
                    "message": f"No documents found with physical_nm: {physical_nm}",
                    "physical_nm": physical_nm,
                    "member_id": member_id,
                    "deleted_count": 0
                }
            
            # Log what we're about to delete for audit purposes
            logger.info(f"Found {len(results['ids'])} documents to delete: {results['ids'][:5]}...")  # Log first 5 IDs
            
            # Perform deletion
            chromadb_collection.delete(ids=results["ids"])
            
            # Verify deletion (optional safety check)
            verification = chromadb_collection.get(
                where={
                    "$and": [
                        {"physical_nm": {"$eq": physical_nm}},
                        {"member_id": {"$eq": member_id}}
                    ]
                }
            )
            
            if verification["ids"]:
                logger.error(f"Deletion verification failed - {len(verification['ids'])} documents still exist")
                return {
                    "status": "error",
                    "message": "Deletion verification failed - some documents may still exist"
                }
            
            logger.info(f"Successfully deleted {len(results['ids'])} documents for physical_nm: '{physical_nm}'")
            
            return {
                "status": "success",
                "message": f"Successfully deleted {len(results['ids'])} documents",
                "physical_nm": physical_nm,
                "member_id": member_id,
                "deleted_count": len(results["ids"]),
                "deleted_ids": results["ids"]  # For audit trail
            }
            
        except Exception as e:
            logger.error(f"Failed to delete physical_nm '{physical_nm}' for member {member_id}: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "message": f"Database error during deletion: {str(e)}",
                "physical_nm": physical_nm,
                "member_id": member_id
            }

    async def update_by_document_id(
    self, 
    target_document_id: int,  
    member_id: str,
    input_data: Dict[str, Any],
    file_urls: List[str] = [],
    physical_nm: List[str] = []
) -> Dict[str, Any]:
        """
        Smart update: Only processes files if they've changed, always updates metadata/content.
        Features transaction safety with rollback capability for metadata-only updates.
        """
        try:
            chromadb_collection = get_chromadb_collection()
            
            # Step 1: Find existing documents (ChromaDB stores document_id as string)
            existing_results = chromadb_collection.get(
                where={
                    "document_id": {"$eq": str(target_document_id)}
                }
            )
            
            if not existing_results["ids"]:
                return {
                    "status": "not_found",
                    "message": f"No documents found with document_id: {target_document_id}"
                }
            
            # Step 2: Extract existing metadata for comparison
            existing_metadata = existing_results["metadatas"][0] if existing_results["metadatas"] else {}
            existing_file_urls = []
            existing_physical_nm = []
            
            # Collect existing file information from all chunks
            for metadata in existing_results["metadatas"]:
                if metadata.get("url") and metadata["url"] not in existing_file_urls:
                    existing_file_urls.append(metadata["url"])
                if metadata.get("physical_nm") and metadata["physical_nm"] not in existing_physical_nm:
                    existing_physical_nm.append(metadata["physical_nm"])
            
            # Step 3: Check if files have changed
            files_changed = (
                set(file_urls) != set(existing_file_urls) or 
                set(physical_nm) != set(existing_physical_nm) or
                len(file_urls) != len(existing_file_urls)
            )
            
            # Step 4: Check if content/metadata has changed
            existing_content = existing_metadata.get("content", "")
            new_content = input_data.get("content", "")
            html_content_changed = existing_content != new_content

            existing_tags = existing_metadata.get("tags_str", "")
            new_tags = ",".join(input_data.get("tags", [])) if input_data.get("tags") else ""

            existing_doc_type = existing_metadata.get("document_type", "")
            new_doc_type = input_data.get("document_type", "")
            
            # Extract existing custom metadata (excluding system fields)
            existing_custom = {}
            new_custom = input_data.get("custom_metadata", {})
            for key, value in existing_metadata.items():
                if key not in ["document_id", "document_type", "tags_str", "content", "url", "physical_nm", "member_id", "uploaded_by", "created_at"]:
                    existing_custom[key] = value
            
            metadata_changed = (
                existing_tags != new_tags or
                existing_doc_type != new_doc_type or
                existing_custom != new_custom
            )
        
            logger.info(f"Change detection - Files changed: {files_changed}, HTML content changed: {html_content_changed}, Metadata changed: {metadata_changed}")
            
            # Step 5: If nothing changed, return early
            if not files_changed and not html_content_changed and not metadata_changed:
                return {
                    "status": "success",
                    "message": "No changes detected - document is already up to date",
                    "document_id": target_document_id,
                    "processed_count": 0,
                    "chunks_stored": 0,
                    "changes_detected": False
                }
            
            need_file_reprocessing = files_changed or html_content_changed
            need_metadata_update = metadata_changed
            
            if need_file_reprocessing:
                # Full reprocessing: delete existing chunks and re-ingest everything
                logger.info(f"Full reprocessing needed - deleting {len(existing_results['ids'])} existing chunks")
                chromadb_collection.delete(ids=existing_results["ids"])
                
                # Prepare for full re-ingestion
                updated_input_data = input_data.copy()
                updated_input_data["document_id"] = str(target_document_id)
                
                # Use provided document_type or preserve original
                if "document_type" not in updated_input_data:
                    updated_input_data["document_type"] = existing_metadata.get("document_type", "unknown")
                
                # Ensure custom_metadata exists
                if "custom_metadata" not in updated_input_data:
                    updated_input_data["custom_metadata"] = {}
                
                # Inject required metadata
                updated_input_data["custom_metadata"]["member_id"] = member_id
                updated_input_data["custom_metadata"]["uploaded_by"] = member_id
                updated_input_data["custom_metadata"]["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                # Track what changed
                update_reasons = []
                if files_changed:
                    update_reasons.append("files_modified")
                if html_content_changed:
                    update_reasons.append("html_content_modified")
                updated_input_data["custom_metadata"]["update_reason"] = ",".join(update_reasons)
                
                # Inject physical_nm if provided
                if physical_nm:
                    updated_input_data["custom_metadata"]["physical_nm"] = physical_nm
                
                # Re-ingest everything
                enhanced_input_data = json.dumps(updated_input_data)
                result = await self.process_direct_uploads_with_urls(enhanced_input_data, file_urls)
                
                if result["status"] == "success":
                    return {
                        "status": "success",
                        "message": f"Successfully updated document {target_document_id} (full reprocessing)",
                        "document_id": target_document_id,
                        "processed_count": result.get("processed_count", 0),
                        "chunks_stored": result.get("chunks_stored", 0),
                        "changes_detected": True,
                        "files_changed": files_changed,
                        "html_content_changed": html_content_changed,
                        "metadata_changed": metadata_changed,
                        "update_type": "full_reprocessing"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to update: {result.get('message', 'Unknown error')}"
                    }
                    
            elif need_metadata_update:
                logger.info(f"Metadata-only update - updating {len(existing_results['ids'])} chunks in place")
                
                # Validate data integrity before proceeding
                if not existing_results["ids"] or not existing_results["documents"]:
                    logger.error("Missing required data for metadata update")
                    return {
                        "status": "error",
                        "message": "Cannot perform metadata update: missing chunk data"
                    }
                
                if len(existing_results["ids"]) != len(existing_results["documents"]):
                    logger.error("Data integrity issue: mismatched IDs and documents count")
                    return {
                        "status": "error", 
                        "message": "Data integrity issue detected - aborting update"
                    }
                
                # Store original data for potential rollback
                original_ids = existing_results["ids"]
                original_documents = existing_results["documents"]
                original_metadatas = existing_results["metadatas"]
                
                logger.info(f"Backup created: {len(original_ids)} chunks backed up for rollback")
                
                try:
                    # Delete existing chunks
                    chromadb_collection.delete(ids=original_ids)
                    logger.info(f"Deleted {len(original_ids)} existing chunks")
                    
                    # Prepare updated metadata
                    new_tags_str = ",".join(input_data.get("tags", [])) if input_data.get("tags") else ""
                    new_doc_type = input_data.get("document_type", existing_metadata.get("document_type", "unknown"))
                    
                    # Build updated metadata
                    updated_metadata = {}
                    for key, value in existing_metadata.items():
                        updated_metadata[key] = value
                    
                    # Update changed fields
                    updated_metadata["tags_str"] = new_tags_str
                    updated_metadata["document_type"] = new_doc_type
                    updated_metadata["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    updated_metadata["update_reason"] = "metadata_only"
                    
                    # Apply custom_metadata updates
                    if input_data.get("custom_metadata"):
                        for key, value in input_data["custom_metadata"].items():
                            if key not in ["member_id", "uploaded_by"]:  # Preserve original ownership
                                updated_metadata[key] = value
                    
                    # Process and re-add chunks
                    all_chunks = []
                    chunk_metadata = []
                    chunk_ids_new = []
                    
                    for i, (chunk_id, chunk_content) in enumerate(zip(original_ids, original_documents)):
                        all_chunks.append(chunk_content)
                        chunk_meta = updated_metadata.copy()
                        chunk_meta["chunk_index"] = i
                        chunk_meta["chunk_count"] = len(original_documents)
                        chunk_metadata.append(chunk_meta)
                        chunk_ids_new.append(chunk_id)  # Keep same IDs
                    
                    # Process in batches with enhanced error handling and progress logging
                    chunks_stored = 0
                    batch_size = 20
                    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
                    
                    logger.info(f"Starting metadata update: {len(all_chunks)} chunks in {total_batches} batches")
                    
                    for batch_idx in range(0, len(all_chunks), batch_size):
                        batch_num = batch_idx // batch_size + 1
                        batch_chunks = all_chunks[batch_idx:batch_idx + batch_size]
                        batch_ids = chunk_ids_new[batch_idx:batch_idx + batch_size]
                        batch_meta = chunk_metadata[batch_idx:batch_idx + batch_size]
                        
                        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                        
                        try:
                            # Generate embeddings
                            async with self.embedding_semaphore:
                                embed_tasks = [self._embed_text(chunk, batch_meta[idx]) for idx, chunk in enumerate(batch_chunks)]
                                embeddings_results = await asyncio.gather(*embed_tasks, return_exceptions=True)
                                
                                embeddings = []
                                valid_indices = []
                                failed_count = 0
                                
                                for idx, result in enumerate(embeddings_results):
                                    if (not isinstance(result, Exception) and 
                                        result is not None and 
                                        len(result) == 2 and 
                                        result[0] is not None):
                                        embeddings.append(result[0])
                                        valid_indices.append(idx)
                                    else:
                                        failed_count += 1
                                        if isinstance(result, Exception):
                                            logger.warning(f"Embedding failed for chunk in batch {batch_num}: {result}")
                            
                            if embeddings:
                                async with self.chroma_lock:
                                    chromadb_collection.add(
                                        ids=[batch_ids[idx] for idx in valid_indices],
                                        embeddings=embeddings,
                                        documents=[batch_chunks[idx] for idx in valid_indices],
                                        metadatas=[batch_meta[idx] for idx in valid_indices]
                                    )
                                    chunks_stored += len(embeddings)
                                    
                                logger.info(f"Batch {batch_num}/{total_batches} complete: {len(embeddings)} stored, {failed_count} failed")
                            else:
                                logger.warning(f"Batch {batch_num}/{total_batches} produced no valid embeddings")
                                    
                        except Exception as batch_error:
                            logger.error(f"Batch {batch_num}/{total_batches} processing failed: {batch_error}")
                            # Continue with next batch rather than failing completely
                            continue
                    
                    logger.info(f"Metadata update complete: {chunks_stored}/{len(all_chunks)} chunks successfully stored")
                    
                    return {
                        "status": "success",
                        "message": f"Successfully updated document {target_document_id} (metadata only)",
                        "document_id": target_document_id,
                        "processed_count": 0,  # No new files processed
                        "chunks_stored": chunks_stored,
                        "changes_detected": True,
                        "files_changed": False,
                        "html_content_changed": False,
                        "metadata_changed": True,
                        "update_type": "metadata_only"
                    }
                    
                except Exception as update_error:
                    logger.error(f"Metadata update failed, attempting rollback: {update_error}")
                    
                    # Attempt to restore original data with batch processing
                    try:
                        rollback_batch_size = 20
                        rollback_chunks_restored = 0
                        total_rollback_batches = (len(original_documents) + rollback_batch_size - 1) // rollback_batch_size
                        
                        logger.info(f"Starting rollback: {len(original_documents)} chunks in {total_rollback_batches} batches")
                        
                        for i in range(0, len(original_documents), rollback_batch_size):
                            batch_num = i // rollback_batch_size + 1
                            batch_docs = original_documents[i:i + rollback_batch_size]
                            batch_ids = original_ids[i:i + rollback_batch_size]
                            batch_metas = original_metadatas[i:i + rollback_batch_size]
                            
                            logger.debug(f"Rollback batch {batch_num}/{total_rollback_batches}")
                            
                            try:
                                # Re-generate embeddings for this batch
                                async with self.embedding_semaphore:
                                    rollback_embed_tasks = [
                                        self._embed_text(doc, meta) 
                                        for doc, meta in zip(batch_docs, batch_metas)
                                    ]
                                    rollback_results = await asyncio.gather(*rollback_embed_tasks, return_exceptions=True)
                                    
                                    # Filter valid embeddings
                                    valid_rollback_embeddings = []
                                    valid_rollback_indices = []
                                    
                                    for idx, result in enumerate(rollback_results):
                                        if (not isinstance(result, Exception) and 
                                            result is not None and 
                                            len(result) == 2 and 
                                            result[0] is not None):
                                            valid_rollback_embeddings.append(result[0])
                                            valid_rollback_indices.append(idx)
                                
                                # Restore this batch if we have valid embeddings
                                if valid_rollback_embeddings:
                                    async with self.chroma_lock:
                                        chromadb_collection.add(
                                            ids=[batch_ids[idx] for idx in valid_rollback_indices],
                                            embeddings=valid_rollback_embeddings,
                                            documents=[batch_docs[idx] for idx in valid_rollback_indices],
                                            metadatas=[batch_metas[idx] for idx in valid_rollback_indices]
                                        )
                                        rollback_chunks_restored += len(valid_rollback_embeddings)
                                        
                                    logger.info(f"Rollback batch {batch_num}/{total_rollback_batches} complete: {len(valid_rollback_embeddings)} restored")
                                else:
                                    logger.warning(f"Rollback batch {batch_num}/{total_rollback_batches} failed - no valid embeddings")
                                    
                            except Exception as rollback_batch_error:
                                logger.error(f"Rollback batch {batch_num} failed: {rollback_batch_error}")
                                continue
                        
                        logger.info(f"Rollback completed: restored {rollback_chunks_restored}/{len(original_ids)} chunks")
                        
                        rollback_status = "success" if rollback_chunks_restored == len(original_ids) else "partial_success"
                        
                        return {
                            "status": "error",
                            "message": f"Metadata update failed but {rollback_chunks_restored}/{len(original_ids)} chunks were restored",
                            "rollback_status": rollback_status,
                            "chunks_restored": rollback_chunks_restored,
                            "chunks_lost": len(original_ids) - rollback_chunks_restored,
                            "original_error": str(update_error)
                        }
                            
                    except Exception as rollback_error:
                        logger.error(f"Rollback failed: {rollback_error}")
                        
                        return {
                            "status": "error",
                            "message": f"CRITICAL: Metadata update failed AND rollback failed. Data may be lost!",
                            "original_error": str(update_error),
                            "rollback_error": str(rollback_error),
                            "chunks_potentially_lost": len(original_ids),
                            "document_id": target_document_id
                        }
            
        except Exception as e:
            logger.error(f"Failed to update document_id {target_document_id}: {str(e)}", exc_info=True)
            return {
                "status": "error", 
                "message": f"Update operation failed: {str(e)}",
                "document_id": target_document_id
            }


    async def delete_by_document_id(self, document_id: int, member_id: str) -> Dict[str, Any]:
        """Delete documents by document_id (bigint) with member isolation."""
        try:
            chromadb_collection = get_chromadb_collection()
            
            # Find documents by document_id + member_id (convert int to string for ChromaDB)
            results = chromadb_collection.get(
                where={
                    
                        "document_id": {"$eq": str(document_id)}  # Convert to string for ChromaDB
                        
                    
                }
            )
            
            if not results["ids"]:
                return {
                    "status": "not_found",
                    "message": f"No documents found with document_id: {document_id}",
                    "document_id": document_id,
                    "deleted_count": 0
                }
            
            # Delete all documents with this document_id
            chromadb_collection.delete(ids=results["ids"])
            
            logger.info(f"Deleted {len(results['ids'])} documents for document_id: {document_id}")
            
            return {
                "status": "success",
                "message": f"Successfully deleted {len(results['ids'])} documents",
                "document_id": document_id,  # Return as int
                "deleted_count": len(results["ids"])
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document_id {document_id}: {str(e)}")
            return {"status": "error", "message": str(e)}


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