# src/core/services/file_utils.py
import os
import re
from pathlib import Path
from src.core.file_handlers.factory import FileHandlerFactory
from src.core.file_handlers.pdf_handler import PDFHandler
from src.core.file_handlers.doc_handler import AdvancedDocHandler
from src.core.file_handlers.hwp_handler import HWPHandler
from src.core.file_handlers.image_handler import ImageHandler
from src.core.file_handlers.msg_handler import MSGHandler
from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
from src.core.utils.file_identification import get_file_type
from src.core.utils.post_processing import clean_extracted_text as clean_text
from src.core.utils.text_chunking import chunk_with_metadata
from src.core.utils.text_chunking import chunk_text
import chromadb
import logging
import ollama
from typing import List, Optional, Dict, Any, Set
import tempfile
from concurrent.futures import ThreadPoolExecutor


import logging
logger = logging.getLogger(__name__)

# Define directories.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(os.getcwd(), "sample_data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db1")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Shared globals (updated at startup)
class GlobalState:
    def __init__(self):
        self._global_prompt = None
        self._rag_chain = None
        self._vector_store = None
        self._chromadb_collection = None
        self._workflow = None
        self._memory = None

    @property
    def chromadb_collection(self):
        return self._chromadb_collection

    @chromadb_collection.setter
    def chromadb_collection(self, value):
        self._chromadb_collection = value

    @property
    def rag_chain(self):
        return self._rag_chain

    @rag_chain.setter
    def rag_chain(self, value):
        self._rag_chain = value

    @property
    def vector_store(self):
        return self._vector_store

    @vector_store.setter
    def vector_store(self, value):
        self._vector_store = value

    @property
    def global_prompt(self):
        return self._global_prompt

    @global_prompt.setter
    def global_prompt(self, value):
        self._global_prompt = value

    @property
    def workflow(self): 
        return self._workflow

    @workflow.setter
    def workflow(self, value):
        self._workflow = value

    @property
    def memory(self): 
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

_state = GlobalState()

# Getters for external access
def get_chromadb_collection():
    if _state.chromadb_collection is None:
        logger.warning("ChromaDB collection not initialized. Initializing now...")
        initialize_chromadb_collection()
    return _state.chromadb_collection

def get_rag_chain():
    return _state.rag_chain

def get_vector_store():
    return _state.vector_store

def get_global_prompt():
    return _state.global_prompt

def get_workflow():
    return _state.workflow

def get_memory():
    return _state.memory

def set_globals(chroma_coll, rag, vect_store, prompt, workflow, memory):
    """Update module-level globals with proper debugging."""
    try:
        logger.debug(f"Setting globals with ChromaDB collection: {chroma_coll}")
        
        # Update the global state
        _state.chromadb_collection = chroma_coll
        _state.rag_chain = rag
        _state.vector_store = vect_store
        _state.global_prompt = prompt
        _state.workflow = workflow
        _state.memory = memory

        # Verify the global state
        logger.debug(f"Global ChromaDB collection after set: {_state.chromadb_collection is not None}")
        logger.debug(f"Global vector store after set: {_state.vector_store is not None}")
        logger.debug(f"Global RAG chain after set: {_state.rag_chain is not None}")
        logger.debug(f"Global vector store after set: {_state.vector_store is not None}")
        logger.debug(f"Global prompt after set: {_state.global_prompt is not None}")

        if _state.chromadb_collection is None:
            logger.error("[ERROR] ChromaDB collection is NOT initialized")
            return False
            
        # Test the ChromaDB collection
        try:
            count = _state.chromadb_collection.count()
            logger.info(f"[Success] ChromaDB collection initialized with {count} documents")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to access ChromaDB collection: {e}")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to set globals: {e}")
        return False

def initialize_chromadb_collection():
    """Initialize the ChromaDB collection if it doesn't exist."""
    try:
        persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
        chroma_coll = persistent_client.get_or_create_collection(
            name="netbackup_docs",
            metadata={"hnsw:space": "cosine"}
        )
        _state.chromadb_collection = chroma_coll
        logger.info(f"Initialized ChromaDB collection at {CHROMA_DIR}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {str(e)}")
        _state.chromadb_collection = None


# Make chromadb_collection accessible through a property
@property
def chromadb_collection():
    return _state.chromadb_collection

def flatten_embedding(embedding):
    """Flatten nested embeddings."""
    while isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding

def clean_extracted_text(text: str) -> str:
    cleaned = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", text)
    return cleaned.strip()

def process_file_content(file_content: bytes, filename: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=200, model_manager=None) -> Dict[str, Any]:
    """
    Process file content using a temporary file on disk.
    """
    temp_file = None
    temp_file_path = None
    try:
        # Ensure file_content is bytes
        if not isinstance(file_content, bytes):
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
                logger.warning(f"Converted string input to bytes for {filename}")
            else:
                logger.error(f"Invalid file_content type for {filename}: {type(file_content)}. Expected bytes.")
                raise TypeError(f"Invalid file_content type for {filename}: {type(file_content)}. Expected bytes.")

        # Determine the file type from the content
        file_type = get_file_type(file_content)
        if not file_type:
            logger.error(f"Could not determine file type for {filename}")
            raise ValueError(f"Could not determine file type for {filename}")

        # If the content type is text/plain, it may be an error page instead of the expected binary file.
        if file_type == "text/plain":
            error_text = file_content.decode('utf-8', errors='ignore')
            if "<Error>" in error_text:
                logger.error(f"File {filename} appears to be an error page: {error_text[:200]}")
                raise ValueError("Downloaded file content is an error page; invalid credentials or expired URL.")

        # Choose handler based on file type
        if file_type.startswith("image/"):
            handler = ImageHandler(model_manager=model_manager) if model_manager else ImageHandler()
        elif file_type == "application/pdf":
            handler = PDFHandler(model_manager=model_manager) if model_manager else PDFHandler()
        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            handler = AdvancedDocHandler(model_manager=model_manager) if model_manager else AdvancedDocHandler()
        elif file_type == "application/x-hwp":
            handler = HWPHandler(model_manager=model_manager) if model_manager else HWPHandler()
        elif file_type == "application/vnd.ms-outlook":
            handler = MSGHandler(model_manager=model_manager) if model_manager else MSGHandler()
        else:
            logger.error(f"Unsupported file type for {filename}: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        # Write file content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
        temp_file.write(file_content)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file.close()

        # Extract text using the chosen handler
        text = handler.extract_text(temp_file_path)

        if not text or not text.strip():
            logger.warning(f"No text extracted from {filename} (path: {temp_file_path})")
            return {"filename": filename, "status": "error", "message": "No text extracted from file"}

        cleaned_text = clean_extracted_text(text)

        # Split text into chunks for embedding
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"No chunks extracted for {filename}: {cleaned_text}")
            return {"filename": filename, "status": "error", "message": "No chunks extracted from text"}

        chromadb_collection = get_chromadb_collection()
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            if chunk_id in set(chromadb_collection.get()["ids"]):
                logger.info(f"Skipping duplicate chunk ID: {chunk_id}")
                continue

            chunk_metadata = {**metadata, "chunk_index": i, "chunk_count": len(chunks)}
            embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
            embedding = embed_response.get("embedding") or embed_response.get("embeddings")
            if embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {i} of {filename}. Skipping.")
                continue

            embedding = flatten_embedding(embedding)
            chromadb_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            chunk_ids.append(chunk_id)

        logger.info(f"Successfully processed {filename} with {len(chunk_ids)} chunks")
        return {
            "filename": filename,
            "status": "success",
            "message": f"File processed successfully with {len(chunk_ids)} chunks",
            "chunk_count": len(chunk_ids)
        }
    except Exception as e:
        logger.error(f"Error processing file content for {filename}: {e}", exc_info=True)
        return {
            "filename": filename,
            "status": "error",
            "message": f"Failed to process file: {str(e)}"
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")

def process_file(file_path: str, chunk_size=1000, chunk_overlap=200, model_manager=None):
    """
    Process a file and extract text, chunks, tables, and status codes.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks for text splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dict with extracted text, chunks, tables, and status codes
    """
    try:
        file_type = get_file_type(file_path)
        
        # Select appropriate handler based on file type
        if file_type.startswith("image/"):
            handler = ImageHandler(model_manager=model_manager,languages=['ko', 'en'])
        elif file_type == "application/pdf":
            handler = PDFHandler(model_manager=model_manager,)
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            handler = AdvancedDocHandler(model_manager=model_manager,)
        elif file_type == "application/x-hwp":
            handler = HWPHandler(model_manager=model_manager,)
        elif file_type == "application/vnd.ms-outlook":
            handler = MSGHandler(model_manager=model_manager,)    
        else:
            logger.error(f"Unsupported file type for {file_path}: {file_type}")
            return {"text": "", "chunks": [], "tables": [], "status_codes": []}
            
        # Extract text
        text = handler.extract_text(file_path)
        tables = handler.extract_tables(file_path) if hasattr(handler, "extract_tables") else []
        
        # # Get status codes if available
        # status_codes = handler.get_status_codes() if hasattr(handler, "get_status_codes") else []
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Create chunks
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap) if chunk_size > 0 else [cleaned_text]
        
        return {
            "text": cleaned_text,
            "chunks": chunks,
            "tables": tables
            
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            "text": "",
            "chunks": [],
            "tables": []
            
        }
    

def process_file_from_server(file_content: bytes, filename: str, metadata: dict, chunk_size=1000, chunk_overlap=200):
    """
    Process file content directly from server API response **without saving to disk**.

    Args:
        file_content: The binary content of the file.
        filename: The original name of the file.
        metadata: Additional metadata (e.g., error_code_id, client_name).
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Overlap between chunks.

    Returns:
        Dict containing processing results.
    """
    try:
        file_type = get_file_type(file_content)  # Pass the file content (bytes) directly

        # Select the appropriate handler
        if file_type.startswith("image/"):
            handler = ImageHandler()
        elif file_type == "application/pdf":
            handler = PDFHandler()
        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            handler = AdvancedDocHandler()
        elif file_type == "application/x-hwp":
            handler = HWPHandler()
        elif file_type == "application/vnd.ms-outlook":
            handler = MSGHandler()    
        else:
            logger.error(f"Unsupported file type for {filename}: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        # Extract text from the file content
        text = handler.extract_text_from_memory(file_content)
        if not text or not text.strip():
            logger.warning(f"No text extracted from {filename}")
            return {
                "filename": filename,
                "status": "error",
                "message": "No text extracted from file."
            }
        cleaned_text = clean_extracted_text(text)

        # Chunk the extracted text
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"Failed to chunk text for {filename}")
            return {
                "filename": filename,
                "status": "error",
                "message": "Failed to chunk extracted text."
            }

        # Store chunks in ChromaDB
        chromadb_collection = get_chromadb_collection()
        if chromadb_collection is None:
            logger.error("ChromaDB collection is not initialized")
            raise RuntimeError("ChromaDB collection is not initialized.")

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            if chunk_id in set(chromadb_collection.get()["ids"]):
                logger.info(f"Skipping duplicate chunk ID: {chunk_id}")
                continue
            chunk_metadata = {**metadata, "chunk_index": i, "chunk_count": len(chunks)}

            # Generate embedding
            embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
            embedding = embed_response.get("embedding") or embed_response.get("embeddings")
            if embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {i} of {filename}. Skipping.")
                continue
            embedding = flatten_embedding(embedding)

            # Add to ChromaDB
            chromadb_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            chunk_ids.append(chunk_id)

        logger.info(f"Successfully processed {filename} with {len(chunk_ids)} chunks")
        return {
            "filename": filename,
            "status": "success",
            "message": f"File processed successfully with {len(chunk_ids)} chunks.",
            "chunk_count": len(chunk_ids)
        }

    except Exception as e:
        logger.error(f"Error processing file '{filename}': {str(e)}", exc_info=True)
        return {
            "filename": filename,
            "status": "error",
            "message": f"Failed to process file: {str(e)}"
        }


def load_documents_to_chroma(pdf_handler, doc_handler, hwp_handler, msg_handler=None):
    """
    Walk DATA_FOLDER and add supported documents to the Chroma collection.
    """
    chroma_coll = get_chromadb_collection()
    if chroma_coll is None:
        raise RuntimeError("ChromaDB collection not initialized")

    try:
        initial_count = chroma_coll.count()
        print(f"Initial document count: {initial_count}")
    except Exception as e:
        raise RuntimeError(f"Cannot access ChromaDB collection: {e}")

    supported_extensions = {".pdf", ".doc", ".docx", ".hwp", ".png", ".jpg", ".jpeg", ".msg"}
    print(f"Loading documents from: {DATA_FOLDER}")

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Extract text using appropriate handler
                if ext == ".pdf":
                    text = pdf_handler.extract_text(file_path)
                elif ext == ".hwp": 
                    text = hwp_handler.extract_text(file_path)
                elif ext in [".png", ".jpg", ".jpeg"]:
                    image_handler = ImageHandler()
                    text = image_handler.extract_text(file_path)
                elif ext == ".msg" and msg_handler:
                    text = msg_handler.extract_text(file_path)
                else:
                    text = doc_handler.extract_text(file_path)
                
                if text and text.strip():
                    cleaned_text = clean_extracted_text(text)
                    
                    # Create base metadata
                    base_metadata = {
                        "source_id": file_path,
                        "filename": file,
                        "file_type": ext[1:],
                    }
                    
                    # Chunk the document without status code
                    chunks = chunk_with_metadata(
                        cleaned_text, 
                        base_metadata,
                        chunk_size=1000,  # Adjust as needed
                        chunk_overlap=200  # Adjust as needed
                    )
                    
                    for i, chunk_data in enumerate(chunks):
                        chunk_text = chunk_data["text"]
                        chunk_metadata = chunk_data["metadata"]
                        
                        chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"
                        
                        try:
                            embed_response = ollama.embed(model="mxbai-embed-large", input=chunk_text)
                            embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                            if embedding is None:
                                logger.error(f"Failed to get embedding for chunk {i} of {file_path}")
                                continue
                            embedding = flatten_embedding(embedding)
                            
                            chroma_coll.add(
                                ids=[chunk_id],
                                embeddings=[embedding],
                                documents=[chunk_text],
                                metadatas=[chunk_metadata]
                            )
                            logger.info(f"Added chunk {i} of document {file_path}")
                        except Exception as e:
                            logger.error(f"Error adding chunk {i} of document {file_path}: {str(e)}")
    
    try:
        count = chroma_coll.count()
    except AttributeError:
        count = chroma_coll._collection.count()
    print(f"Chroma collection now contains {count} documents.")

def stream_llama_response(prompt: str):
    """
    Call the Ollama model via command line and yield its output line-by-line.
    """
    import subprocess
    cmd = ["ollama", "run", "deepseek-r1:8b"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    process.stdin.write(prompt + "\n")
    process.stdin.flush()
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            yield line
    finally:
        process.stdout.close()
        process.wait()


def get_file_handler(file_extension_or_mime_type):
    """
    Get the appropriate file handler based on file extension or MIME type.
    
    Args:
        file_extension_or_mime_type: File extension or MIME type
        
    Returns:
        An instance of the appropriate FileHandler
    """
    # Check if input is a MIME type
    if '/' in file_extension_or_mime_type:
        return FileHandlerFactory.get_handler(file_extension_or_mime_type)
    else:
        # Assume it's a file extension
        return FileHandlerFactory.get_handler_for_extension(file_extension_or_mime_type)

def process_html_content(html_content: str, metadata: Dict[str, Any], html_handler: HTMLContentHandler, 
                        vision_extractor: GraniteVisionExtractor, chunk_size=1000, chunk_overlap=200) -> Dict[str, Any]:
    """
    Process HTML content with embedded images in batch, deduplicate text, and store in ChromaDB.
    
    Args:
        html_content: Raw HTML content from MariaDB
        metadata: Metadata with error_code_id, client_name, os_version
        html_handler: Initialized HTMLContentHandler instance
        vision_extractor: Initialized GraniteVisionExtractor instance
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between chunks
    
    Returns:
        Dict with processing status and details
    """
    try:
        report_id = metadata.get('resolve_id', 'unknown')
        base_url = metadata.get('url', None)
        if base_url:
            html_handler.base_url = base_url

        # Log the raw HTML content for debugging
        logger.debug(f"Raw HTML content for report {report_id}: {html_content[:200]}... (truncated)")

        # Extract HTML text and images
        result = html_handler.process_html(html_content, download_images=True, extract_image_text=False)
        html_text = result['html_text']
        logger.debug(f"Extracted HTML text for report {report_id}: {html_text[:200]}... (truncated)")
        logger.debug(f"Found {len(result['images'])} images in HTML content for report {report_id} with URLs: {[img['src'] for img in result['images']]}")

        if not html_text.strip():
            logger.warning(f"No text extracted from HTML content: {report_id}")
            # Proceed to check images even if HTML text is empty
            if not result['images']:
                logger.info(f"No embedded images found in HTML content for report {report_id}, relying on attachments if available")
                return {"report_id": report_id, "status": "warning", "message": "No text extracted from HTML and no images found"}

        # Batch process images
        image_texts: List[str] = []
        image_urls: Set[str] = set()  # Track unique image URLs to avoid duplicates

        def extract_text_from_image(img_info):
            img_src = img_info['src']
            if img_src in image_urls:
                logger.debug(f"Skipping duplicate image URL: {img_src}")
                return None
            image_urls.add(img_src)
            image_content = html_handler._download_image(img_src)
            if image_content:
                text = vision_extractor.extract_text_from_bytes(image_content)
                logger.debug(f"Extracted text from image {img_src}: {text[:100]}... (truncated)")
                return text
            else:
                logger.warning(f"Failed to download image content for {img_src}")
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            image_texts = list(executor.map(extract_text_from_image, result['images']))

        # Combine unique text, filtering out None and empty strings
        combined_text = [html_text] + [text for text in image_texts if text and text.strip()]
        logger.debug(f"Combined text components for report {report_id}: {len(combined_text)} parts")
        full_text = "\n".join(combined_text)
        cleaned_text = clean_extracted_text(full_text)
        logger.debug(f"Cleaned full text for report {report_id}: {cleaned_text[:200]}... (truncated)")

        # Chunk the text
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"No chunks created for report: {report_id}")
            return {"report_id": report_id, "status": "warning", "message": "No chunks created from combined text"}

        # Store in ChromaDB
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error(f"ChromaDB collection not initialized for report {report_id}")
            return {"report_id": report_id, "status": "error", "message": "ChromaDB collection not initialized"}

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{report_id}_chunk_{i}"
            if chunk_id in set(chromadb_collection.get()["ids"]):
                logger.info(f"Skipping duplicate chunk ID: {chunk_id}")
                continue

            chunk_metadata = {**metadata, "chunk_index": i, "chunk_count": len(chunks)}
            embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
            embedding = embed_response.get("embedding") or embed_response.get("embeddings")
            if embedding is None:
                logger.warning(f"Failed to embed chunk {i} for report {report_id}")
                continue

            embedding = flatten_embedding(embedding)
            chromadb_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            chunk_ids.append(chunk_id)

        logger.info(f"Processed HTML report {report_id} with {len(chunk_ids)} chunks")
        return {
            "report_id": report_id,
            "status": "success",
            "message": f"HTML content processed with {len(chunk_ids)} chunks",
            "chunk_count": len(chunk_ids)
        }

    except Exception as e:
        logger.error(f"Error processing HTML content for report {report_id}: {e}", exc_info=True)
        return {"report_id": report_id, "status": "error", "message": f"Failed to process: {str(e)}"}