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
from src.core.utils.file_identification import get_file_type
from src.core.utils.post_processing import clean_extracted_text as clean_text
from src.core.utils.text_chunking import chunk_with_metadata
from src.core.utils.text_chunking import chunk_text
import chromadb
import logging
import ollama
from typing import List, Optional, Dict, Any
import tempfile


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

def process_file_content(file_content: bytes, filename: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=200):
    """
    Process file content using a temporary file on disk.
    """
    try:
        file_type = get_file_type(filename)
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        text = handler.extract_text(temp_file_path)
        os.unlink(temp_file_path)

        if not text or not text.strip():
            logger.warning(f"No text extracted from {filename} (path: {temp_file_path})")
            return {
                "filename": filename,
                "status": "error",
                "message": "No text extracted from file"
            }
        cleaned_text = clean_extracted_text(text)

        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"No chunks extracted for {filename}: {cleaned_text}")
            return {
                "filename": filename,
                "status": "error",
                "message": "No chunks extracted from text"
            }

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

def process_file(file_path: str, chunk_size=1000, chunk_overlap=200):
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
            handler = ImageHandler(languages=['ko', 'en'])
        elif file_type == "application/pdf":
            handler = PDFHandler()
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            handler = AdvancedDocHandler()
        elif file_type == "application/x-hwp":
            handler = HWPHandler()
        elif file_type == "application/vnd.ms-outlook":
            handler = MSGHandler()    
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Extract text
        text = handler.extract_text(file_path)
        tables = handler.extract_tables(file_path) if hasattr(handler, "extract_tables") else []
        
        # Get status codes if available
        status_codes = handler.get_status_codes() if hasattr(handler, "get_status_codes") else []
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Create chunks
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap) if chunk_size > 0 else [cleaned_text]
        
        return {
            "text": cleaned_text,
            "chunks": chunks,
            "tables": tables,
            "status_codes": status_codes
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            "text": "",
            "chunks": [],
            "tables": [],
            "status_codes": []
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



# def load_documents_to_chroma2(pdf_handler, doc_handler, hwp_handler):
#     """
#     Walk DATA_FOLDER and add supported documents to the Chroma collection.
#     """
#     chroma_coll=get_chromadb_collection()
#     if chroma_coll is None:
#         raise RuntimeError("ChromaDB collection not initialized")

#     try:
#         initial_count = chroma_coll.count()
#         print(f"Initial document count: {initial_count}")
#     except Exception as e:
#         raise RuntimeError(f"Cannot access ChromaDB collection: {e}")

#     supported_extensions = {".pdf", ".doc", ".docx", ".hwp",".png",".jpg",".jpeg"}
#     print(f"Loading documents from: {DATA_FOLDER}")

#     import ollama
#     for root, dirs, files in os.walk(DATA_FOLDER):
#         for file in files:
#             ext = Path(file).suffix.lower()
#             if ext in supported_extensions:
#                 file_path = os.path.join(root, file)
#                 print(f"Processing file: {file_path}")
#                 if ext == ".pdf":
#                     text = pdf_handler.extract_text(file_path)
#                     status_codes=pdf_handler.get_status_codes()
#                 elif ext == ".hwp": 
#                     text=hwp_handler.extract_text(file_path)
#                     status_codes=hwp_handler.get_status_codes()
#                 elif ext in [".png", ".jpg", ".jpeg"]:  # Handle image files
#                     image_handler = ImageHandler()
#                     text = image_handler.extract_text(file_path)
#                     status_codes = image_handler.get_status_codes()
#                 else:
#                     text = doc_handler.extract_text(file_path)
#                     status_codes=doc_handler.get_status_codes()
#                 if text and text.strip():
#                     cleaned_text = clean_extracted_text(text)
#                     try:
#                         embed_response = ollama.embed(model="mxbai-embed-large", input=cleaned_text)
#                         embedding = embed_response.get("embedding") or embed_response.get("embeddings")
#                         if embedding is None:
#                             print(f"Failed to get embedding for {file_path}")
#                             continue
#                         embedding = flatten_embedding(embedding)
#                     except Exception as e:
#                         print(f"Error during embedding for {file_path}: {e}")
#                         continue

#                     base_metadata = {
#                         "id": file_path,
#                         "filename": file,
#                         "file_type": ext[1:],  
#                     }
#                     if status_codes:
#                         for code in status_codes:
#                             metadata=base_metadata.copy()
#                             metadata["status_code"]=str(code)
                         
#                             try:
#                                 doc_id=f"{file_path}_{code}"
#                                 chroma_coll.add(
#                                 #ids=[file_path],
#                                 ids=[doc_id],  # Unique ID for each status code
#                                 embeddings=[embedding],
#                                 documents=[cleaned_text],
#                                 metadatas=[metadata]
#                             )
#                                 print(f"Added document to Chroma: {doc_id}")
#                             except Exception as e:
#                                 print(f"Error adding document with status code {code} to Chroma: {e}")
#                     else:
#                         try:
#                             chroma_coll.add(
#                                 ids=[file_path],
#                                 embeddings=[embedding],
#                                 documents=[cleaned_text],
#                                 metadatas=[base_metadata]
#                             )
#                             print(f"Added document to Chroma: {file_path}")
#                         except Exception as e:
#                             print(f"Error Processing {file_path}:{e}")
#     try:
#         count = chroma_coll.count()
#     except AttributeError:
#         count = chroma_coll._collection.count()
#     print(f"Chroma collection now contains {count} documents.")


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
                    status_codes = pdf_handler.get_status_codes()
                elif ext == ".hwp": 
                    text = hwp_handler.extract_text(file_path)
                    status_codes = hwp_handler.get_status_codes()
                elif ext in [".png", ".jpg", ".jpeg"]:
                    image_handler = ImageHandler()
                    text = image_handler.extract_text(file_path)
                    status_codes = image_handler.get_status_codes()
                elif ext == ".msg" and msg_handler:
                    text = msg_handler.extract_text(file_path)
                    status_codes = msg_handler.get_status_codes()
                else:
                    text = doc_handler.extract_text(file_path)
                    status_codes = doc_handler.get_status_codes()
                
                if text and text.strip():
                    cleaned_text = clean_extracted_text(text)
                    
                    # Create base metadata
                    base_metadata = {
                        "source_id": file_path,
                        "filename": file,
                        "file_type": ext[1:],
                    }
                    
                    # Process with or without status codes
                    if status_codes:
                        for code in status_codes:
                            # Add status code to metadata
                            code_metadata = base_metadata.copy()
                            code_metadata["status_code"] = str(code)
                            
                            # Chunk the document with metadata
                            chunks = chunk_with_metadata(
                                cleaned_text, 
                                code_metadata,
                                chunk_size=1000,  # Adjust as needed
                                chunk_overlap=200  # Adjust as needed
                            )
                            
                            # Add each chunk to Chroma
                            for i, chunk_data in enumerate(chunks):
                                chunk_text = chunk_data["text"]
                                chunk_metadata = chunk_data["metadata"]
                                
                                chunk_id = f"{os.path.basename(file_path)}_{code}_chunk_{i}"
                                
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
                                    logger.info(f"Added chunk {i} for status code {code} of document {file_path}")
                                except Exception as e:
                                    logger.error(f"Error adding chunk {i} for status code {code}: {str(e)}")
                    else:
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
                                logger.error(f"Error adding chunk {i} of document {file_path}")
    
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