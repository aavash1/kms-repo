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
from src.core.file_handlers.excel_handler import ExcelHandler  # New import
from src.core.file_handlers.pptx_handler import PPTXHandler  # New import
from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
from src.core.utils.file_identification import get_file_type
from src.core.utils.post_processing import clean_extracted_text as clean_text
from src.core.utils.text_chunking import chunk_with_metadata
from src.core.utils.text_chunking import chunk_text
from src.core.services.file_download import download_file_from_url
import chromadb
import logging
import ollama
from typing import List, Optional, Dict, Any, Set
import tempfile
from concurrent.futures import ThreadPoolExecutor
import asyncio
import magic
from datetime import datetime, timedelta
from langchain_ollama import OllamaEmbeddings


import logging
logger = logging.getLogger(__name__)

# Define directories.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(os.getcwd(), "sample_data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db1")
CHAT_DIR   = os.path.join(PROJECT_ROOT, "chroma_chat")
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CHAT_DIR,   exist_ok=True)
download_semaphore = asyncio.Semaphore(5)

# Shared globals (updated at startup)
class GlobalState:
    def __init__(self):
        self._global_prompt = None
        self._rag_chain = None
        self._vector_store = None
        self._chromadb_collection = None
        self._workflow = None
        self._memory = None
        self._personal_vector_store = None 

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

def get_personal_vector_store():
    if _state._personal_vector_store is None:
        import chromadb, time
        import numpy as np
        
        # Use a single client with CHROMA_DIR (the main directory)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Get or create the chat_files collection
        try:
            collection = client.get_collection("chat_files")
            logger.info(f"Found chat_files collection with {collection.count()} documents")
        except ValueError:
            collection = client.create_collection("chat_files")
            logger.info(f"Created new chat_files collection")
        
        # Create a custom wrapper that works directly with ChromaDB
        class DirectChromaStore:
            def __init__(self, collection, embedding_function):
                self._collection = collection
                self._embedding_function = embedding_function
            
            def similarity_search(self, query, k=4, filter=None):
                from langchain_core.documents import Document
                
                # Generate embedding for the query
                embedding = self._embedding_function.embed_query(query)
                
                # Perform the search using direct ChromaDB API
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    where=filter
                )
                
                # Convert results to Document objects
                documents = []
                if results and 'documents' in results and results['documents']:
                    for i, doc_text in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas'][0]) else {}
                        doc = Document(page_content=doc_text, metadata=metadata)
                        documents.append(doc)
                
                return documents
        
        # Create our wrapper with the ChromaDB collection
        emb = OllamaEmbeddings(model="mxbai-embed-large")
        _state._personal_vector_store = DirectChromaStore(collection, emb)
        
        logger.info(f"Initialized direct ChromaDB vector store in {CHROMA_DIR}")
        
    return _state._personal_vector_store

def clean_expired_chat_vectors(days: int = 7):
    """Delete chat vectors older than *days* from chat_files collection."""
    store = get_personal_vector_store()
    col = store._collection  # direct chroma collection
    keep = []
    for meta, id_ in zip(col.get()["metadatas"], col.get()["ids"]):
        created = meta.get("created_at")
        if not created:
            keep.append(id_)
            continue
        if datetime.utcnow() - datetime.fromisoformat(created) < timedelta(days=days):
            keep.append(id_)
    delete_ids = [id_ for id_ in col.get()["ids"] if id_ not in keep]
    if delete_ids:
        col.delete(ids=delete_ids)
        logger.info(f"Cleaned {len(delete_ids)} expired chat vectors")

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

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
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
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    return sanitized


# In src/core/services/file_utils.py
async def process_file_content(file_content: bytes, filename: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200, model_manager=None) -> Dict[str, Any]:
    """Process file content to extract text and create embeddings."""
    temp_file = None
    temp_file_path = None
    
    try:
        # Write content to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
        temp_file_path = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()
        logger.debug(f"Wrote file content to temporary file: {temp_file_path}")
        
        # Get MIME type and file type
        import magic
        mime_type = magic.Magic(mime=True).from_buffer(file_content)
        file_type = get_file_type(file_content=file_content, filename=filename, content_type=mime_type)
        
        logger.debug(f"Identified file type for {filename}: {file_type}")
        
        # Process based on file type
        if file_type == "pdf":
            handler = PDFHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "hwp":
            # Use the HWP handler specifically
            logger.debug(f"Processing HWP file: {filename}")
            handler = HWPHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "doc":
            handler = AdvancedDocHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "image":
            handler = ImageHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "excel":
            handler = ExcelHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "pptx":
            handler = PPTXHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        elif file_type == "msg":
            handler = MSGHandler(model_manager=model_manager)
            handler_result = await handler.extract_text(temp_file_path)
        else:
            raise ValueError(f"Unsupported or unknown file type for {filename}")
        
        # Consistently handle both string and tuple returns
        if isinstance(handler_result, tuple):
            text, tables = handler_result
        else:
            text = handler_result
            tables = []
            
        # Ensure we got text from file and it's a string (not a tuple)
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)} from handler for {filename}")
            try:
                text = str(text)
            except:
                text = ""
            
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning(f"No text extracted from {filename} (path: {temp_file_path})")
            return {
                "filename": filename,
                "status": "warning",
                "message": "No text extracted from file",
                "chunks": []
            }
        
        # Clean the extracted text
        cleaned_text = clean_extracted_text(text)
        
        # Chunk the text
        from src.core.utils.text_chunking import chunk_text
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        
        if not chunks:
            logger.warning(f"Text chunking failed for {filename}")
            return {
                "filename": filename, 
                "status": "warning", 
                "message": "Failed to chunk text.",
                "chunks": []
            }
        
        # Add tables to metadata if present
        updated_metadata = metadata.copy() if metadata else {}
        if tables:
            try:
                # Convert tables to a string representation for metadata
                tables_str = "\n".join([str(table) for table in tables])
                updated_metadata["tables"] = tables_str[:1000]  # Limit size
            except Exception as e:
                logger.warning(f"Failed to convert tables to string: {e}")
        
        return {
            "filename": filename,
            "status": "success",
            "chunks": chunks,
            "chunk_count": len(chunks),
            "metadata": updated_metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing file content for {filename}: {e}", exc_info=True)
        return {
            "filename": filename,
            "status": "error",
            "message": f"Failed to process file: {str(e)}",
            "chunks": []
        }
    finally:
        # Clean up temporary file
        if temp_file and not temp_file.closed:
            temp_file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {temp_file_path}: {e}")


async def process_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, filename: Optional[str] = None, model_manager=None) -> Dict[str, Any]:
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()

        metadata = {"file_path": file_path}
        result = await process_file_content(
            file_content=file_content,
            filename=filename or os.path.basename(file_path),
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_manager=model_manager
        )
        return result
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to process file: {str(e)}", "chunks": []}


    

def process_file_from_server(file_content: bytes, filename: str, metadata: dict, chunk_size=1000, chunk_overlap=200) -> Dict[str, Any]:
    try:
        file_type = get_file_type(file_content)
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

        text = handler.extract_text_from_memory(file_content)
        if not text or not text.strip():
            logger.warning(f"No text extracted from {filename}")
            return {"filename": filename, "status": "error", "message": "No text extracted from file."}
        cleaned_text = clean_extracted_text(text)

        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        if not chunks:
            logger.warning(f"Failed to chunk text for {filename}")
            return {"filename": filename, "status": "error", "message": "Failed to chunk extracted text."}

        chromadb_collection = get_chromadb_collection()
        if chromadb_collection is None:
            logger.error("ChromaDB collection is not initialized")
            raise RuntimeError("ChromaDB collection is not initialized.")

        batch_size = 20
        all_chunk_ids = []
        existing_ids = set(chromadb_collection.get()["ids"])

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_ids = [f"{filename}_chunk_{i + j}" for j in range(len(batch))]
            batch_metadata = [{**metadata, "chunk_index": i + j, "chunk_count": len(chunks)} for j in range(len(batch))]
            batch_embeddings = []

            valid_batch = [(id_, chunk) for id_, chunk in zip(batch_ids, batch) if id_ not in existing_ids]
            if not valid_batch:
                logger.info(f"Skipping batch {i // batch_size + 1} for {filename} - all chunks are duplicates")
                continue

            batch_ids, batch_chunks = zip(*valid_batch)
            batch_metadata = batch_metadata[:len(batch_ids)]

            for chunk in batch_chunks:
                try:
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                    if embedding is None:
                        embedding = [0.0] * 1024
                    embedding = flatten_embedding(embedding)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed chunk in {filename}: {e}")
                    batch_embeddings.append([0.0] * 1024)

            chromadb_collection.add(
                ids=list(batch_ids),
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metadata
            )
            all_chunk_ids.extend(batch_ids)

        logger.info(f"Successfully processed {filename} with {len(all_chunk_ids)} chunks")
        return {
            "filename": filename,
            "status": "success",
            "message": f"File processed successfully with {len(all_chunk_ids)} chunks.",
            "chunk_count": len(all_chunk_ids)
        }
    except Exception as e:
        logger.error(f"Error processing file '{filename}': {str(e)}", exc_info=True)
        return {"filename": filename, "status": "error", "message": f"Failed to process file: {str(e)}"}


def load_documents_to_chroma(pdf_handler, doc_handler, hwp_handler, msg_handler=None):
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

    batch_size = 20
    all_chunks = []

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
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
                    base_metadata = {"source_id": file_path, "filename": file, "file_type": ext[1:]}
                    chunks = chunk_with_metadata(cleaned_text, base_metadata, chunk_size=1000, chunk_overlap=200)
                    all_chunks.extend(chunks)

    existing_ids = set(chroma_coll.get()["ids"])
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_ids = [f"{chunk['metadata']['filename']}_chunk_{i + j}" for j, chunk in enumerate(batch)]
        batch_chunks = [chunk["text"] for chunk in batch]
        batch_metadata = [chunk["metadata"] for chunk in batch]
        batch_embeddings = []

        valid_batch = [(id_, chunk) for id_, chunk in zip(batch_ids, batch_chunks) if id_ not in existing_ids]
        if not valid_batch:
            logger.info(f"Skipping batch {i // batch_size + 1} - all chunks are duplicates")
            continue

        batch_ids, batch_chunks = zip(*valid_batch)
        batch_metadata = batch_metadata[:len(batch_ids)]

        for chunk in batch_chunks:
            try:
                embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                if embedding is None:
                    embedding = [0.0] * 1024
                embedding = flatten_embedding(embedding)
                batch_embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed chunk: {e}")
                batch_embeddings.append([0.0] * 1024)

        chroma_coll.add(
            ids=list(batch_ids),
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metadata
        )
        logger.info(f"Added batch {i // batch_size + 1} with {len(batch_ids)} chunks")

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



async def process_html_content(
    html_content: str,
    metadata: Dict[str, Any],
    html_handler,
    vision_extractor,
    chunk_size=1000,
    chunk_overlap=200
) -> Dict[str, Any]:
    """
    Process HTML content, extract text and images, embed the content, and store in ChromaDB.
    
    Args:
        html_content: Raw HTML content as a string.
        metadata: Metadata dictionary containing report details (e.g., resolve_id, url).
        html_handler: HTMLContentHandler instance for processing HTML.
        vision_extractor: GraniteVisionExtractor instance for extracting text from images.
        chunk_size: Size of text chunks for embedding.
        chunk_overlap: Overlap between text chunks.
        
    Returns:
        Dict with processing status, message, and chunk count.
    """
    try:
        document_id = metadata.get('document_id', 'unknown')
        document_type = metadata.get('document_type', 'unknown')
        
        # Set base_url on html_handler if provided (from your implementation)
        base_url = metadata.get('url', None)
        if base_url:
            html_handler.base_url = base_url

        # Process HTML content using HTMLContentHandler
        result = await html_handler.process_html(
            html_content=html_content,
            download_images=True,
            extract_image_text=False  # We'll handle image text extraction with GraniteVisionExtractor
        )

        # Extract text from HTML
        html_text = result.get('html_text', '')
        if not html_text or not html_text.strip():
            logger.warning(
                f"No text extracted from HTML content for document id {document_id}. "
                f"HTML length: {len(html_content)}, HTML preview: {html_content[:200]}..."
            )
        else:
            logger.debug(f"Extracted HTML text for document {document_id}: {html_text[:200]}... (truncated)")

        # Log the number of images found
        logger.debug(f"Found {len(result['images'])} images in HTML content for document {document_id} with URLs: {[img['src'] for img in result['images']]}")

        # Check if there's nothing to process
        if not html_text.strip() and not result['images']:
            logger.warning(f"No text or images extracted from HTML content: {document_id}")
            return {
                "document_id": document_id,
                "status": "warning",
                "message": f"No text extracted from HTML (length: {len(html_content)}) and no images found"
            }

        # Process images using GraniteVisionExtractor
        image_texts: List[str] = []
        image_urls: Set[str] = set()

        async def process_image(img_info):
            img_src = img_info['src']
            if img_src in image_urls:
                logger.debug(f"Skipping duplicate image URL: {img_src}")
                return None
            image_urls.add(img_src)
            try:
                async with download_semaphore:  # Limit concurrent downloads
                    download_result = await download_file_from_url(img_src)
                if isinstance(download_result, tuple):
                    image_content, content_type = download_result
                else:
                    image_content = download_result
                    content_type = None
                if image_content:
                    # Use GraniteVisionExtractor to extract text
                    extracted_text = await vision_extractor.extract_text_from_bytes(image_content, timeout_seconds=30)
                    if extracted_text and extracted_text.strip():
                        logger.debug(f"Extracted text from image {img_src}: {extracted_text[:100]}... (truncated)")
                        return extracted_text
                    else:
                        logger.debug(f"No text extracted from image at {img_src}")
                else:
                    logger.warning(f"Failed to download image from {img_src}")
            except Exception as e:
                logger.error(f"Error processing image from {img_src}: {e}")
            return None

        # Process images concurrently
        tasks = [process_image(img_info) for img_info in result['images']]
        image_texts = await asyncio.gather(*tasks)
        image_texts = [text for text in image_texts if text and text.strip()]

        # Process HTML text into chunks
        html_chunks = []
        if html_text.strip():
            html_chunks = chunk_text(html_text, chunk_size, chunk_overlap)
            logger.debug(f"Split HTML text into {len(html_chunks)} chunks")

        # Combine HTML chunks and image texts
        all_chunks = html_chunks + image_texts
        if not all_chunks:
            logger.warning(f"No chunks created for document: {document_id}")
            return {
                "document_id": document_id,
                "status": "warning",
                "message": "No chunks created from combined text"
            }

        # Embed and store in ChromaDB (from your implementation)
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error(f"ChromaDB collection not initialized for document {document_id}")
            return {
                "document_id": document_id,
                "status": "error",
                "message": "ChromaDB collection not initialized"
            }

        batch_size = 20
        all_chunk_ids = []
        existing_ids = set(chromadb_collection.get()["ids"])

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_ids = [f"{document_id}_chunk_{i + j}" for j in range(len(batch_chunks))]
            batch_metadata = [{**sanitize_metadata(metadata), "chunk_index": i + j, "chunk_count": len(all_chunks)} for j in range(len(batch_chunks))]
            batch_embeddings = []

            # Filter out duplicates
            valid_indices = [idx for idx, id_ in enumerate(batch_ids) if id_ not in existing_ids]
            if not valid_indices:
                logger.info(f"Skipping batch {i // batch_size + 1} for {document_id} - all chunks are duplicates")
                continue

            batch_ids = [batch_ids[idx] for idx in valid_indices]
            batch_chunks = [batch_chunks[idx] for idx in valid_indices]
            batch_metadata = [batch_metadata[idx] for idx in valid_indices]

            # Generate embeddings for the batch
            for chunk in batch_chunks:
                try:
                    embed_response = await asyncio.to_thread(ollama.embed, model="mxbai-embed-large", input=chunk)
                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                    if embedding is None:
                        logger.warning(f"Embedding failed for chunk in {document_id}, using zero vector")
                        embedding = [0.0] * 1024
                    embedding = flatten_embedding(embedding)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed chunk in {document_id}: {e}")
                    batch_embeddings.append([0.0] * 1024)

            # Ensure lengths match
            if not (len(batch_ids) == len(batch_metadata) == len(batch_embeddings) == len(batch_chunks)):
                logger.error(
                    f"Mismatched lengths in batch: ids={len(batch_ids)}, metadata={len(batch_metadata)}, "
                    f"embeddings={len(batch_embeddings)}, documents={len(batch_chunks)}"
                )
                raise ValueError(
                    f"Unequal lengths for fields: ids: {len(batch_ids)}, metadatas: {len(batch_metadata)}, "
                    f"embeddings: {len(batch_embeddings)}, documents: {len(batch_chunks)}"
                )

            chromadb_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metadata
            )
            all_chunk_ids.extend(batch_ids)

        logger.info(f"Processed HTML document {document_id} with {len(all_chunk_ids)} chunks")
        return {
            "document_id": document_id,
            "status": "success",
            "message": f"HTML content processed with {len(all_chunk_ids)} chunks",
            "chunk_count": len(all_chunk_ids)
        }

    except Exception as e:
        logger.error(f"Error processing HTML content for document {document_id}: {e}", exc_info=True)
        return {
            "document_id": document_id,
            "status": "error",
            "message": f"Failed to process: {str(e)}"
        }