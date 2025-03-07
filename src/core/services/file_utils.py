# src/core/services/file_utils.py
import os
import re
from pathlib import Path
from src.core.file_handlers.pdf_handler import PDFHandler
from src.core.file_handlers.doc_handler import AdvancedDocHandler
from src.core.file_handlers.hwp_handler import HWPHandler
from src.core.file_handlers.image_handler import ImageHandler
from src.core.utils.file_identification import get_file_type
from src.core.utils.post_processing import clean_extracted_text as clean_text
from src.core.utils.text_chunking import chunk_with_metadata


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

def process_file(file_path: str, chunk_size=1000, chunk_overlap=200):
    """
    Identify file type and extract text/tables using the appropriate handler.
    """
    from src.core.file_handlers.pdf_handler import PDFHandler
    from src.core.file_handlers.doc_handler import AdvancedDocHandler
    from src.core.file_handlers.hwp_handler import HWPHandler
    from src.core.file_handlers.image_handler import ImageHandler
    from src.core.utils.text_chunking import chunk_text

    file_type = get_file_type(file_path)
    if file_type.startswith("image/"):
        image_handler = ImageHandler(languages=['ko', 'en'])
        # Use Tesseract OCR by default
        ocr_results = image_handler.process_image(file_path, engine="tesseract")
        text = image_handler.reconstruct_aligned_text(ocr_results)
        tables = []  # Images typically don't have tables
    else:
        handlers = {
            "application/pdf": PDFHandler(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": AdvancedDocHandler(),
            "application/msword": AdvancedDocHandler(),
            "application/x-hwp": HWPHandler(),
            "image/": ImageHandler(), 
        }
        handler = next((h for mt, h in handlers.items() if file_type.startswith(mt)), None)
        if handler is None:
            raise ValueError(f"Unsupported file type: {file_type}")
        text = handler.extract_text(file_path)
        tables = handler.extract_tables(file_path)
    
    # Clean the extracted text
    cleaned_text = clean_extracted_text(text)
    
    # Chunk the text if requested
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap) if chunk_size > 0 else [cleaned_text]
    
    #return {"text": text, "tables": tables}
    return {
        "text": cleaned_text,  # Original cleaned text
        "chunks": chunks,      # Chunked text
        "tables": tables,      # Extracted tables
        "status_codes": handler.get_status_codes() if hasattr(handler, "get_status_codes") else []
    }
    

def load_documents_to_chroma2(pdf_handler, doc_handler, hwp_handler):
    """
    Walk DATA_FOLDER and add supported documents to the Chroma collection.
    """
    chroma_coll=get_chromadb_collection()
    if chroma_coll is None:
        raise RuntimeError("ChromaDB collection not initialized")

    try:
        initial_count = chroma_coll.count()
        print(f"Initial document count: {initial_count}")
    except Exception as e:
        raise RuntimeError(f"Cannot access ChromaDB collection: {e}")

    supported_extensions = {".pdf", ".doc", ".docx", ".hwp",".png",".jpg",".jpeg"}
    print(f"Loading documents from: {DATA_FOLDER}")

    import ollama
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                if ext == ".pdf":
                    text = pdf_handler.extract_text(file_path)
                    status_codes=pdf_handler.get_status_codes()
                elif ext == ".hwp": 
                    text=hwp_handler.extract_text(file_path)
                    status_codes=hwp_handler.get_status_codes()
                elif ext in [".png", ".jpg", ".jpeg"]:  # Handle image files
                    image_handler = ImageHandler()
                    text = image_handler.extract_text(file_path)
                    status_codes = image_handler.get_status_codes()
                else:
                    text = doc_handler.extract_text(file_path)
                    status_codes=doc_handler.get_status_codes()
                if text and text.strip():
                    cleaned_text = clean_extracted_text(text)
                    try:
                        embed_response = ollama.embed(model="mxbai-embed-large", input=cleaned_text)
                        embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                        if embedding is None:
                            print(f"Failed to get embedding for {file_path}")
                            continue
                        embedding = flatten_embedding(embedding)
                    except Exception as e:
                        print(f"Error during embedding for {file_path}: {e}")
                        continue

                    base_metadata = {
                        "id": file_path,
                        "filename": file,
                        "file_type": ext[1:],  
                    }
                    if status_codes:
                        for code in status_codes:
                            metadata=base_metadata.copy()
                            metadata["status_code"]=str(code)
                         
                            try:
                                doc_id=f"{file_path}_{code}"
                                chroma_coll.add(
                                #ids=[file_path],
                                ids=[doc_id],  # Unique ID for each status code
                                embeddings=[embedding],
                                documents=[cleaned_text],
                                metadatas=[metadata]
                            )
                                print(f"Added document to Chroma: {doc_id}")
                            except Exception as e:
                                print(f"Error adding document with status code {code} to Chroma: {e}")
                    else:
                        try:
                            chroma_coll.add(
                                ids=[file_path],
                                embeddings=[embedding],
                                documents=[cleaned_text],
                                metadatas=[base_metadata]
                            )
                            print(f"Added document to Chroma: {file_path}")
                        except Exception as e:
                            print(f"Error Processing {file_path}:{e}")
    try:
        count = chroma_coll.count()
    except AttributeError:
        count = chroma_coll._collection.count()
    print(f"Chroma collection now contains {count} documents.")


def load_documents_to_chroma(pdf_handler, doc_handler, hwp_handler):
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

    supported_extensions = {".pdf", ".doc", ".docx", ".hwp", ".png", ".jpg", ".jpeg"}
    print(f"Loading documents from: {DATA_FOLDER}")

    import ollama
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
                                
                                # Create unique ID for each chunk
                                #chunk_id = f"{file_path}_{code}_chunk_{i}"
                                chunk_id = f"{file.filename}_{code}_chunk_{i}"
                                
                                try:
                                    # Generate embedding for the chunk
                                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk_text)
                                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                                    if embedding is None:
                                        print(f"Failed to get embedding for chunk {i} of {file_path}")
                                        continue
                                    embedding = flatten_embedding(embedding)
                                    
                                    # Add to Chroma
                                    chroma_coll.add(
                                        ids=[chunk_id],
                                        embeddings=[embedding],
                                        documents=[chunk_text],
                                        metadatas=[chunk_metadata]
                                    )
                                    print(f"Added chunk {i} for status code {code} of document {file_path}")
                                except Exception as e:
                                    print(f"Error adding chunk {i} for status code {code}: {e}")
                    else:
                        # Chunk the document without status code
                        chunks = chunk_with_metadata(
                            cleaned_text, 
                            base_metadata,
                            chunk_size=1000,  # Adjust as needed
                            chunk_overlap=200  # Adjust as needed
                        )
                        
                        # Add each chunk to Chroma
                        for i, chunk_data in enumerate(chunks):
                            chunk_text = chunk_data["text"]
                            chunk_metadata = chunk_data["metadata"]
                            
                            # Create unique ID for each chunk
                            chunk_id = f"{file_path}_chunk_{i}"
                            
                            try:
                                # Generate embedding for the chunk
                                embed_response = ollama.embed(model="mxbai-embed-large", input=chunk_text)
                                embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                                if embedding is None:
                                    print(f"Failed to get embedding for chunk {i} of {file_path}")
                                    continue
                                embedding = flatten_embedding(embedding)
                                
                                # Add to Chroma
                                chroma_coll.add(
                                    ids=[chunk_id],
                                    embeddings=[embedding],
                                    documents=[chunk_text],
                                    metadatas=[chunk_metadata]
                                )
                                print(f"Added chunk {i} of document {file_path}")
                            except Exception as e:
                                print(f"Error adding chunk {i}: {e}")
    
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
