# src/core/services/ingest_service.py
import os
import tempfile
import json
import ollama
import hashlib
import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import List, Optional, Dict, Any
from src.core.services.file_utils import process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection, process_file_content
from src.core.services.file_server import fetch_file_from_server
from src.core.mariadb.mariadb_connector import get_file_metadata
import logging

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self):
        self.sample_data_dir = os.path.join(os.getcwd(), "sample_data")
        os.makedirs(self.sample_data_dir, exist_ok=True)
        
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and encoding issues."""
        # Create a safe filename without changing the extension
        base, ext = os.path.splitext(filename)
        safe_name = f"{hashlib.md5(filename.encode('utf-8')).hexdigest()[:10]}{ext}"
        return safe_name
       
    async def process_uploaded_files_optimized(self, files: List[UploadFile], status_code: str, metadata: Optional[str] = None, chunk_size=1000, chunk_overlap=200):
        """Optimized function to process multiple files with chunk-based embedding."""
        results = []
        
        # Create directory for status code
        status_dir = os.path.join(self.sample_data_dir, status_code)
        os.makedirs(status_dir, exist_ok=True)
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except Exception:
            metadata_dict = {"metadata": metadata}
        
        chromadb_collection = get_chromadb_collection()
        if chromadb_collection is None:
            logger.error("❌ ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized."}

        # Process each file
        for file in files:
            temp_file_path = None
            try:
                safe_filename = self._sanitize_filename(file.filename)
                file_path = os.path.join(status_dir, safe_filename)

                # Save file
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                # Extract text and chunk it
                extraction_result = process_file(file_path, chunk_size, chunk_overlap)
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "No text was extracted from the file."
                    })
                    continue

                base_metadata = {
                    "filename": file.filename,
                    "status_code": status_code,
                    "source_id": file_path,
                    **metadata_dict
                }

                # Store chunks in ChromaDB
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file.filename}_chunk_{i}"
                    chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}

                    # Generate embedding
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                    if embedding is None:
                        logger.warning(f"Failed to generate embedding for chunk {i}. Skipping.")
                        continue
                    embedding = flatten_embedding(embedding)

                    chromadb_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[chunk_metadata]
                    )
                    chunk_ids.append(chunk_id)

                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"File stored successfully with {len(chunk_ids)} chunks.",
                    "chunk_count": len(chunk_ids)
                })

            except Exception as e:
                logger.error(f"Error processing file '{file.filename}': {str(e)}")
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
            finally:
            # Clean up temp file if it exists
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {str(e)}")

        successful = len([r for r in results if r["status"] == "success"])
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "results": results
        }


    async def process_server_files(self, metadata_df: pd.DataFrame, status_code: str) -> List[Dict[str, Any]]:
        """
        Process files directly from the file server based on metadata.
        Generates embeddings without storing files locally.
        """
        try:
            results = []
            
            # Get ChromaDB collection using the global getter
            chromadb_collection = get_chromadb_collection()
            if chromadb_collection is None:
                raise HTTPException(status_code=500, detail="ChromaDB collection is not initialized")

            for _, row in metadata_df.iterrows():
                file_id = row["file_id"]
                file_path = row["file_path"]
                file_metadata = row.get("metadata", {})

                # Fetch file content from the file server
                file_content = await fetch_file_from_server(file_path)
                if not file_content:
                    results.append({"file_id": file_id, "status": "error", "message": "Failed to fetch file from server"})
                    continue

                # Process file content in-memory
                from src.core.services.file_utils import process_file_content
                extraction_result = process_file_content(file_content, os.path.basename(file_path), chunk_size=1000, chunk_overlap=200)
                chunks = extraction_result.get("chunks", [])

                if not chunks:
                    results.append({"file_id": file_id, "status": "error", "message": "No text was extracted from the file."})
                    continue

                # Prepare metadata
                base_metadata = {
                    "file_id": file_id,
                    "filename": os.path.basename(file_path),
                    "status_code": status_code,
                    "source_id": file_path,
                    **(json.loads(file_metadata) if isinstance(file_metadata, str) else file_metadata)
                }

                # Process and store each chunk
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_id}_chunk_{i}"
                    chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}

                    # Generate embedding
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")

                    if embedding is None:
                        logger.warning(f"Failed to generate embedding for chunk {i} of file {file_id}. Skipping.")
                        continue
                    embedding = flatten_embedding(embedding)

                    # Store in ChromaDB
                    chromadb_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[chunk_metadata]
                    )
                    chunk_ids.append(chunk_id)

                results.append({"file_id": file_id, "status": "success", "message": f"Processed {len(chunk_ids)} chunks.", "chunk_count": len(chunk_ids)})

            return results

        except Exception as e:
            logger.error(f"Error in process_server_files: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing server files: {str(e)}")
            
    async def process_text_content(self, text_content: str, status_code: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200):
        """
        Process text content directly without a file.
        
        Args:
            text_content: The text to process
            status_code: Status code for the content
            metadata: Additional metadata
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            Processing result
        """
        try:
            if not text_content or not text_content.strip():
                return {
                    "content_id": f"content_{status_code}",
                    "status": "error",
                    "message": "No text content provided."
                }
                
            chromadb_collection = get_chromadb_collection()
            if chromadb_collection is None:
                return {
                    "content_id": f"content_{status_code}",
                    "status": "error",
                    "message": "ChromaDB collection is not initialized."
                }
                
            # Clean and chunk the text
            cleaned_text = clean_extracted_text(text_content)
            from src.core.utils.text_chunking import chunk_text
            chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
            
            if not chunks:
                return {
                    "content_id": f"content_{status_code}",
                    "status": "error",
                    "message": "Failed to chunk text content."
                }
                
            # Prepare base metadata
            base_metadata = {
                "content_type": "text",
                "status_code": status_code,
                **(metadata or {})
            }
            
            # Process and store each chunk
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                content_id = f"content_{status_code}_{i}"
                chunk_metadata = {
                    **base_metadata, 
                    "chunk_index": i, 
                    "chunk_count": len(chunks)
                }
                
                # Generate embedding
                embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for text chunk {i}. Skipping.")
                    continue
                embedding = flatten_embedding(embedding)
                
                # Add to ChromaDB
                chromadb_collection.add(
                    ids=[content_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
                chunk_ids.append(content_id)
            
            return {
                "content_id": f"content_{status_code}",
                "status": "success",
                "message": f"Text content processed successfully with {len(chunk_ids)} chunks.",
                "chunk_count": len(chunk_ids)
            }
            
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return {
                "content_id": f"content_{status_code}",
                "status": "error",
                "message": str(e)
            }

def process_file_content(file_content: bytes, filename: str, chunk_size=1000, chunk_overlap=200):
    """
    Process file content in memory without saving to disk.
    
    Args:
        file_content: The binary content of the file
        filename: Name of the file (for determining file type)
        chunk_size: Size of chunks for text splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dict with extracted text, chunks, tables, and status codes
    """
    try:
        from src.core.utils.file_identification import identify_file_type_from_content
        file_type = identify_file_type_from_content(file_content, filename)
        
        # Create a temporary file just for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Select appropriate handler based on file type
            from src.core.file_handlers.pdf_handler import PDFHandler
            from src.core.file_handlers.doc_handler import AdvancedDocHandler
            from src.core.file_handlers.hwp_handler import HWPHandler
            from src.core.file_handlers.image_handler import ImageHandler
            
            if file_type.startswith("image/"):
                handler = ImageHandler(languages=['ko', 'en'])
            elif file_type == "application/pdf":
                handler = PDFHandler()
            elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                handler = AdvancedDocHandler()
            elif file_type == "application/x-hwp":
                handler = HWPHandler()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Extract text
            text = handler.extract_text(temp_file_path)
            tables = handler.extract_tables(temp_file_path) if hasattr(handler, "extract_tables") else []
            
            # We don't need to extract status codes from the file since they're provided in the request
            # but we'll keep the placeholder for compatibility
            status_codes = []
            
            # Clean text
            cleaned_text = clean_extracted_text(text)
            
            # Create chunks
            from src.core.utils.text_chunking import chunk_text
            chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap) if chunk_size > 0 else [cleaned_text]
            
            return {
                "text": cleaned_text,
                "chunks": chunks,
                "tables": tables,
                "status_codes": status_codes
            }
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing file content: {str(e)}")
        return {
            "text": "",
            "chunks": [],
            "tables": [],
            "status_codes": []
        }