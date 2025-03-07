# src/core/services/ingest_service.py
import os
import tempfile
import json
import ollama
import hashlib
from fastapi import HTTPException, UploadFile
from typing import List, Optional
from src.core.services.file_utils import process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection
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
            try:
                file_path = os.path.join(status_dir, file.filename)

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

        successful = len([r for r in results if r["status"] == "success"])
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "results": results
        }


    async def process_server_files(self, metadata_df, status_code: str):
        """Service method to process files from server"""
        try:
            results = []
            for _, row in metadata_df.iterrows():
                file_id = row["file_id"]
                file_path = row["file_path"]

                # Create temporary file
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))

                try:
                    # Process file
                    extraction_result = process_file(temp_file_path)
                    extracted_text = extraction_result.get("text", "")
                    if not extracted_text.strip():
                        continue

                    cleaned_text = clean_extracted_text(extracted_text)

                    # Generate embedding
                    embed_response = ollama.embed(model="mxbai-embed-large", input=cleaned_text)
                    embedding = embed_response.get("embedding") or embed_response.get("embeddings")
                    if embedding is None:
                        continue
                    embedding = flatten_embedding(embedding)

                    # Store in ChromaDB
                    chromadb_collection = get_chromadb_collection()
                    chromadb_collection.add(
                        ids=[file_id],
                        embeddings=[embedding],
                        documents=[cleaned_text],
                        metadatas=[{"filename": file_path, "status_code": status_code}]
                    )

                    results.append({
                        "file_id": file_id,
                        "status": "success"
                    })

                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            return results

        except Exception as e:
            logger.error(f"Error in process_server_files: {e}")
            raise