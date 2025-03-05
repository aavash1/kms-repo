# src/core/services/ingest_service.py
import os
import tempfile
import json
import ollama
from fastapi import HTTPException
from src.core.services.file_utils import process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection
import logging

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self):
        self.sample_data_dir = os.path.join(os.getcwd(), "sample_data")
        os.makedirs(self.sample_data_dir, exist_ok=True)
        
    async def process_uploaded_file(self, file, status_code: str, metadata: str = None):
        """Service method to handle file upload and processing"""
        try:
            # Create status code directory
            status_dir = os.path.join(self.sample_data_dir, status_code)
            os.makedirs(status_dir, exist_ok=True)
            file_path = os.path.join(status_dir, file.filename)

            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Extract and process text
            extraction_result = process_file(file_path)
            extracted_text = extraction_result.get("text", "")
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No text was extracted from the file.")

            cleaned_text = clean_extracted_text(extracted_text)

            # Generate embedding
            embed_response = ollama.embed(model="mxbai-embed-large", input=cleaned_text)
            embedding = embed_response.get("embedding") or embed_response.get("embeddings")
            if embedding is None:
                raise HTTPException(status_code=500, detail="Failed to generate embedding.")
            embedding = flatten_embedding(embedding)

            # Parse metadata
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except Exception:
                metadata_dict = {"metadata": metadata}

            # Store in ChromaDB
            chromadb_collection = get_chromadb_collection()
            if chromadb_collection is None:
                logger.error("❌ ChromaDB collection is not initialized!")
                raise HTTPException(status_code=500, detail="ChromaDB collection is not initialized.")
            
            chromadb_collection.add(
                ids=[file.filename],
                embeddings=[embedding],
                documents=[cleaned_text],
                metadatas=[{"filename": file.filename, "status_code": status_code, **metadata_dict}]
            )

            return {
                "status": "success",
                "message": f"File '{file.filename}' stored under status_code '{status_code}' successfully.",
                "id": file.filename
            }

        except Exception as e:
            logger.error(f"Error in process_uploaded_file: {e}")
            raise

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