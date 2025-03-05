# src/api/routes/ingest.py
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
from src.core.services.ingest_service import IngestService
from src.core.mariadb.mariadb_connector import get_file_metadata
from src.core.services.file_server import fetch_file_from_server

import logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Global IngestService instance
ingest_service = IngestService()

@router.post("/upload/{status_code}")
async def ingest_uploaded_file(
    status_code: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload a file, store it in sample_data, extract text, generate vector embeddings, and store in ChromaDB.
    """
    try:
        return await ingest_service.process_uploaded_file(file, status_code, metadata)
    except Exception as e:
        logger.error(f"Error ingesting uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/fetch/{status_code}")
async def ingest_file_from_server(status_code: str):
    """
    Ingest files associated with a specific status_code from the file server.
    """
    try:
        # Fetch file metadata from MariaDB
        metadata_df = get_file_metadata(status_code)
        if metadata_df is None or metadata_df.empty:
            raise HTTPException(status_code=404, detail=f"No files found for status code {status_code}")

        # Process files using the service
        results = await ingest_service.process_server_files(metadata_df, status_code)

        return {
            "status": "success",
            "message": f"Processed {len(results)} files for status_code '{status_code}'",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error ingesting files from server: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")