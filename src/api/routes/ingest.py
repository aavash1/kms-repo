# src/api/routes/ingest.py
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import Optional, List
from src.core.services.ingest_service import IngestService
from src.core.mariadb.mariadb_connector import get_file_metadata
from src.core.services.file_server import fetch_file_from_server
import pandas as pd
from io import StringIO


import logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Global IngestService instance
ingest_service = None

def get_ingest_service():
    """Get the IngestService instance."""
    try:
        from src.main import ingest_service_instance
        if ingest_service_instance is None:
            # Fallback if the global instance is not set
            return IngestService()
        return ingest_service_instance
    except ImportError:
        return IngestService()

@router.post("/upload/{status_code}")
async def ingest_uploaded_file(
    status_code: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Upload a file, store it in sample_data, extract text, generate vector embeddings, and store in ChromaDB.
    This endpoint handles a single file but uses the same processing logic as the multiple files endpoint.
    """
    try:
        # Process as a list with a single file
        result = await ingest_service.process_uploaded_files([file], status_code, metadata)
        
        # If the single file processing failed, extract the error
        if result["successful"] == 0 and len(result["results"]) > 0:
            error_message = result["results"][0].get("message", "Unknown error")
            raise HTTPException(status_code=500, detail=error_message)
            
        # Return the first file's result for backward compatibility
        if len(result["results"]) > 0:
            first_result = result["results"][0]
            return {
                "status": first_result["status"],
                "message": first_result["message"],
                "id": first_result["filename"]
            }
        else:
            return {
                "status": "error",
                "message": "No file was processed",
                "id": None
            }
            
    except HTTPException as he:
        # Re-raise HTTP exceptions directly
        raise he
    except Exception as e:
        logger.error(f"Error ingesting uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# @router.post("/fetch/{status_code}")
# async def ingest_file_from_server(status_code: str):
#     """
#     Ingest files associated with a specific status_code from the file server.
#     """
#     try:
#         # Fetch file metadata from MariaDB
#         metadata_df = get_file_metadata(status_code)
#         if metadata_df is None or metadata_df.empty:
#             raise HTTPException(status_code=404, detail=f"No files found for status code {status_code}")

#         # Process files using the service
#         results = await ingest_service.process_server_files(metadata_df, status_code)

#         return {
#             "status": "success",
#             "message": f"Processed {len(results)} files for status_code '{status_code}'",
#             "results": results
#         }

#     except Exception as e:
#         logger.error(f"Error ingesting files from server: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@router.post("/upload-multiple/{status_code}")
async def ingest_multiple_files(
    status_code: str,
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    ingest_service: IngestService = Depends(get_ingest_service),
    background_tasks: BackgroundTasks = None
):
    """
    Upload multiple files at once, process them, and store in ChromaDB.
    For large numbers of files, processes them in the background.
    """
    try:
        # Check if there are any files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Log the number of files received
        logger.info(f"Received {len(files)} files for upload to status code: {status_code}")
        
        # For more than 10 files, process in background
        if len(files) > 10 and background_tasks:
            import uuid
            task_id = str(uuid.uuid4())
            
            # Define background task
            async def process_in_background():
                try:
                    await ingest_service.process_uploaded_files(files, status_code, metadata)
                    logger.info(f"Background task {task_id} completed for {len(files)} files")
                except Exception as e:
                    logger.error(f"Background task {task_id} failed: {e}")
            
            # Add task to background tasks
            background_tasks.add_task(process_in_background)
            return {
                "status": "processing",
                "message": f"Processing {len(files)} files in the background",
                "task_id": task_id
            }
        else:
            # Process synchronously
            return await ingest_service.process_uploaded_files(files, status_code, metadata)
            
    except HTTPException as he:
        # Re-raise HTTP exceptions directly
        raise he
    except Exception as e:
        logger.error(f"Error ingesting multiple files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@router.post("/server-files/{status_code}")
async def ingest_server_files(
    status_code: str,
    metadata_csv: str = Form(...),
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Process files already on the server based on provided metadata CSV.
    """
    try:
        # Parse CSV from string
        metadata_df = pd.read_csv(StringIO(metadata_csv))
        
        if 'file_id' not in metadata_df.columns or 'file_path' not in metadata_df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain 'file_id' and 'file_path' columns"
            )
        
        results = await ingest_service.process_server_files(metadata_df, status_code)
        return {
            "status": "success",
            "files_processed": len(results),
            "results": results
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing server files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing server files: {str(e)}")