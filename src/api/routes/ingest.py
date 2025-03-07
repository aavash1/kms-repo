# src/api/routes/ingest.py
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import Optional, List
from src.core.services.ingest_service import IngestService
from src.core.mariadb.mariadb_connector import get_file_metadata
from src.core.services.file_server import fetch_file_from_server
import pandas as pd
from io import StringIO
from pydantic import BaseModel


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

class FileDto(BaseModel):
    id: Optional[int] = None
    resolveId: Optional[int] = None
    path: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    isImage: Optional[bool] = False

class TroubleshootingReport(BaseModel):
    resolveId: int
    errorCodeId: int
    clientNm: Optional[str] = None
    osVersionId: Optional[int] = None
    content: Optional[str] = None
    imgFiles: Optional[List[FileDto]] = None
    files: Optional[List[FileDto]] = None


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


@router.post("/query/kmschatbot/troubleshooting")
async def process_troubleshooting_reports(
    report: TroubleshootingReport,
    ingest_service: IngestService = Depends(IngestService)
):
    try:
        metadata_df = pd.DataFrame([{"file_id": f.id, "file_path": f.path} for f in (report.files or [])])
        results = await ingest_service.process_server_files(metadata_df, str(report.errorCodeId))

        return {
            "status": "success",
            "message": f"Processed {len(results)} files.",
            "resolve_id": report.resolveId,
            "error_code_id": report.errorCodeId,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing troubleshooting report: {str(e)}")


@router.post("/kmschatbot/troubleshooting")
async def process_troubleshooting_report(
    report: TroubleshootingReport,
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Process troubleshooting report data from the Spring Boot backend.
    
    This endpoint is called by the Spring Boot application after a user submits
    a troubleshooting report through the UI. It processes the report data and
    associated files retrieved from the file server.
    
    Args:
        report: Pydantic model containing troubleshooting report data including:
               - resolveId: Resolve record ID
               - errorCodeId: Error code identifier
               - clientNm: Client/customer name
               - osVersionId: OS version identifier
               - content: Report content
               - imgFiles: List of image files in the content
               - files: List of attached files
    """
    try:
        # Extract required fields
        resolve_id = report.resolveId
        error_code_id = report.errorCodeId
        client_name = report.clientNm
        os_version_id = report.osVersionId
        content = report.content
        img_files = report.imgFiles or []
        attached_files = report.files or []
        
        if not error_code_id:
            raise HTTPException(status_code=400, detail="Error code ID is required")
        
        # Create metadata 
        metadata = {
            "resolve_id": resolve_id,
            "error_code_id": error_code_id,
            "client_name": client_name,
            "os_version_id": os_version_id,
            "status_code": str(error_code_id),  # Using error code as status code
            "content_summary": content[:100] if content else ""  # First 100 chars as summary
        }
        
        # Process files
        all_files = []
        
        # Process image files embedded in content
        for img in img_files:
            if img.path:
                all_files.append({
                    "file_id": img.id,
                    "file_path": img.path,
                    "file_name": img.name,
                    "is_image": True,
                    "metadata": metadata
                })
        
        # Process attached files
        for file in attached_files:
            if file.path:
                all_files.append({
                    "file_id": file.id,
                    "file_path": file.path,
                    "file_name": file.name,
                    "is_image": False,
                    "metadata": metadata
                })
        
        # Process files if any exist
        if all_files:
            # Convert to DataFrame for processing
            metadata_df = pd.DataFrame(all_files)
            
            # Process files from server without saving locally
            results = await ingest_service.process_server_files(metadata_df, str(error_code_id))
            
            return {
                "status": "success",
                "message": f"Processed {len(results)} files for troubleshooting report",
                "resolve_id": resolve_id,
                "error_code_id": error_code_id,
                "results": results
            }
        # Process just the content if no files
        elif content:
            # Generate embedding for the content directly
            result = await ingest_service.process_text_content(
                content, 
                str(error_code_id),
                metadata
            )
            
            return {
                "status": "success",
                "message": "Processed report content",
                "resolve_id": resolve_id,
                "error_code_id": error_code_id,
                "results": [result]
            }
        else:
            return {
                "status": "warning",
                "message": "No content or files to process",
                "resolve_id": resolve_id,
                "error_code_id": error_code_id
            }
                
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing troubleshooting report: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing report: {str(e)}")