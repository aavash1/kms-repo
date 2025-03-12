from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from typing import Optional, List, Dict, Any
from src.core.services.ingest_service import IngestService
import pandas as pd
from io import StringIO
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global IngestService instance
ingest_service = None

def get_ingest_service(request: Request) -> IngestService:
    """
    Dependency to get the IngestService instance with access to the DB connector.

    Args:
        request (Request): FastAPI request object to access app state.

    Returns:
        IngestService: Configured IngestService instance with DB connector.
    """
    db_connector = request.app.state.db_connector
    # Connection state is managed by MariaDBConnector internally
    return IngestService(db_connector=db_connector)

class FileDto(BaseModel):
    id: Optional[int] = None
    resolveId: Optional[int] = None
    path: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    isImage: Optional[bool] = False

class TroubleshootingReport(BaseModel):
    logical_names: List[str]  # Array of logical file names
    error_code_id: str        # Error code ID
    metadata: Dict[str, Any]  # Metadata including client_name and os_version

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
        result = await ingest_service.process_uploaded_files_optimized([file], status_code, metadata)
        
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
        raise he
    except Exception as e:
        logger.error(f"Error ingesting uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

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
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
            
        logger.info(f"Received {len(files)} files for upload to status code: {status_code}")
        
        if len(files) > 10 and background_tasks:
            import uuid
            task_id = str(uuid.uuid4())
            
            async def process_in_background():
                try:
                    await ingest_service.process_uploaded_files_optimized(files, status_code, metadata)
                    logger.info(f"Background task {task_id} completed for {len(files)} files")
                except Exception as e:
                    logger.error(f"Background task {task_id} failed: {e}")
            
            background_tasks.add_task(process_in_background)
            return {
                "status": "processing",
                "message": f"Processing {len(files)} files in the background",
                "task_id": task_id
            }
        else:
            return await ingest_service.process_uploaded_files_optimized(files, status_code, metadata)
            
    except HTTPException as he:
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

@router.post("/kmschatbot/troubleshooting")
async def process_troubleshooting_report(
    report: TroubleshootingReport,
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Process troubleshooting report data using error_code_id and logical_names.

    Delegates business logic to IngestService.process_troubleshooting_report.

    Args:
        report: JSON payload with logical_names, error_code_id, and metadata.
        ingest_service: Dependency-injected IngestService instance.

    Returns:
        dict: Processing results from IngestService.
    """
    try:
        result = await ingest_service.process_troubleshooting_report(
            report.logical_names,
            report.error_code_id,
            report.metadata
        )
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing troubleshooting report: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing report: {str(e)}")

@router.post("/kmschatbot/troubleshooting1")
async def process_troubleshooting_report1(
    report: Dict[str, Any],
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Handles troubleshooting reports from Spring Boot.

    - Fetches file URLs from MariaDB using logical names.
    - Passes the file metadata to `ingest_service.process_files_by_logical_names`.
    - Returns the processed embedding results.
    """
    try:
        error_code_id = report.get("error_code_id")
        client_name = report.get("client_name")
        os_version = report.get("os_version")
        logical_file_names = report.get("files", [])

        if not logical_file_names:
            logger.warning("No files provided in the request.")
            return {
                "status": "warning",
                "message": "No files to process",
                "error_code_id": error_code_id
            }

        logger.info(f"Processing report with error_code_id: {error_code_id}, files: {logical_file_names}")

        results = await ingest_service.process_files_by_logical_names(
            logical_file_names, error_code_id,
            metadata={"client_name": client_name, "os_version": os_version}
        )

        return {
            "status": "success",
            "message": f"Processed {len(results)} files",
            "error_code_id": error_code_id,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error processing troubleshooting report: {e}")
        return {
            "status": "error",
            "message": f"Error processing report: {str(e)}"
        }