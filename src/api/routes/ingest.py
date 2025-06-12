from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request, Response
from typing import Optional, List, Dict, Any
from src.core.services.ingest_service import IngestService
from src.core.services.query_service import QueryService
import pandas as pd
from io import StringIO
from pydantic import BaseModel
import logging
import json
from src.core.startup import get_components
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from src.core.services.file_utils import clean_expired_chat_vectors
from src.core.auth.auth_middleware import verify_api_key_and_member_id

logger = logging.getLogger(__name__)

router = APIRouter()

# Global IngestService instance
ingest_service = None

class IngestDocumentRequest(BaseModel):
    input_data: Dict[str, Any]  # Flexible for any custom fields
    file_urls: List[str] = []
    physical_nm: List[str] = []

class UpdateDocumentRequest(BaseModel):
    input_data: Dict[str, Any]
    file_urls: List[str] = []
    physical_nm: List[str] = []

def get_ingest_service(components=Depends(get_components)) -> IngestService:
    model_manager = components.get('model_manager')
    return components['ingest_service'] if 'ingest_service' in components else IngestService(model_manager=model_manager)

def get_ingest_service2(request: Request) -> IngestService:
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

def get_ingest_service_with_postgres_lazy(request: Request) -> IngestService:
    """
    Dependency to get the IngestService instance for delete operations only.
    Uses lazy_init=True to skip file handler initialization.
    """
    if hasattr(request.app.state, 'postgresql_db') and request.app.state.postgresql_db:
        logger.debug("Using PostgreSQL connector for IngestService (lazy)")
        return IngestService(db_connector=request.app.state.postgresql_db, lazy_init=True)
    else:
        logger.warning("No database connector available")
        return IngestService(db_connector=None, lazy_init=True)

def get_ingest_service_with_postgres(request: Request) -> IngestService:
    """
    Dependency to get the IngestService instance with PostgreSQL connector.
    Falls back to MariaDB if PostgreSQL is not available.
    """
    # Try PostgreSQL first
    if hasattr(request.app.state, 'postgresql_db') and request.app.state.postgresql_db:
        logger.debug("Using PostgreSQL connector for IngestService")
        return IngestService(db_connector=request.app.state.postgresql_db)
    
    # Fallback to MariaDB if available
    elif hasattr(request.app.state, 'db_connector') and request.app.state.db_connector:
        logger.warning("PostgreSQL not available, falling back to MariaDB")
        return IngestService(db_connector=request.app.state.db_connector)
    
    # No database available
    else:
        logger.warning("No database connector available, using IngestService without database")
        return IngestService(db_connector=None)

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


    class Config:
        schema_extra = {
            "example": {
                "error_code_id": "2",
                "client_name": "TestClient",
                "os_version_id": "RHEL",
                "content": "<p>Sample content with <img src='screenshot'></p>",
                "img_files": [
                    {"name": "screenshot1.png", "type": "image/png"},
                    {"name": "screenshot2.jpg", "type": "image/jpeg"}
                ]
            }
        }    

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

def get_query_service() -> QueryService:
    """Get the QueryService instance."""
    components = get_components()
    # Use the pre-initialized service from startup
    return components.get('query_service')

@router.post("/upload-chat", summary="Upload a file for per-chat use (isolated)")
async def upload_chat_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    ingest: IngestService = Depends(get_ingest_service),
    query_service: QueryService = Depends(get_query_service)
):
    """Embeds *one* file into the *chat_files* collection and returns its id."""
    try:
        if not file.filename:
            raise HTTPException(400, "File must have a valid filename")

        meta_dict = json.loads(metadata) if metadata else {}
        
        # Ensure created_at is included in metadata
        if "created_at" not in meta_dict:
            from datetime import datetime
            meta_dict["created_at"] = datetime.utcnow().isoformat()
            
        # Ensure document_type is set to filechat
        if "document_type" not in meta_dict:
            meta_dict["document_type"] = "filechat"
            
        res = await ingest.process_uploaded_files_optimized(
            [file],
            status_code="chat",
            metadata=json.dumps(meta_dict),
            scope="chat"
        )

        if res.get("successful", 0) > 0 and res.get("results"):
            result = res["results"][0]
            # Check if message contains "successfully" regardless of status
            if result.get("message", "").lower().find("successfully") != -1:
                # Refresh the vector stores to pick up the new file
                query_service.refresh_stores()
                return {
                    "status": "success",
                    "id": result.get("id") or file.filename,
                    "message": result.get("message", "File processed successfully")
                }
        
        error_message = res.get("results", [{}])[0].get("message", "Unknown error during file processing")
        # Don't log a warning if the message actually indicates success
        if "successfully" not in error_message.lower():
            logger.warning(f"File processing issue: {error_message}")
        return {
            "status": "error",
            "message": error_message,
            "id": None
        }

    except json.JSONDecodeError:
        logger.error("Invalid JSON metadata provided")
        raise HTTPException(400, "Metadata must be valid JSON")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to process file: {str(e)}",
            "id": None
        }

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

@router.post("/troubleshooting-with-files")
async def process_troubleshooting_report_with_files(
    resolve_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    ingest_service: IngestService = Depends(get_ingest_service)
):
    """
    Process troubleshooting report data and uploaded files on the GPU server with streaming support.

    Accepts a JSON string (resolveData) with errorCodeId, clientNm, osVersionId, and content (text only),
    along with a list of uploaded files (attachments and content images). Processes text and files directly
    using temporary storage and removes files after processing.

    Args:
        resolve_data: JSON string containing errorCodeId, clientNm, osVersionId, and content (text only).
        files: List of uploaded files (attachments and content images) streamed concurrently.
        ingest_service: Dependency-injected IngestService instance.

    Returns:
        dict: Processing results including status, total files, and details.
    """
    try:
        result = await ingest_service.process_direct_uploads(resolve_data, files)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing troubleshooting report with files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing report with files: {str(e)}")


# Maybe change the API to @router.post("/kmschatbot/ingest-document, from /kmschatbot/troubleshooting-with-url")
@router.post("/ingest-documents")
async def process_troubleshooting_report_with_S3files(
    request: IngestDocumentRequest,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    ingest_service: IngestService = Depends(get_ingest_service_with_postgres) 
):
    """
    Process data and S3 file URLs using PostgreSQL with dynamic document type validation.
    Supports multiple files with individual physical_nm tracking.
    document_id (board_id) is always provided by backend - no auto-generation needed.
    """
    try:
        # Extract member information from auth
        member_id = auth_data["member_id"]
        
        # VALIDATE physical_nm array length
        if request.physical_nm and len(request.file_urls) != len(request.physical_nm):
            raise HTTPException(
                status_code=400,
                detail=f"file_urls({len(request.file_urls)}) != physical_nm({len(request.physical_nm)})"
            )
        
        # Work with input_data dict directly
        input_data_dict = request.input_data.copy()
        
        # VALIDATE required document_id (must be provided by backend)
        document_id = input_data_dict.get("document_id")
        if not document_id:
            raise HTTPException(
                status_code=400,
                detail="document_id is required and must be provided by backend"
            )
        
        # Ensure document_id is integer for bigint compatibility
        try:
            document_id = int(document_id)
            input_data_dict["document_id"] = str(document_id)  # Store as string for ChromaDB
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="document_id must be a valid integer (bigint)"
            )

        # Validate document_type early (before processing)
        document_type = input_data_dict.get("document_type")
        if document_type:
            valid_types = await ingest_service.get_valid_document_types()
            if document_type not in valid_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid document_type: {document_type}. Valid types: {valid_types}"
                )
        
        # Inject member_id into custom_metadata for isolation
        if "custom_metadata" not in input_data_dict:
            input_data_dict["custom_metadata"] = {}
        
        input_data_dict["custom_metadata"]["member_id"] = member_id
        input_data_dict["custom_metadata"]["uploaded_by"] = member_id
        
        # Inject physical_nm into custom_metadata
        if request.physical_nm:
            input_data_dict["custom_metadata"]["physical_nm"] = request.physical_nm
        
        # Convert to JSON string for the processing method
        enhanced_input_data = json.dumps(input_data_dict)
        
        # Process with enhanced data
        result = await ingest_service.process_direct_uploads_with_urls(enhanced_input_data, request.file_urls)
        
        # Add member context to response
        result["member_id"] = member_id
        result["document_id"] = document_id  # Return as integer
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing document with file URLs: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document with file URLs: {str(e)}")


##OLD API
@router.post("/kmschatbot/troubleshooting-with-urls")
async def process_troubleshooting_report_with_filess(
        resolve_data: str = Form(...),
        file_urls: List[str] = Form(default=[]),  # Changed from List[UploadFile] to List[str]
        ingest_service: IngestService = Depends(get_ingest_service)
    ):
        """
        Process troubleshooting report data and S3 file URLs on the GPU server with streaming support.

        Accepts a JSON string (resolveData) with errorCodeId, clientNm, osVersionId, and content (text only),
        along with a list of S3 URLs pointing to uploaded files (attachments and content images). Processes
        text and downloads files from S3 URLs directly, using temporary storage and removing files after processing.

        Args:
            resolve_data: JSON string containing errorCodeId, clientNm, osVersionId, and content (text only).
            file_urls: List of S3 URLs pointing to uploaded files (attachments and content images).
            ingest_service: Dependency-injected IngestService instance.

        Returns:
            dict: Processing results including status, total files, and details.
        """
        try:
            result = await ingest_service.process_direct_uploads_with_urls(resolve_data, file_urls)
            return result
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing troubleshooting report with file URLs: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing report with file URLs: {str(e)}")


@router.post("/process-mariadb-troubleshooting")
async def process_mariadb_troubleshooting(
    ingest_service: IngestService = Depends(get_ingest_service),
    components=Depends(get_components)
):
    """Process unprocessed troubleshooting reports from MariaDB and embed into ChromaDB."""
    try:
        html_handler = components['document_handlers']['html']
        vision_extractor = components['document_handlers']['granite_vision']

        result = await ingest_service.process_mariadb_troubleshooting_data(
            html_handler=html_handler,
            vision_extractor=vision_extractor
        )
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing MariaDB troubleshooting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing MariaDB data: {str(e)}")

@router.get("/kmschatbot/related-resolves/{error_code_id}")
async def get_related_resolves(error_code_id: str):
    """
    Retrieve all Resolve entries related to a given error_code_id using the knowledge graph.

    Args:
        error_code_id (str): The error code ID to query.

    Returns:
        dict: List of related Resolve entries and their attachments.
    """
    try:
        from src.core.services.knowledge_graph import knowledge_graph
        result = knowledge_graph.query_related_resolves(error_code_id)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Error querying related resolves for error_code_id {error_code_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying related resolves: {str(e)}")


@router.post("/kmschatbot/refresh-static-data")
async def refresh_static_data():
    """
    Refresh the static error code data in the cache.
    """
    try:
        from src.core.services.static_data_cache import static_data_cache
        static_data_cache.refresh_static_data()
        return {"status": "success", "message": "Static data cache refreshed"}
    except Exception as e:
        logger.error(f"Error refreshing static data: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing static data: {str(e)}")

@router.put("/updateExisting/{document_id}")
async def update_by_document_id(
    document_id: str, 
    request: UpdateDocumentRequest,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    ingest_service: IngestService = Depends(get_ingest_service_with_postgres)
):
    """
    Update documents by document_id (bigint - equivalent to board_id).
    Supports both file-based and HTML-only content updates.
    """
    try:
        member_id = auth_data["member_id"]
        
         #  Validate string input
        if not document_id or not document_id.strip():
            raise HTTPException(400, "document_id cannot be empty")
        
        #  Validate it's numeric
        if not document_id.isdigit():
            raise HTTPException(400, "document_id must be numeric")
        
        #  Convert to integer for service call
        try:
            document_id_int = int(document_id)
            if document_id_int <= 0:
                raise HTTPException(400, "document_id must be positive")
        except ValueError:
            raise HTTPException(400, "document_id must be a valid integer")
                
        # Validate physical_nm array length if provided
        if request.physical_nm and len(request.file_urls) != len(request.physical_nm):
            raise HTTPException(
                status_code=400,
                detail=f"file_urls({len(request.file_urls)}) != physical_nm({len(request.physical_nm)})"
            )
        
        # Call update service
        result = await ingest_service.update_by_document_id(
            target_document_id=document_id_int,  # Pass as int
            member_id=member_id,
            input_data=request.input_data,
            file_urls=request.file_urls,
            physical_nm=request.physical_nm
        )
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating document_id {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.delete("/deleteWithDocumentId/{document_id}")
async def delete_by_document_id(
    document_id: str,  # Changed from str to int for bigint
    auth_data: dict = Depends(verify_api_key_and_member_id),
    ingest_service: IngestService = Depends(get_ingest_service_with_postgres_lazy)
):
    """Delete documents by document_id (bigint - board_id) with member isolation"""
    try:
        member_id = auth_data["member_id"]
        
     # ✅ Validate string input
        if not document_id or not document_id.strip():
            raise HTTPException(400, "document_id cannot be empty")
        
        # ✅ Validate it's numeric
        if not document_id.isdigit():
            raise HTTPException(400, "document_id must be numeric")
        
        # ✅ Convert to integer for service call
        try:
            document_id_int = int(document_id)
            if document_id_int <= 0:
                raise HTTPException(400, "document_id must be positive")
        except ValueError:
            raise HTTPException(400, "document_id must be a valid integer")
        
        result = await ingest_service.delete_by_document_id(document_id_int, member_id)
        
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail=result["message"])
        elif result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/refresh-chromadb")
async def refresh_chromadb_collection(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    ingest_service: IngestService = Depends(get_ingest_service_with_postgres),
    response: Response = None
):
    """
    Refresh ChromaDB collection and all related services after updating or deleting.
    Member ID is used only for authentication verification.
    
    This refreshes:
    - ChromaDB collection connection
    - Vector store
    - RAG chain  
    - QueryService stores
    
    Returns:
        Refresh summary with collection status and operations performed
    """
    try:
        # Extract authenticated member (only for verification/audit)
        auth_member_id = auth_data["member_id"]
        logger.info(f"Member {auth_member_id} requesting ChromaDB refresh")
        
        # Call business logic
        result = await ingest_service.refresh_chromadb_collection(
            refreshed_by=auth_member_id
        )
        
        # Set cache headers if refresh was successful
        if result.get("status") == "success" and response:
            response.headers["Cache-Control"] = "public, max-age=1800"  # 30 minutes
        
        # Return result from business logic
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in refresh_chromadb_collection controller: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing refresh request: {str(e)}"
        )


@router.post("/cleanup-duplicates")
async def cleanup_duplicate_entries(
    dry_run: bool = True,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    ingest_service: IngestService = Depends(get_ingest_service_with_postgres_lazy)
):
    """Clean up duplicate entries in ChromaDB"""
    try:
        result = await ingest_service.cleanup_duplicate_entries(dry_run=dry_run)
        return result
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))