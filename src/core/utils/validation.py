#/src/core/utils/validation.py
"""
Shared validation utilities for API endpoints
"""
from typing import List
from fastapi import HTTPException


def validate_document_request(document_id: str, file_urls: List[str], physical_nm: List[str]) -> int:
    """
    Shared validation logic for both ingest and update endpoints.
    
    Args:
        document_id: String representation of document ID
        file_urls: List of file URLs
        physical_nm: List of physical file names
        
    Returns:
        int: Validated document_id as integer
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate document_id
    if not document_id or not document_id.strip():
        raise HTTPException(400, "document_id cannot be empty")
    
    if not document_id.isdigit():
        raise HTTPException(400, "document_id must be numeric")
    
    try:
        document_id_int = int(document_id)
        if document_id_int <= 0:
            raise HTTPException(400, "document_id must be positive")
    except ValueError:
        raise HTTPException(400, "document_id must be a valid integer")
    
    # Validate physical_nm array length if provided
    if physical_nm and len(file_urls) != len(physical_nm):
        raise HTTPException(
            status_code=400,
            detail=f"file_urls({len(file_urls)}) != physical_nm({len(physical_nm)})"
        )
    
    return document_id_int


def validate_document_type(document_type: str, valid_types: List[str]) -> None:
    """
    Validate document type against allowed types.
    
    Args:
        document_type: Type to validate
        valid_types: List of valid document types
        
    Raises:
        HTTPException: If document type is invalid
    """
    if document_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid document_type: {document_type}. Valid types: {valid_types}"
        )