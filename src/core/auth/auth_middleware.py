# src/core/auth/auth_middleware.py
"""
Authentication middleware for securing the API endpoints.
"""

import os
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import secrets
import logging

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Get API key from environment or generate a secure one
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Generate a secure API key if none is provided
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"No API_KEY found in environment. Generated new key: {API_KEY}")
    logger.warning("Add this key to your .env file for persistence across restarts.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """
    Verify the API key provided in the X-API-Key header.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key header missing")
    
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return api_key

# Function to get the current API key (for sharing with other services)
def get_current_api_key():
    """Get the current API key for use in other services."""
    return API_KEY

# Class-based middleware for FastAPI
class APIKeyMiddleware:
    """
    Middleware to check API key for all endpoints except those in excluded_paths.
    """
    def __init__(self, excluded_paths=None):
        self.excluded_paths = excluded_paths or ["/docs", "/redoc", "/openapi.json"]
    
    async def __call__(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
            
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return HTTPException(status_code=401, detail="API Key header missing")
        
        if api_key != API_KEY:
            return HTTPException(status_code=401, detail="Invalid API Key")
            
        # Continue processing the request
        return await call_next(request)