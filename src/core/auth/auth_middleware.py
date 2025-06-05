# src/core/auth/auth_middleware.py
"""
Authentication middleware for securing the API endpoints.
Enhanced with optional session support while maintaining backward compatibility.
"""

import os
from typing import Optional
from fastapi import Request, HTTPException, Depends, Header
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
session_header = APIKeyHeader(name="X-Session-ID", auto_error=False)

class SessionValidator:
    """Helper class to validate sessions when database is available."""
    
    def __init__(self):
        self._db = None
    
    def set_db_connector(self, db_connector):
        """Set the database connector (called during startup)."""
        self._db = db_connector
    
    def validate_session(self, session_id: str) -> Optional[dict]:
        """Validate session and return session data."""
        if not self._db:
            return None  # No database available, skip session validation
            
        try:
            query = """
                SELECT session_id, member_id, member_email, member_nm, expires_at
                FROM user_sessions 
                WHERE session_id = %s AND is_active = true AND expires_at > NOW()
            """
            result = self._db.execute_query(query, (session_id,))
            
            if result:
                # Update last_accessed
                self._db.execute_query(
                    "UPDATE user_sessions SET last_accessed = NOW() WHERE session_id = %s",
                    (session_id,)
                )
                return dict(result[0])
            return None
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None

# Global session validator
_session_validator = SessionValidator()

def set_db_connector(db_connector):
    """Set database connector for session validation (called during startup)."""
    _session_validator.set_db_connector(db_connector)

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """
    Verify the API key provided in the X-API-Key header.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key header missing")
    
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return api_key

async def verify_session(session_id: Optional[str] = Depends(session_header)):
    """
    Verify the session ID provided in the X-Session-ID header.
    Returns None if no session ID provided or database not available.
    """
    if session_id is None:
        return None  # Optional session
    
    session_data = _session_validator.validate_session(session_id)
    if not session_data:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return session_data

async def optional_session(session_id: Optional[str] = Depends(session_header)):
    """
    Optional session validation for endpoints that can work with or without sessions.
    Returns session data if valid, None otherwise.
    """
    if session_id is None:
        return None
    
    session_data = _session_validator.validate_session(session_id)
    return session_data  # Returns None if invalid, no exception raised

async def verify_api_key_and_member_id(
    x_api_key: str = Header(..., alias="X-API-Key"),
    member_id: str = Header(..., alias="X-Member-ID")
):
    """
    Verify API key and extract member_id from headers.
    This is for SpringBoot session control where member_id is passed with each request.
    """
    # Verify API key using existing logic
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API Key header missing")
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Verify member_id is provided
    if not member_id or member_id.strip() == "":
        raise HTTPException(status_code=400, detail="Member ID required")
        
    return {
        "api_key_valid": True,
        "member_id": member_id.strip()
    }

async def verify_api_key_and_optional_session(
    api_key: Optional[str] = Depends(api_key_header),
    session_id: Optional[str] = Depends(session_header)
):
    """
    Verify API key and optionally validate session if provided.
    This is the main dependency for most endpoints.
    """
    # Verify API key (required)
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key header missing")
    
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Validate session if provided (optional)
    session_data = None
    if session_id:
        session_data = _session_validator.validate_session(session_id)
        # Don't raise exception for invalid session, just set to None
        # This allows backward compatibility
    
    return {
        "api_key": api_key,
        "session_data": session_data
    }

async def require_valid_session(
    api_key: Optional[str] = Depends(api_key_header),
    session_id: Optional[str] = Depends(session_header)
):
    """
    Require both valid API key and valid session.
    Use this for endpoints that specifically need session support.
    """
    # Verify API key
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key header missing")
    
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Verify session (required)
    if session_id is None:
        raise HTTPException(status_code=401, detail="Session ID header missing")
    
    session_data = _session_validator.validate_session(session_id)
    if not session_data:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return {
        "api_key": api_key,
        "session_data": session_data
    }

# Function to get the current API key (for sharing with other services)
def get_current_api_key():
    """Get the current API key for use in other services."""
    return API_KEY

# Class-based middleware for FastAPI (enhanced version)
class APIKeyMiddleware:
    """
    Middleware to check API key for all endpoints except those in excluded_paths.
    Enhanced with optional session support.
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
        
        # Check for optional session
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            session_data = _session_validator.validate_session(session_id)
            # Add session data to request state for use in route handlers
            request.state.session_data = session_data
        else:
            request.state.session_data = None
            
        # Continue processing the request
        return await call_next(request)