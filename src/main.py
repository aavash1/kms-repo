# src/main.py
import os
import sys
from fastapi import FastAPI, Depends
import uvicorn
import argparse
from dotenv import load_dotenv
import warnings
import logging
from contextlib import asynccontextmanager
import streamlit as st

import transformers
transformers.logging.set_verbosity_error()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Optional: Suppress specific loggers that are too verbose
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('layoutparser').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Set up the project root in the module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import routers from the API modules
from src.api.routes.query import router as query_router
from src.api.routes.ingest import router as ingest_router
from src.core.startup import startup_event, init_service_instances
from src.core.auth.auth_middleware import verify_api_key, get_current_api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application state before serving requests."""
    try:
        # Run startup event and store initialized components
        logger.info("Starting initialization process...")
        initialized_components = startup_event()
        
        # Initialize service instances for dependency injection
        init_service_instances(initialized_components)
        
        # Verify all critical components
        verify_initialization(initialized_components)


        api_key=get_current_api_key()    
        logger.info("✅ Application initialization completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        raise

def verify_initialization(components):
    """Verify that all required components are properly initialized"""
    required_components = [
        'chroma_collection',
        'vector_store',
        'rag_chain',
        'query_service',
        'ingest_service',
        'document_handlers',
        'workflow',
        'memory'
    ]
    
    for component in required_components:
        if component not in components or components[component] is None:
            raise RuntimeError(f"Required component '{component}' not properly initialized")
            
    # Additional verification of document handlers
    handlers = components['document_handlers']
    required_handlers = ['pdf', 'doc', 'hwp']
    for handler in required_handlers:
        if handler not in handlers or handlers[handler] is None:
            raise RuntimeError(f"Required document handler '{handler}' not properly initialized")

def create_app():
    app = FastAPI(
        title="Document Retrieval API with ChromaDB & Ollama Run Deepseek Model",
        lifespan=lifespan)

    # Include the routers
    app.include_router(query_router, prefix="/query", tags=["Query"],dependencies=[Depends(verify_api_key)])
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"],dependencies=[Depends(verify_api_key)])

    return app

def run_streamlit():
    """Run the Streamlit application."""
    try:
        import subprocess
        streamlit_script = os.path.join(PROJECT_ROOT, "src", "streamlit_app.py")
        
        # Verify that the streamlit script exists
        if not os.path.exists(streamlit_script):
            raise FileNotFoundError(f"Streamlit script not found at {streamlit_script}")
            
        # Run Streamlit with the specified script
        cmd = ["streamlit", "run", streamlit_script, "--server.port=8501"]
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Failed to run Streamlit: {e}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the NetBackup Assistant application")
    parser.add_argument("--ui", action="store_true", help="Run with Streamlit UI")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    return parser.parse_args()



# if __name__ == "__main__":
#     uvicorn.run("src.main:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    args = parse_args()
    
    if args.ui:
        logger.info("Starting Streamlit UI...")
        run_streamlit()
    else:
        logger.info(f"Starting FastAPI server on port {args.port}...")
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)