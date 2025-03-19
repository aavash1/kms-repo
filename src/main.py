# src/main.py
import os
import sys
from fastapi import FastAPI, Depends, Request, HTTPException
import uvicorn
import argparse
from dotenv import load_dotenv
import warnings
import logging
from contextlib import asynccontextmanager
import streamlit as st
import transformers
import multiprocessing

from src.core.startup import startup_event, get_components

transformers.logging.set_verbosity_error()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")  # Added file handler for persistent logs
    ]
)

# Suppress verbose loggers
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
from src.core.startup import startup_event, init_service_instances, verify_initialization
from src.core.auth.auth_middleware import verify_api_key, get_current_api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application state before serving requests and clean up on shutdown.

    Responsibilities:
    - Initializes core components (ChromaDB, vector store, etc.).
    - Sets up the MariaDB connection and stores it in the app state.
    - Ensures proper cleanup of resources (e.g., closing DB connection).

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    try:
        # Run startup event and store initialized components
        logger.info("Starting initialization process...")
        initialized_components = startup_event()
        app.state.components = initialized_components
        init_service_instances(initialized_components)
        verify_initialization(initialized_components)

        # Initialize MariaDB connection with retry logic
        from src.core.mariadb_db.mariadb_connector import MariaDBConnector
        db_connector = MariaDBConnector()
        max_retries = 3
        delay = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to MariaDB (Attempt {attempt + 1}/{max_retries})...")
                db_connector.connect()
                app.state.db_connector = db_connector  # Store in app state for access
                logger.info("MariaDB connection established successfully.")
                break
            except ConnectionError as e:
                logger.error(f"MariaDB connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(delay)
                else:
                    raise
        api_key = get_current_api_key()    
        logger.info("Application initialization completed successfully, including MariaDB")
        yield
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up: Close MariaDB connection if it exists and is active
        db_connector = getattr(app.state, 'db_connector', None)
        if db_connector and db_connector.is_connection_active():
            db_connector.close()
            logger.info("MariaDB connection closed during shutdown")
        elif db_connector:
            logger.warning("MariaDB connection was not active during shutdown, skipping close.")
        else:
            logger.debug("No MariaDB connection to close during shutdown")
        if hasattr(app.state, 'components') and 'model_manager' in app.state.components:
            logger.info("Cleaning up ModelManager...")
            app.state.components['model_manager'].cleanup()

def verify_initialization(components):
    """Verify that all required components are properly initialized."""
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
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Document Retrieval API with ChromaDB & Ollama Run Deepseek Model",
        lifespan=lifespan
    )

    # Include the routers with API key verification
    app.include_router(query_router, prefix="/query", tags=["Query"], dependencies=[Depends(verify_api_key)])
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"], dependencies=[Depends(verify_api_key)])

    return app

def run_streamlit():
    """Run the Streamlit application."""
    try:
        streamlit_script = os.path.join(PROJECT_ROOT, "src", "streamlit_app.py")
        if not os.path.exists(streamlit_script):
            raise FileNotFoundError(f"Streamlit script not found at {streamlit_script}")
        cmd = ["streamlit", "run", streamlit_script, "--server.port=8501"]
        import subprocess
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
#     args = parse_args()
    
#     if args.ui:
#         logger.info("Starting Streamlit UI...")
#         run_streamlit()
#     else:
#         logger.info(f"Starting FastAPI server on port {args.port}...")
#         app = create_app()
#         uvicorn.run(app, host="0.0.0.0", port=args.port, workers=4, loop="asyncio")

app = create_app()


if __name__ == "__main__":
    args = parse_args()
    
    if args.ui:
        logger.info("Starting Streamlit UI...")
        run_streamlit()
    else:
        # Calculate workers based on CPU cores
        num_workers = max(8, multiprocessing.cpu_count())  # Adjusted to 16 workers on your 8-core system
        logger.info(f"Starting FastAPI server on port {args.port} with {num_workers} workers...")
        # Run with Gunicorn + Uvicorn for production (or Uvicorn directly on Windows)
        if os.name == 'nt':  # Windows
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        else:
            os.system(f"gunicorn -w {num_workers} -k uvicorn.workers.UvicornWorker src.main:app --bind 0.0.0.0:{args.port}")