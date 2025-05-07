# src/main.py
import os
import sys
from fastapi import FastAPI, Depends
import uvicorn
import argparse
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from subprocess import Popen
import transformers

transformers.logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log",encoding="utf-8")
    ]
)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('layoutparser').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.api.routes.query import router as query_router
from src.api.routes.ingest import router as ingest_router
from src.core.startup import startup_event, init_service_instances, verify_initialization
from src.core.auth.auth_middleware import verify_api_key, get_current_api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_connector = None
    try:
        logger.info("Starting initialization process...")
        initialized_components = startup_event()
        app.state.components = initialized_components
        init_service_instances(initialized_components)
        verify_initialization(initialized_components)

        from src.core.mariadb_db.mariadb_connector import MariaDBConnector
        db_connector = MariaDBConnector()
        max_retries = 3
        delay = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to MariaDB (Attempt {attempt + 1}/{max_retries})...")
                db_connector.connect()
                app.state.db_connector = db_connector
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
        if hasattr(app.state, 'components') and 'query_service' in app.state.components:
            query_service = app.state.components['query_service']
            query_service.save_policy("policy_network.pth")
            logger.info("Saved RL policy to policy_network.pth during shutdown")
        
        if db_connector:
            if db_connector.is_connection_active():
                try:
                    db_connector.close()
                    logger.info("MariaDB connection closed during shutdown")
                except Exception as e:
                    logger.warning(f"Failed to close MariaDB connection: {str(e)}")
            else:
                logger.warning("MariaDB connection was not active during shutdown, skipping close.")
        if hasattr(app.state, 'components') and 'model_manager' in app.state.components:
            logger.info("Cleaning up ModelManager...")
            app.state.components['model_manager'].cleanup()

def create_app():
    app = FastAPI(
        title="Document Retrieval API with ChromaDB & Ollama Run Deepseek Model",
        lifespan=lifespan
    )
    app.include_router(query_router, prefix="/query", tags=["Query"], dependencies=[Depends(verify_api_key)])
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"], dependencies=[Depends(verify_api_key)])
    return app

def run_streamlit():
    try:
        streamlit_script = os.path.join(PROJECT_ROOT, "src", "streamlit_app.py")
        if not os.path.exists(streamlit_script):
            raise FileNotFoundError(f"Streamlit script not found at {streamlit_script}")
        cmd = ["streamlit", "run", streamlit_script, "--server.port=8501"]
        process = Popen(cmd, shell=False)
        logger.info(f"Started Streamlit on port 8501 with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to run Streamlit: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Run the NetBackup Assistant application")
    parser.add_argument("--ui", action="store_true", help="Run with Streamlit UI")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    return parser.parse_args()

app = create_app()

if __name__ == "__main__":
    args = parse_args()
    
    if args.ui:
        logger.info("Starting Streamlit UI...")
        run_streamlit()
    else:
        logger.info(f"Starting FastAPI server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)