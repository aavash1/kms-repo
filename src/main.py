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
from apscheduler.schedulers.background import BackgroundScheduler
from src.core.services.file_utils import clean_expired_chat_vectors
from src.core.services.chat_vector_manager import get_chat_vector_manager
from datetime import datetime, timedelta
import signal
import atexit
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
from src.api.routes.chat_routes import router as chat_router
from src.core.startup import startup_event, init_service_instances, verify_initialization
from src.core.auth.auth_middleware import verify_api_key, get_current_api_key, set_db_connector

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_connector = None
    cleanup_scheduler = None
    chat_vector_mgr = None
    try:
        logger.info("Starting initialization process...")
        
        # ✅ MAIN CHANGE: startup_event() now handles both MariaDB AND PostgreSQL
        initialized_components = await startup_event()
        app.state.components = initialized_components
        init_service_instances(initialized_components)
        verify_initialization(initialized_components)

        # ✅ Get database connectors from initialized components
        postgresql_db = initialized_components.get('postgresql_db')
        
        # ✅ Initialize MariaDB connection (keep existing logic for MariaDB)
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

        # ✅ Set PostgreSQL in app state if available (already initialized in startup_event)
        if postgresql_db:
            app.state.postgresql_db = postgresql_db
            # Set database connector for auth middleware
            set_db_connector(postgresql_db)
            logger.info("PostgreSQL connector available from startup initialization")
        else:
            logger.warning("PostgreSQL not available from startup. Session management will use file storage.")

        # ✅ Initialize ChatHistoryManager (already handled in startup_event if PostgreSQL is available)
        if 'chat_history_manager' not in initialized_components and postgresql_db:
            from src.core.services.chat_history_manager import ChatHistoryManager
            
            batch_manager = None
            if 'query_service' in initialized_components:
                batch_manager = initialized_components['query_service'].batch_manager
            
            chat_history_manager = ChatHistoryManager(
                batch_manager=batch_manager,
                db_connector=postgresql_db
            )
            initialized_components['chat_history_manager'] = chat_history_manager
            logger.info("ChatHistoryManager initialized with PostgreSQL support in main.py")

        # Initialize and start the ChatVectorManager
        logger.info("Starting ChatVectorManager...")
        chat_vector_mgr = get_chat_vector_manager()
        chat_vector_mgr.start()
        app.state.chat_vector_manager = chat_vector_mgr
        
        logger.info("Setting up chat vector cleanup scheduler...")
        cleanup_scheduler = BackgroundScheduler()
        cleanup_scheduler.add_job(
            clean_expired_chat_vectors, 
            'interval', 
            days=1,  # Run daily to check for expired vectors
            args=[7],  # Keep vectors for 7 days
            id='cleanup_chat_vectors',
            next_run_time=datetime.now() + timedelta(minutes=5)  # First run 5 minutes after startup
        )
        cleanup_scheduler.start()
        logger.info("Cleanup scheduler started successfully")
        
        api_key = get_current_api_key()    
        logger.info("Application initialization completed successfully")
        yield
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        raise
    finally:
        try:
            from src.core.file_handlers.factory import FileHandlerFactory
            FileHandlerFactory.cleanup_on_shutdown()
        except Exception as e:
            logger.warning(f"Failed to cleanup file handlers: {str(e)}")
        # Shutdown ChatVectorManager
        if chat_vector_mgr:
            try:
                chat_vector_mgr.shutdown()
                logger.info("ChatVectorManager shut down successfully")
            except Exception as e:
                logger.warning(f"Failed to shut down ChatVectorManager: {str(e)}")
                
        # Shutdown cleanup scheduler
        if cleanup_scheduler:
            try:
                cleanup_scheduler.shutdown()
                logger.info("Cleanup scheduler shut down successfully")
            except Exception as e:
                logger.warning(f"Failed to shut down cleanup scheduler: {str(e)}")
                
        # Shutdown query service components
        if hasattr(app.state, 'components') and isinstance(app.state.components, dict):
            if 'query_service' in app.state.components:
                query_service = app.state.components['query_service']
                try:
                    # Shutdown batch managers if they exist
                    if hasattr(query_service, 'batch_manager'):
                        await query_service.batch_manager.shutdown()
                        logger.info("QueryService batch manager shut down")
                    if hasattr(query_service, 'analysis_batch_manager'):
                        await query_service.analysis_batch_manager.shutdown()
                        logger.info("Analysis batch manager shut down")
                    
                    # Save RL policy
                    query_service.save_policy("policy_network.pth")
                    logger.info("Saved RL policy to policy_network.pth during shutdown")
                except Exception as e:
                    logger.warning(f"Error during query_service shutdown: {e}")
                    
        # Close MariaDB connection
        if db_connector:
            if db_connector.is_connection_active():
                try:
                    db_connector.close()
                    logger.info("MariaDB connection closed during shutdown")
                except Exception as e:
                    logger.warning(f"Failed to close MariaDB connection: {str(e)}")
            else:
                logger.warning("MariaDB connection was not active during shutdown, skipping close.")
                
        # Close PostgreSQL connections (get from components, not local variable)
        if hasattr(app.state, 'components') and 'postgresql_db' in app.state.components:
            postgresql_connector = app.state.components['postgresql_db']
            if postgresql_connector:
                try:
                    postgresql_connector.close_all_connections()
                    logger.info("PostgreSQL connections closed during shutdown")
                except Exception as e:
                    logger.warning(f"Failed to close PostgreSQL connections: {str(e)}")
                    
        # Cleanup ModelManager
        if hasattr(app.state, 'components') and isinstance(app.state.components, dict):
            if 'model_manager' in app.state.components:
                try:
                    logger.info("Cleaning up ModelManager...")
                    app.state.components['model_manager'].cleanup()
                except Exception as e:
                    logger.warning(f"Error during ModelManager cleanup: {e}")

def create_app():
    app = FastAPI(
        title="Document Retrieval API with ChromaDB & Ollama hosted opensource Model",
        lifespan=lifespan, docs_url="/api/docs"
    )
    app.include_router(query_router, prefix="/query", tags=["Query"], dependencies=[Depends(verify_api_key)])
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"], dependencies=[Depends(verify_api_key)])
    app.include_router(chat_router, prefix="/chat", tags=["ChatRoute"], dependencies=[Depends(verify_api_key)])
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
    
    streamlit_process = None
    
    # Function to handle termination and cleanup
    def cleanup_processes():
        if streamlit_process:
            logger.info(f"Terminating Streamlit process (PID {streamlit_process.pid})...")
            streamlit_process.terminate()
            streamlit_process.wait()  # Wait for process to terminate
            logger.info("Streamlit process terminated successfully")
    
    # Register the cleanup function to run on exit
    atexit.register(cleanup_processes)
    
    # Set up signal handling
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        cleanup_processes()
        sys.exit(0)
    
    # Register signal handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    if args.ui:
        # Only run Streamlit UI
        logger.info("Starting Streamlit UI only...")
        streamlit_process = run_streamlit()
        # Keep the main process running until interrupted
        try:
            # Wait for the Streamlit process to complete (which it won't unless terminated)
            streamlit_process.wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            cleanup_processes()
    else:
        # Only run FastAPI server
        logger.info(f"Starting FastAPI server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)