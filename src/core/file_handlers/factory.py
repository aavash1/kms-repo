# src/core/file_handlers/factory.py
from .base_handler import FileHandler
from .pdf_handler import PDFHandler
from .image_handler import ImageHandler
from .hwp_handler import HWPHandler
from .doc_handler import AdvancedDocHandler
from .msg_handler import MSGHandler
from .pptx_handler import PPTXHandler
from .excel_handler import ExcelHandler
from .htmlcontent_handler import HTMLContentHandler
from .txt_handler import TXTHandler
from .rtf_handler import RTFHandler
from typing import Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)

class FileHandlerFactory:
    _instance = None
    _handlers: Dict[str, Type[FileHandler]] = {}
    _handler_instances: Dict[str, FileHandler] = {}  # NEW: Cache for instances
    _model_manager = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_manager):
        """Initialize the factory with a ModelManager instance."""
        if not cls._initialized:
            cls._model_manager = model_manager
            cls._handlers = {
                'pdf': PDFHandler,
                'png': ImageHandler,
                'jpg': ImageHandler,
                'jpeg': ImageHandler,
                'hwp': HWPHandler,
                'doc': AdvancedDocHandler,
                'docx': AdvancedDocHandler,
                'msg': MSGHandler,
                'ppt': PPTXHandler,  
                'pptx': PPTXHandler,  
                'xls': ExcelHandler,
                'xlsx': ExcelHandler,
                'html': HTMLContentHandler,
                'excel': ExcelHandler,
                'txt': TXTHandler,
                'rtf': RTFHandler,
            }
            cls._handler_instances = {}  # NEW: Initialize instance cache
            cls._initialized = True
            logger.info("FileHandlerFactory initialized with ModelManager and handlers registered")
        else:
            logger.debug("FileHandlerFactory already initialized, skipping.")

    @classmethod
    def get_handler_for_extension(cls, extension: str) -> Optional[FileHandler]:
        """Get a cached handler instance for a file extension."""
        logger.debug(f"Getting handler for extension '{extension}' from {__name__}")
        if not cls._initialized:
            logger.error("FileHandlerFactory not initialized. Call initialize() first.")
            return None

        ext = extension.lower().lstrip('.')
        
        if ext not in cls._handlers:
            logger.warning(f"No handler registered for extension: {ext}")
            return None
        
        # NEW: Check if we already have a cached instance
        if ext in cls._handler_instances:
            logger.debug(f"Reusing cached handler instance for extension: {ext}")
            return cls._handler_instances[ext]
        
        # NEW: Create and cache the instance
        handler_class = cls._handlers[ext]
        
        # Initialize handler with model_manager if supported
        if cls._model_manager is not None and 'model_manager' in handler_class.__init__.__code__.co_varnames:
            handler_instance = handler_class(model_manager=cls._model_manager)
        else:
            handler_instance = handler_class()
        
        # NEW: Cache the instance for future use
        cls._handler_instances[ext] = handler_instance
        logger.debug(f"Created and cached new handler instance for extension: {ext}")
        
        return handler_instance

    @classmethod
    def clear_cache(cls):
        """Clear the handler instance cache (useful for testing or cleanup)."""
        cls._handler_instances.clear()
        logger.info("FileHandlerFactory instance cache cleared")

    @classmethod
    def get_cached_handlers_count(cls) -> int:
        """Get the number of cached handler instances."""
        return len(cls._handler_instances)
    
    @classmethod
    def cleanup_on_shutdown(cls):
        """Clean up handler instances and their resources on application shutdown."""
        import shutil
        
        for ext, handler in cls._handler_instances.items():
            try:
                # Clean up temp directories
                if hasattr(handler, 'temp_dir') and handler.temp_dir:
                    shutil.rmtree(handler.temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned temp directory for {ext} handler")
                
                # Release model references
                if hasattr(handler, 'model_manager'):
                    handler.model_manager = None
                    
                # Close any open file handles
                if hasattr(handler, 'close'):
                    handler.close()
                    
                logger.debug(f"Cleaned up handler for {ext}")
            except Exception as e:
                logger.warning(f"Error cleaning up handler {ext}: {e}")
        
        cls._handler_instances.clear()
        logger.info("All file handlers cleaned up")