# src/core/file_handlers/factory.py
from .base_handler import FileHandler
from .pdf_handler import PDFHandler
from .image_handler import ImageHandler
from .hwp_handler import HWPHandler
from .doc_handler import AdvancedDocHandler
from .msg_handler import MSGHandler
from typing import Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)

class FileHandlerFactory:
    _instance = None
    _handlers: Dict[str, Type[FileHandler]] = {}
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
            }
            cls._initialized = True
            logger.info("FileHandlerFactory initialized with ModelManager and handlers registered")
        else:
            logger.debug("FileHandlerFactory already initialized, skipping.")

    @classmethod
    def get_handler_for_extension(cls, extension: str) -> Optional[FileHandler]:
        """Get an initialized handler for a file extension."""
        logger.debug(f"Creating handler for extension '{extension}' from {__name__}")
        if not cls._initialized:
            logger.error("FileHandlerFactory not initialized. Call initialize() first.")
            return None

        ext = extension.lower().lstrip('.')
        
        if ext not in cls._handlers:
            logger.warning(f"No handler registered for extension: {ext}")
            return None
        
        handler_class = cls._handlers[ext]
        
        # Initialize handler with model_manager if supported
        if cls._model_manager is not None and 'model_manager' in handler_class.__init__.__code__.co_varnames:
            return handler_class(model_manager=cls._model_manager)
        else:
            return handler_class()