# # src/core/file_handlers/factory.py
# from .doc_handler import DocHandler, DocxHandler
# from .hwp_handler import HWPHandler
# from .msg_handler import MsgHandler
# from .image_handler import ImageHandler
# from .pdf_handler import PDFHandler
# from .base_handler import FileHandler

# class FileHandlerFactory:
#     handlers = {
#         'application/msword': DocHandler,
#         'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocxHandler,
#         'application/hwp': HWPHandler,
#         'application/vnd.ms-outlook': MsgHandler,
#         'image/png': ImageHandler,
#         'image/jpeg': ImageHandler,
#         'image/tiff': ImageHandler,
#         'application/pdf': PDFHandler
#     }

#     @classmethod
#     def get_handler(cls, mime_type):
#         return cls.handlers.get(mime_type, cls.default_handler)()
    
#     @classmethod
#     def default_handler(cls):
#         # Fallback handler
#         return FileHandler()


####################################################
# src/core/file_handlers/factory.py
from .base_handler import FileHandler
from typing import Dict, Optional, Type
import logging
logger = logging.getLogger(__name__)

class FileHandlerFactory:
    _handlers: Dict[str, Type[FileHandler]] = {}
    _model_manager = None
    
    @classmethod
    def initialize(cls, model_manager):
        """Initialize the factory with a ModelManager instance."""
        cls._model_manager = model_manager
        logger.info("FileHandlerFactory initialized with ModelManager")
    
    @classmethod
    def get_handler_for_extension(cls, extension: str) -> Optional[FileHandler]:
        """Get an initialized handler for a file extension."""
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