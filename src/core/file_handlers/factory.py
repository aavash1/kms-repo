# src/core/file_handlers/factory.py
from .doc_handler import DocHandler, DocxHandler
from .hwp_handler import HWPHandler
from .msg_handler import MsgHandler
from .image_handler import ImageHandler
from .pdf_handler import PDFHandler
from .base_handler import FileHandler

class FileHandlerFactory:
    handlers = {
        'application/msword': DocHandler,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocxHandler,
        'application/hwp': HWPHandler,
        'application/vnd.ms-outlook': MsgHandler,
        'image/png': ImageHandler,
        'image/jpeg': ImageHandler,
        'image/tiff': ImageHandler,
        'application/pdf': PDFHandler
    }

    @classmethod
    def get_handler(cls, mime_type):
        return cls.handlers.get(mime_type, cls.default_handler)()
    
    @classmethod
    def default_handler(cls):
        # Fallback handler
        return FileHandler()