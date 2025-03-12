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

class FileHandlerFactory:
    """
    Factory class for creating file handlers based on MIME type or file extension.
    Uses lazy imports to avoid circular dependencies.
    """
    
    @classmethod
    def get_handler(cls, mime_type):
        """
        Get the appropriate file handler for the given MIME type.
        
        Args:
            mime_type: MIME type of the file
            
        Returns:
            An instance of the appropriate FileHandler subclass
        """
        # Import handlers only when needed to avoid circular imports
        if mime_type == 'application/msword' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            from .doc_handler import AdvancedDocHandler
            return AdvancedDocHandler()
        elif mime_type == 'application/x-hwp':
            from .hwp_handler import HWPHandler
            return HWPHandler()
        elif mime_type == 'application/vnd.ms-outlook':
            from .msg_handler import MSGHandler
            return MSGHandler()
        elif mime_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff']:
            from .image_handler import ImageHandler
            return ImageHandler()
        elif mime_type == 'application/pdf':
            from .pdf_handler import PDFHandler
            return PDFHandler()
        else:
            # Return base FileHandler for unsupported types
            return FileHandler()
    
    @classmethod
    def get_handler_for_extension(cls, file_extension):
        """
        Get the appropriate file handler based on file extension.
        
        Args:
            file_extension: File extension (with or without the dot)
            
        Returns:
            An instance of the appropriate FileHandler subclass
        """
        # Normalize extension
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
            
        extension_to_mime = {
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'hwp': 'application/x-hwp',
            'msg': 'application/vnd.ms-outlook',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'tiff': 'image/tiff',
            'tif': 'image/tiff',
            'pdf': 'application/pdf'
        }
        
        mime_type = extension_to_mime.get(file_extension.lower())
        if mime_type:
            return cls.get_handler(mime_type)
        else:
            # Return base handler for unsupported extensions
            return FileHandler()