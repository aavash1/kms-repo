# #src/core/file_handlers/base_handler.py
# from abc import ABC, abstractmethod

# class FileHandler(ABC):
#     @abstractmethod
#     def extract_text(self, file_path):
#         pass

#     @abstractmethod
#     def extract_tables(self, file_path):
#         pass

#############################################

# src/core/file_handlers/base_handler.py
from abc import ABC, abstractmethod

class FileHandler(ABC):
    @abstractmethod
    def extract_text(self, file_path):
        pass

    def extract_text_from_memory(self, file_content):
        """
        Default implementation that saves to a temporary file.
        Subclasses should override with more efficient implementations.
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(file_content)
        
        try:
            return self.extract_text(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def extract_tables(self, file_path):
        """
        Default implementation returns an empty list.
        Subclasses should override if they support table extraction.
        """
        return []
    
    def get_status_codes(self):
        """
        Default implementation returns an empty list.
        Subclasses should override if they extract status codes.
        """
        return []