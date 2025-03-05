#src/core/file_handlers/base_handler.py
from abc import ABC, abstractmethod

class FileHandler(ABC):
    @abstractmethod
    def extract_text(self, file_path):
        pass

    @abstractmethod
    def extract_tables(self, file_path):
        pass