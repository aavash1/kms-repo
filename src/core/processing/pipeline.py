# src/core/processing/pipeline.py
from pathlib import Path
from ..utils.file_identification import get_file_type
from ..file_handlers.factory import FileHandlerFactory

class ProcessingPipeline:
    def __init__(self):
        self.cache = {}
        
    def process_file(self, file_path):
        file_type = get_file_type(file_path)
        handler = FileHandlerFactory.get_handler(file_type)
        
        text = handler.extract_text(file_path)
        tables = handler.extract_tables(file_path)
        
        return {
            'text': self._post_process(text),
            'tables': tables,
            'metadata': self._extract_metadata(file_path)
        }

    def _post_process(self, text):
        from ..utils.post_processing import (
            clean_extracted_text,
            fix_hyphenation,
            preserve_line_breaks
        )
        text = fix_hyphenation(text)
        text = preserve_line_breaks(text)
        return clean_extracted_text(text)

    def _extract_metadata(self, file_path):
        return {
            'path': str(file_path),
            'size': Path(file_path).stat().st_size,
            'modified': Path(file_path).stat().st_mtime
        }