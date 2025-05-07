# src/core/file_handlers/excel_handler.py
import os
import logging
from typing import List, Tuple
from pathlib import Path
import tempfile
from zipfile import ZipFile
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
import xlrd
from PIL import Image
import io
import ollama
import asyncio

from src.core.file_handlers.base_handler import FileHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ExcelHandler(FileHandler):
    """
    Handler for Excel files (.xls and .xlsx) using openpyxl and xlrd.
    """

    def __init__(self, model_manager=None):
        """Initialize the Excel handler."""
        self.model_manager = model_manager
        self.temp_dir = tempfile.TemporaryDirectory()
        
    async def extract_text(self, file_path: str, sheet_name: str = None) -> str:
        """
        Extract text from an Excel file.
        
        Args:
            file_path: Path to the Excel file (.xls or .xlsx)
            sheet_name: Specific sheet to extract (None for all sheets)
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.xls', '.xlsx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .xls and .xlsx are supported.")

        try:
            # Extract text from sheets
            sheet_text = await self._extract_text_from_sheets(file_path, sheet_name)
            logger.debug(f"Successfully extracted text from {file_path}")
            return sheet_text
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            return ""

    async def _extract_text_from_sheets(self, file_path: str, sheet_name: str = None) -> str:
        """
        Extract text from Excel sheets.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Specific sheet to extract (None for all sheets)
            
        Returns:
            Extracted text as a string
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.xlsx':
            return await self._extract_from_xlsx(file_path, sheet_name)
        elif ext == '.xls':
            return await self._extract_from_xls(file_path, sheet_name)
        return ""

    async def _extract_from_xlsx(self, file_path: str, sheet_name: str = None) -> str:
        """Extract text from XLSX file."""
        import openpyxl
        
        # Use asyncio to run this in a thread pool since openpyxl is synchronous
        def _read_xlsx():
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            try:
                text = []
                sheets = [sheet_name] if sheet_name and sheet_name in workbook.sheetnames else workbook.sheetnames
                
                for sheet in sheets:
                    worksheet = workbook[sheet]
                    text.append(f"=== Sheet: {sheet} ===")
                    
                    for row in worksheet.iter_rows():
                        row_text = []
                        for cell in row:
                            if cell.value is not None:
                                row_text.append(str(cell.value))
                        
                        if row_text:
                            text.append('\t'.join(row_text))
                
                return '\n'.join(text)
            finally:
                workbook.close()
                
        # Run synchronous code in a thread pool
        return await asyncio.to_thread(_read_xlsx)

    async def _extract_from_xls(self, file_path: str, sheet_name: str = None) -> str:
        """Extract text from XLS file."""
        import xlrd
        
        # Use asyncio to run this in a thread pool since xlrd is synchronous
        def _read_xls():
            workbook = xlrd.open_workbook(file_path)
            text = []
            
            sheets = [sheet_name] if sheet_name and sheet_name in workbook.sheet_names() else workbook.sheet_names()
            
            for sheet in sheets:
                worksheet = workbook.sheet_by_name(sheet)
                text.append(f"=== Sheet: {sheet} ===")
                
                for row_idx in range(worksheet.nrows):
                    row_text = []
                    for col_idx in range(worksheet.ncols):
                        value = worksheet.cell_value(row_idx, col_idx)
                        if value:
                            row_text.append(str(value))
                    
                    if row_text:
                        text.append('\t'.join(row_text))
            
            return '\n'.join(text)
            
        # Run synchronous code in a thread pool
        return await asyncio.to_thread(_read_xls)

    async def extract_tables(self, file_path: str, sheet_name: str = None) -> List[List[List[str]]]:
        """Extract tables from an Excel file (each sheet as a table)."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.xls', '.xlsx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .xls and .xlsx are supported.")

        try:
            if ext == '.xlsx':
                return await self._extract_tables_from_xlsx(file_path, sheet_name)
            elif ext == '.xls':
                return await self._extract_tables_from_xls(file_path, sheet_name)
            return []
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []

    async def _extract_tables_from_xlsx(self, file_path: str, sheet_name: str = None) -> List[List[List[str]]]:
        """Extract tables from XLSX file."""
        import openpyxl
        
        # Use asyncio to run this in a thread pool
        def _read_tables_xlsx():
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            try:
                tables = []
                sheets = [sheet_name] if sheet_name and sheet_name in workbook.sheetnames else workbook.sheetnames
                
                for sheet in sheets:
                    worksheet = workbook[sheet]
                    table_data = []
                    
                    for row in worksheet.iter_rows():
                        row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                        if any(row_data):  # Skip empty rows
                            table_data.append(row_data)
                    
                    if table_data:  # Skip empty tables
                        tables.append(table_data)
                
                return tables
            finally:
                workbook.close()
                
        # Run synchronous code in a thread pool
        return await asyncio.to_thread(_read_tables_xlsx)

    async def _extract_tables_from_xls(self, file_path: str, sheet_name: str = None) -> List[List[List[str]]]:
        """Extract tables from XLS file."""
        import xlrd
        
        # Use asyncio to run this in a thread pool
        def _read_tables_xls():
            workbook = xlrd.open_workbook(file_path)
            tables = []
            
            sheets = [sheet_name] if sheet_name and sheet_name in workbook.sheet_names() else workbook.sheet_names()
            
            for sheet in sheets:
                worksheet = workbook.sheet_by_name(sheet)
                table_data = []
                
                for row_idx in range(worksheet.nrows):
                    row_data = [str(worksheet.cell_value(row_idx, col_idx)) if worksheet.cell_value(row_idx, col_idx) else "" 
                               for col_idx in range(worksheet.ncols)]
                    if any(row_data):  # Skip empty rows
                        table_data.append(row_data)
                
                if table_data:  # Skip empty tables
                    tables.append(table_data)
            
            return tables
            
        # Run synchronous code in a thread pool
        return await asyncio.to_thread(_read_tables_xls)

    def __del__(self):
        """Clean up temporary directory when the handler is destroyed."""
        if hasattr(self, 'temp_dir') and self.temp_dir:
            self.temp_dir.cleanup()