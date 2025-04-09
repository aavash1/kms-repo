# src/core/file_handlers/excel_handler.py
import os
import logging
from typing import List
from pathlib import Path
import tempfile
from zipfile import ZipFile
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
import xlrd
from PIL import Image
import io
import ollama

from src.core.file_handlers.base_handler import FileHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ExcelHandler(FileHandler):
    """
    Handler for Excel files (.xls and .xlsx) with text and table extraction, plus vision-based image text extraction.
    
    Supports:
    - Text extraction from sheets (all or specific)
    - Table extraction as structured data
    - Image text extraction using a vision model (for .xlsx only)
    - Both .xls and .xlsx formats
    """

    def __init__(self):
        """Initialize the Excel handler."""
        self.temp_dir = tempfile.TemporaryDirectory()  # For temporary image storage
        self.vision_model = "gemma3:12b"  # Default vision model; adjust as needed
        self.prompt = "Extract all readable text from these images and format it as structured Markdown."

    def extract_text(self, file_path: str, sheet_name: str = None) -> str:
        """
        Extract text from an Excel file, including text from sheets and images via vision model.
        
        :param file_path: Path to the Excel file (.xls or .xlsx)
        :param sheet_name: Specific sheet to extract (None for all sheets)
        :return: Extracted text as a string, including Markdown-formatted image text
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
            sheet_text = self._extract_text_from_sheets(file_path, sheet_name)
            # Extract text from images (only for .xlsx)
            image_text = self._extract_text_from_images(file_path)
            # Combine with a separator
            combined_text = f"{sheet_text}\n\n=== IMAGE TEXT (Markdown) ===\n{image_text}"
            logger.debug(f"Successfully extracted text from {file_path}")
            return combined_text
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            return ""

    def extract_tables(self, file_path: str, sheet_name: str = None) -> List[List[List[str]]]:
        """
        Extract tables from an Excel file (each sheet treated as a table).
        
        :param file_path: Path to the Excel file
        :param sheet_name: Specific sheet to extract (None for all sheets)
        :return: List of tables, where each table is a list of rows,
                 and each row is a list of cell strings
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.xls', '.xlsx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .xls and .xlsx are supported.")

        try:
            return self._extract_tables_from_document(file_path, sheet_name)
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []

    def _extract_text_from_sheets(self, file_path: str, sheet_name: str = None) -> str:
        """
        Extract text from Excel sheets.
        
        :param file_path: Path to the Excel file
        :param sheet_name: Specific sheet to extract (None for all sheets)
        :return: Extracted text as a string
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.xlsx':
            return self._extract_from_xlsx(file_path, sheet_name)
        elif ext == '.xls':
            return self._extract_from_xls(file_path, sheet_name)

    def _extract_from_xlsx(self, file_path: str, sheet_name: str = None) -> str:
        """Extract text from .xlsx files using openpyxl."""
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        try:
            text = []
            sheets = [sheet_name] if sheet_name else workbook.sheetnames
            
            for sheet in sheets:
                worksheet = workbook[sheet]
                text.append(f"=== Sheet: {sheet} ===")
                for row in worksheet.iter_rows():
                    row_text = [str(cell.value) for cell in row if cell.value is not None]
                    if row_text:
                        text.append('\t'.join(row_text))
            
            return '\n'.join(text)
        finally:
            workbook.close()

    def _extract_from_xls(self, file_path: str, sheet_name: str = None) -> str:
        """Extract text from legacy .xls files using xlrd."""
        workbook = xlrd.open_workbook(file_path)
        text = []
        
        sheets = [sheet_name] if sheet_name else workbook.sheet_names()
        
        for sheet in sheets:
            worksheet = workbook.sheet_by_name(sheet)
            text.append(f"=== Sheet: {sheet} ===")
            for row_idx in range(worksheet.nrows):
                row_text = [str(worksheet.cell_value(row_idx, col_idx))
                           for col_idx in range(worksheet.ncols)
                           if worksheet.cell_value(row_idx, col_idx)]
                if row_text:
                    text.append('\t'.join(row_text))
        
        return '\n'.join(text)

    def _extract_text_from_images(self, file_path: str) -> str:
        """
        Extract images from the Excel file and extract text using a vision model.
        
        :param file_path: Path to the Excel file
        :return: Text extracted from images as Markdown-formatted string
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.xlsx':
            return self._extract_images_from_xlsx(file_path)
        elif ext == '.xls':
            logger.warning(f"Image extraction from .xls files ({file_path}) is not supported natively.")
            return "Image extraction from .xls files is not supported without additional conversion."
        return ""

    def _extract_images_from_xlsx(self, file_path: str) -> str:
        """
        Extract images from an .xlsx file and process them with a vision model.
        
        :param file_path: Path to the Excel file
        :return: Extracted text from images as Markdown
        """
        try:
            image_bytes_list = []
            image_dir = Path(self.temp_dir.name) / "images"
            image_dir.mkdir(exist_ok=True)
            
            # Open the .xlsx file as a ZIP archive to extract images
            with ZipFile(file_path, 'r') as zip_ref:
                # Find image files in the xl/media directory
                image_files = [f for f in zip_ref.namelist() if f.startswith('xl/media/')]
                for idx, img_file in enumerate(image_files):
                    img_data = zip_ref.read(img_file)
                    img_path = image_dir / f"image_{idx}{Path(img_file).suffix}"
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    image_bytes_list.append(self._image_to_bytes(img_path))
            
            if not image_bytes_list:
                logger.debug(f"No images found in {file_path}")
                return ""
            
            # Query the vision model with all images
            extracted_text = self._query_vision_model(image_bytes_list)
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting images from {file_path}: {e}")
            return ""

    def _extract_tables_from_document(self, file_path: str, sheet_name: str = None) -> List[List[List[str]]]:
        """
        Extract tables from the Excel file (each sheet as a table).
        
        :param file_path: Path to the Excel file
        :param sheet_name: Specific sheet to extract (None for all sheets)
        :return: List of tables in the format [tables][rows][cells]
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.xlsx':
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            try:
                tables = []
                sheets = [sheet_name] if sheet_name else workbook.sheetnames
                
                for sheet in sheets:
                    worksheet = workbook[sheet]
                    table_data = []
                    for row in worksheet.iter_rows():
                        row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                        if any(row_data):  # Only add non-empty rows
                            table_data.append(row_data)
                    if table_data:
                        tables.append(table_data)
                
                logger.debug(f"Extracted {len(tables)} tables from {file_path}")
                return tables
            finally:
                workbook.close()
        elif ext == '.xls':
            workbook = xlrd.open_workbook(file_path)
            tables = []
            sheets = [sheet_name] if sheet_name else workbook.sheet_names()
            
            for sheet in sheets:
                worksheet = workbook.sheet_by_name(sheet)
                table_data = []
                for row_idx in range(worksheet.nrows):
                    row_data = [str(worksheet.cell_value(row_idx, col_idx))
                               if worksheet.cell_value(row_idx, col_idx) else ""
                               for col_idx in range(worksheet.ncols)]
                    if any(row_data):  # Only add non-empty rows
                        table_data.append(row_data)
                if table_data:
                    tables.append(table_data)
            
            logger.debug(f"Extracted {len(tables)} tables from {file_path}")
            return tables

    def _image_to_bytes(self, image_path: Path) -> bytes:
        """
        Convert an image file to raw bytes for vision model input.
        
        :param image_path: Path to the image file
        :return: Raw bytes of the image in PNG format
        """
        try:
            with Image.open(image_path) as img:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                return img_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Failed to convert image {image_path} to bytes: {e}")
            return b""

    def _query_vision_model(self, image_bytes_list: List[bytes]) -> str:
        """
        Query the vision model (e.g., Gemma-3) to extract text from images.
        
        :param image_bytes_list: List of image bytes
        :return: Extracted text formatted as Markdown
        """
        try:
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": self.prompt,
                    "images": image_bytes_list
                }]
            )
            extracted_text = response["message"]["content"]
            logger.debug(f"Vision model extracted text from {len(image_bytes_list)} images")
            return extracted_text
        except Exception as e:
            logger.error(f"Vision model query failed: {e}")
            return ""

    def __del__(self):
        """Clean up temporary directory when the handler is destroyed."""
        self.temp_dir.cleanup()