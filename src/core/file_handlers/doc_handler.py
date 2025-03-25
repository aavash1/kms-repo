# src/core/file_handlers/doc_handler.py
import os
import sys
import logging
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel

import cv2
import numpy as np
from PIL import Image

from docx import Document
import docx2txt

# Import the PDFHandler for processing converted PDFs.
from .pdf_handler import PDFHandler

# Ensure the project root is in the path if needed.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ocr.tesseract_wrapper import TesseractOCR
from src.core.file_handlers.base_handler import FileHandler
from src.core.config import load_ocr_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Set PDFMiner logger to WARNING to hide debug messages
logging.getLogger("pdfminer").setLevel(logging.WARNING)

class AdvancedDocHandler(FileHandler):
    """
    Enhanced DOC/DOCX handler with layout preservation and fallback mechanisms.

    This handler supports:
      - Extraction of text (native and via OCR for embedded images)
      - Extraction of tables (from DOCX files directly or via PDF conversion for DOC files)
      - Processing of both legacy .doc and modern .docx files.
      
    For legacy .doc files, LibreOffice is used to convert the file to PDF, which is then
    processed by the PDFHandler. If the conversion fails, a fallback method using antiword
    is attempted.
    """

    def __init__(self, model_manager):
        # Initialize the Tesseract OCR wrapper.
        self.ocr = TesseractOCR()
        # Initialize the PDF handler to process PDF-converted documents.
        self.pdf_handler = PDFHandler(model_manager=model_manager)
        # Create a temporary directory for intermediate files.
        self.temp_dir = tempfile.TemporaryDirectory()

        self.model_manager = model_manager
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        self.bert_tokenizer = model_manager.get_klue_tokenizer()
        self.bert_model = model_manager.get_klue_bert()

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a DOC or DOCX file with layout preservation.
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.docx':
            return self._process_docx(file_path)
        elif ext == '.doc':
            return self._process_doc(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return ""

    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from a DOC or DOCX file.

        For DOCX files, tables are extracted directly. For DOC files,
        the file is converted to PDF and tables are extracted using PDFHandler.
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.docx':
            return self._extract_docx_tables(file_path)
        elif ext == '.doc':
            pdf_path = self._convert_doc_to_pdf(file_path)
            if pdf_path and pdf_path.exists():
                try:
                    tables = self.pdf_handler.extract_tables(str(pdf_path))
                    return tables
                finally:
                    pdf_path.unlink()
            else:
                logger.error("PDF conversion failed for table extraction.")
                return []
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return []

    def _process_doc(self, file_path: str) -> str:
        """
        Process a legacy .doc file.

        The DOC file is first converted to a PDF using LibreOffice. Then, text is
        extracted from the PDF using the PDFHandler. If PDF conversion fails, a fallback
        using antiword is attempted.
        """
        try:
            pdf_path = self._convert_doc_to_pdf(file_path)
            if pdf_path and pdf_path.exists():
                logger.debug(f"DOC converted to PDF: {pdf_path}")
                text = self.pdf_handler.extract_text(str(pdf_path))
                pdf_path.unlink()  # Clean up the temporary PDF.
                return text
            else:
                raise Exception("PDF conversion returned no file.")
        except Exception as e:
            logger.error(f"LibreOffice PDF conversion failed: {e}")
            # Fallback to antiword extraction.
            text = self._extract_via_antiword(file_path)
            return text

    def _convert_doc_to_pdf(self, file_path: str) -> Optional[Path]:
        """
        Convert a DOC file to PDF using LibreOffice's command-line interface.
        The command used is:
            soffice --headless --convert-to pdf --outdir {temp_dir} {file_path}
        """
        try:
            # Load configuration.
            config = load_ocr_config()
            libreoffice_config = config.get('libreoffice', {})
            soffice_path = libreoffice_config.get('path', None)

            if not soffice_path or not os.path.exists(soffice_path):
                logger.error("LibreOffice path not found in config or does not exist. Please check your config file.")
                return None

            output_dir = Path(self.temp_dir.name)
            cmd = [
                soffice_path,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(file_path)
            ]
            logger.debug(f"Running LibreOffice conversion command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pdf_file = output_dir / (Path(file_path).stem + ".pdf")
            if pdf_file.exists():
                logger.debug(f"PDF conversion successful: {pdf_file}")
                return pdf_file
            else:
                raise Exception("PDF file was not created by LibreOffice conversion.")
        except Exception as e:
            logger.error(f"LibreOffice conversion error: {e}")
            return None

    def _extract_via_antiword(self, file_path: str) -> str:
        """
        Fallback text extraction for DOC files using antiword.

        Note: antiword must be installed and available in the system PATH.
        """
        try:
            if not shutil.which("antiword"):
                logger.warning("antiword not found in system PATH. Please install antiword for fallback DOC extraction.")
                return ""
            result = subprocess.run(
                ['antiword', '-m', 'UTF-8.txt', file_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True
            )
            logger.debug("Antiword extraction succeeded.")
            return result.stdout
        except Exception as e:
            logger.warning(f"Antiword extraction failed: {e}")
            return ""

    def _process_docx(self, file_path: str) -> str:
        """
        Process a DOCX file.

        Extracts text from paragraphs and tables while preserving layout.
        Additionally, extracts any embedded images and applies OCR to recover text.
        """
        try:
            doc = Document(file_path)
            text_content = self._parse_docx_structure(doc)
            image_texts = self._extract_docx_images(file_path)
            combined_text = f"{text_content}\n\n=== IMAGE TEXTS ===\n" + "\n".join(image_texts)
            return combined_text
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            return ""

    def _parse_docx_structure(self, doc: Document) -> str:
        """
        Parse the structure of a DOCX document.

        This method processes paragraphs (with basic indentation preservation)
        and tables, returning a formatted string.
        """
        content = []
        # Process paragraphs.
        for para in doc.paragraphs:
            indent = para.paragraph_format.left_indent or 0
            indent_str = " " * int(indent / 360)  # Approximate conversion from EMU to spaces.
            if para.style.name.startswith('List'):
                content.append(f"{indent_str}• {para.text}")
            else:
                content.append(f"{indent_str}{para.text}")
        # Process tables.
        for table in doc.tables:
            table_content = ["\n=== TABLE ==="]
            for row in table.rows:
                cells = [cell.text.replace('\n', ' ') for cell in row.cells]
                table_content.append(" | ".join(cells))
            content.append("\n".join(table_content))
        return "\n".join(content)

    def _extract_docx_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from a DOCX file as a list of tables, each table being a list
        of rows and each row a list of cell strings.
        """
        tables_data = []
        try:
            doc = Document(file_path)
            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    row_data = [cell.text.replace('\n', ' ') for cell in row.cells]
                    table_rows.append(row_data)
                tables_data.append(table_rows)
        except Exception as e:
            logger.error(f"Error extracting DOCX tables: {e}")
        return tables_data

    def _extract_docx_images(self, file_path: str) -> List[str]:
        """
        Extract images from a DOCX file and perform OCR on each image.

        Images are extracted to a temporary directory using docx2txt.
        """
        image_texts = []
        try:
            image_dir = Path(self.temp_dir.name) / "images"
            image_dir.mkdir(exist_ok=True)
            # Extract images; docx2txt saves images in the specified folder.
            docx2txt.process(file_path, str(image_dir))
            # Process each image file.
            for img_path in image_dir.glob("*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    processed_text = self._ocr_image(str(img_path))
                    if processed_text:
                        image_texts.append(processed_text)
        except Exception as e:
            logger.error(f"Image extraction from DOCX failed: {e}")
        return image_texts

    def _ocr_image(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Unable to read image: {image_path}")
                return ""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # Use TrOCR instead of Tesseract - fixed variable names
            from PIL import Image
            pil_img = Image.fromarray(thresh)
            with torch.no_grad():
                inputs = self.handwritten_processor(images=pil_img, return_tensors="pt").to(self.handwritten_model.device)
                generated_ids = self.handwritten_model.generate(inputs.pixel_values)
                ocr_text = self.handwritten_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return ocr_text
        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""