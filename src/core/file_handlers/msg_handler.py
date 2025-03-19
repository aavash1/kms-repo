import os
import sys
import logging
import tempfile
import re
from pathlib import Path
from typing import List, Dict
from html.parser import HTMLParser

import extract_msg
from bs4 import BeautifulSoup

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.file_handlers.base_handler import FileHandler
from src.core.file_handlers.factory import FileHandlerFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class MSGHandler(FileHandler):
    """
    Handler for Microsoft Outlook .msg files.
    Supports extraction of email body, tables, and status codes.
    """
    
    def __init__(self, model_manager):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.status_codes = []

        self.model_manager = model_manager
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        
        logger.debug("MSGHandler initialized.")

    # def cleanup(self):
    #     """Explicitly clean up temporary directory."""
    #     try:
    #         self.temp_dir.cleanup()
    #         logger.debug("Temporary directory cleaned up.")
    #     except Exception as e:
    #         logger.warning(f"Temporary directory cleanup failed: {e}")

    def extract_text(self, file_path: str) -> str:
        """
        Extract only the body content from an .msg file.
        """
        try:
            msg = extract_msg.Message(file_path)
            body = self._process_body(msg)
            self.status_codes = self._extract_status_codes(body)  # Still extract status codes from body
            return body.strip() if body else ""
        except Exception as e:
            logger.error(f"MSG body extraction failed: {e}")
            return ""
        finally:
            self.cleanup()

    def extract_text_from_memory(self, file_content: bytes) -> str:
        """Extract body from .msg file content in memory."""
        with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            return self.extract_text(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            self.cleanup()

    def get_status_codes(self) -> List[str]:
        """Return the list of status codes found in the last processed email."""
        return self.status_codes

    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """Extract tables from an .msg file."""
        try:
            msg = extract_msg.Message(file_path)
            tables = []
            if msg.htmlBody:
                body_tables = self._extract_tables_from_html(msg.htmlBody)
                tables.extend(body_tables)
            attachment_tables = self._extract_tables_from_attachments(msg)
            tables.extend(attachment_tables)
            return tables
        except Exception as e:
            logger.error(f"MSG table extraction failed: {e}")
            return []
        finally:
            self.cleanup()

    def _extract_metadata(self, msg: extract_msg.Message) -> Dict[str, str]:
        """Extract metadata from an MSG file (not used in extract_text)."""
        return {
            'subject': msg.subject or '',
            'sender': msg.sender or '',
            'to': msg.to or '',
            'cc': msg.cc or '',
            'date': str(msg.date) if msg.date else '',
        }

    def _process_body(self, msg: extract_msg.Message) -> str:
        """Process the email body, prioritizing plain text over HTML."""
        try:
            if msg.body:
                return msg.body
            elif msg.htmlBody:
                class MLStripper(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.reset()
                        self.strict = False
                        self.convert_charrefs = True
                        self.text = []
                    def handle_data(self, d):
                        self.text.append(d)
                    def get_data(self):
                        return ''.join(self.text)
                
                stripper = MLStripper()
                stripper.feed(msg.htmlBody)
                return stripper.get_data()
            return ""
        except Exception as e:
            logger.error(f"Body processing failed: {e}")
            return ""

    def _process_attachments(self, msg: extract_msg.Message) -> str:
        """Process all attachments in the email (not used in extract_text)."""
        attachment_texts = []
        for attachment in msg.attachments:
            if not attachment.data:
                continue
            filename = attachment.longFilename or attachment.shortFilename or f"attachment_{len(attachment_texts)}"
            temp_path = Path(self.temp_dir.name) / filename
            with open(temp_path, 'wb') as f:
                f.write(attachment.data)
            attachment_text = self._process_single_attachment(temp_path)
            if attachment_text:
                attachment_texts.append(f"--- {filename} ---\n{attachment_text}")
        return "\n\n".join(attachment_texts)

    def _process_single_attachment(self, file_path: Path) -> str:
        try:
            ext = file_path.suffix.lower()
            handler = FileHandlerFactory.get_handler_for_extension(ext)
            if handler:
                if hasattr(handler, 'extract_text_from_memory'):
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    if ext in ['png', 'jpg', 'jpeg']:
                        # Use handwritten_processor instead of trocr_processor
                        text = handler.extract_text_from_memory(file_content, self.handwritten_processor, self.handwritten_model)
                        return text if text else handler.extract_text(str(file_path))
                    return handler.extract_text(str(file_path))
                return handler.extract_text(str(file_path))
            logger.warning(f"No handler available for file type: {ext}")
            return f"[No handler available for {ext} files]"
        except Exception as e:
            logger.error(f"Single attachment processing failed: {e}")
            return f"[Error processing attachment: {e}]"

    def _extract_tables_from_html(self, html_content: str) -> List[List[List[str]]]:
        """Extract tables from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = []
            for table_elem in soup.find_all('table'):
                table_data = []
                for row in table_elem.find_all('tr'):
                    row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if row_data:
                        table_data.append(row_data)
                if table_data:
                    tables.append(table_data)
            return tables
        except Exception as e:
            logger.error(f"HTML table extraction failed: {e}")
            return []

    def _extract_tables_from_attachments(self, msg: extract_msg.Message) -> List[List[List[str]]]:
        """Extract tables from all attachments."""
        all_tables = []
        for attachment in msg.attachments:
            if not attachment.data:
                continue
            filename = attachment.longFilename or attachment.shortFilename or f"attachment_{len(all_tables)}"
            temp_path = Path(self.temp_dir.name) / filename
            with open(temp_path, 'wb') as f:
                f.write(attachment.data)
            ext = Path(filename).suffix.lower()
            handler = FileHandlerFactory.get_handler_for_extension(ext)
            if handler and hasattr(handler, 'extract_tables'):
                tables = handler.extract_tables(str(temp_path))
                all_tables.extend(tables)
        return all_tables

    def _extract_status_codes(self, text: str) -> List[str]:
        """Extract status codes from text using regex patterns."""
        patterns = [
            r'[Ss]tatus\s+[Cc]ode\s+(\d+)',
            r'[Ss]tatus\s+(\d+)',
        ]
        status_codes = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                status_codes.add(match.group(1))
        return list(status_codes)