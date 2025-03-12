# src/core/file_handlers/msg_handler.py
import os
import sys
import logging
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import extract_msg

# Ensure the project root is in the path if needed.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.file_handlers.base_handler import FileHandler
from src.core.file_handlers.factory import FileHandlerFactory
from src.core.config import load_ocr_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def extract_status_codes(text):
    """Extract status codes from text using regex patterns."""
    import re
    patterns = [
        r'[Ss]tatus\s+[Cc]ode\s+(\d+)',  # Matches "Status Code XXX" or "status code XXX"
        r'[Ss]tatus\s+(\d+)',            # Matches "Status XXX" or "status XXX"
    ]
    status_codes = set()
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            status_codes.add(match.group(1))
    return list(status_codes)

class MSGHandler(FileHandler):
    """
    Handler for Microsoft Outlook .msg files.
    
    This handler supports:
      - Extraction of email content (subject, sender, recipients, body)
      - Processing of both plain text and HTML email bodies
      - Extraction of attached files (using appropriate handlers for each file type)
      - Identification of status codes in the email content
    """
    
    def __init__(self):
        # Create a temporary directory for extracted attachments
        self.temp_dir = tempfile.TemporaryDirectory()
        self.status_codes = []
        logger.debug("MSGHandler initialized.")

    def __del__(self):
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Temporary directory cleanup failed: {e}")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from an .msg file.
        
        Parameters:
        file_path (str): Path to the .msg file
        
        Returns:
        str: Extracted text including email metadata and body
        """
        try:
            # Open the MSG file
            msg = extract_msg.Message(file_path)
            
            # Extract email metadata
            metadata = self._extract_metadata(msg)
            
            # Extract email body
            body = self._process_body(msg)
            
            # Process attachments
            attachments_text = self._process_attachments(msg)
            
            # Combine all content
            full_text = (
                f"Subject: {metadata['subject']}\n"
                f"From: {metadata['sender']}\n"
                f"To: {metadata['to']}\n"
                f"CC: {metadata['cc']}\n"
                f"Date: {metadata['date']}\n\n"
                f"{body}\n\n"
            )
            
            if attachments_text:
                full_text += "=== ATTACHMENTS ===\n" + attachments_text
            
            # Extract status codes from the full text
            all_status_codes = extract_status_codes(full_text)
            self.status_codes = all_status_codes
            
            return full_text
            
        except Exception as e:
            logger.error(f"MSG extraction failed: {e}")
            return ""
    
    def get_status_codes(self):
        """Return the list of status codes found in the last processed email."""
        return self.status_codes
    
    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from an .msg file.
        
        This method attempts to extract tables from the email body if it's in HTML format.
        It also extracts tables from any attachments that might contain tables.
        
        Parameters:
        file_path (str): Path to the .msg file
        
        Returns:
        List[List[List[str]]]: A list of tables, each table being a list of rows,
                               each row being a list of cell strings
        """
        try:
            tables = []
            
            # Open the MSG file
            msg = extract_msg.Message(file_path)
            
            # Extract tables from HTML body if available
            if msg.htmlBody:
                body_tables = self._extract_tables_from_html(msg.htmlBody)
                tables.extend(body_tables)
            
            # Extract tables from attachments
            attachment_tables = self._extract_tables_from_attachments(msg)
            tables.extend(attachment_tables)
            
            return tables
            
        except Exception as e:
            logger.error(f"MSG table extraction failed: {e}")
            return []

    def _extract_metadata(self, msg: extract_msg.Message) -> Dict[str, str]:
        """
        Extract metadata from an MSG file.
        
        Parameters:
        msg (extract_msg.Message): The MSG message object
        
        Returns:
        Dict[str, str]: Dictionary containing email metadata
        """
        metadata = {
            'subject': msg.subject or '',
            'sender': msg.sender or '',
            'to': msg.to or '',
            'cc': msg.cc or '',
            'date': str(msg.date) if msg.date else '',
        }
        return metadata

    def _process_body(self, msg: extract_msg.Message) -> str:
        """
        Process the email body, prioritizing plain text over HTML.
        
        Parameters:
        msg (extract_msg.Message): The MSG message object
        
        Returns:
        str: The processed email body text
        """
        body = ""
        
        # Try to use plain text body first
        if msg.body:
            body = msg.body
        # Fall back to HTML body if available
        elif msg.htmlBody:
            # Basic HTML-to-text conversion
            from html.parser import HTMLParser
            
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
            body = stripper.get_data()
        
        return body

    def _process_attachments(self, msg: extract_msg.Message) -> str:
        """
        Process all attachments in the email.
        
        Parameters:
        msg (extract_msg.Message): The MSG message object
        
        Returns:
        str: Text extracted from all attachments, concatenated
        """
        attachment_texts = []
        
        try:
            for attachment in msg.attachments:
                # Skip empty attachments
                if not attachment.data:
                    continue
                
                # Get attachment filename
                filename = attachment.longFilename or attachment.shortFilename or f"attachment_{len(attachment_texts)}"
                
                # Save attachment to temp directory
                temp_path = Path(self.temp_dir.name) / filename
                with open(temp_path, 'wb') as f:
                    f.write(attachment.data)
                
                # Process the attachment based on its file type
                attachment_text = self._process_single_attachment(temp_path)
                
                if attachment_text:
                    attachment_texts.append(f"--- {filename} ---\n{attachment_text}")
        
        except Exception as e:
            logger.error(f"Attachment processing failed: {e}")
        
        return "\n\n".join(attachment_texts)

    def _process_single_attachment(self, file_path: Path) -> str:
        """Process a single attachment file."""
        try:
            # Get the appropriate handler based on file extension
            ext = file_path.suffix.lower()
                
            # Use FileHandlerFactory directly instead of get_file_handler
            handler = FileHandlerFactory.get_handler_for_extension(ext)
                
            if handler:
                # Extract text using the appropriate handler
                return handler.extract_text(str(file_path))
            else:
                logger.warning(f"No handler available for file type: {ext}")
                return f"[No handler available for {ext} files]"
                    
        except Exception as e:
            logger.error(f"Single attachment processing failed: {e}")
            return f"[Error processing attachment: {e}]"

    def _extract_tables_from_html(self, html_content: str) -> List[List[List[str]]]:
        """
        Extract tables from HTML content.
        
        Parameters:
        html_content (str): HTML content
        
        Returns:
        List[List[List[str]]]: List of tables
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = []
            
            for table_elem in soup.find_all('table'):
                table_data = []
                
                for row in table_elem.find_all('tr'):
                    row_data = []
                    
                    # Process both th and td elements
                    for cell in row.find_all(['td', 'th']):
                        # Remove any nested tables to avoid duplication
                        for nested_table in cell.find_all('table'):
                            nested_table.decompose()
                        
                        # Get text content
                        cell_text = cell.get_text(strip=True)
                        row_data.append(cell_text)
                    
                    if row_data:  # Skip empty rows
                        table_data.append(row_data)
                
                if table_data:  # Skip empty tables
                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            logger.error(f"HTML table extraction failed: {e}")
            return []

    def _extract_tables_from_attachments(self, msg: extract_msg.Message) -> List[List[List[str]]]:
        """
        Extract tables from all attachments.
        
        Parameters:
        msg (extract_msg.Message): The MSG message object
        
        Returns:
        List[List[List[str]]]: List of tables from all attachments
        """
        all_tables = []
        
        try:
            for attachment in msg.attachments:
                # Skip empty attachments
                if not attachment.data:
                    continue
                
                # Get attachment filename
                filename = attachment.longFilename or attachment.shortFilename or f"attachment_{len(all_tables)}"
                
                # Save attachment to temp directory
                temp_path = Path(self.temp_dir.name) / filename
                with open(temp_path, 'wb') as f:
                    f.write(attachment.data)
                
                # Get the appropriate handler based on file extension
                ext = Path(filename).suffix.lower()
                handler = FileHandlerFactory.get_handler_for_extension(ext)
                
                if handler and hasattr(handler, 'extract_tables'):
                    # Extract tables using the appropriate handler
                    tables = handler.extract_tables(str(temp_path))
                    if tables:
                        all_tables.extend(tables)
        
        except Exception as e:
            logger.error(f"Attachment table extraction failed: {e}")
        
        return all_tables