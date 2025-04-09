# src/core/file_handlers/pptx_handler.py
import os
import logging
from typing import List
from pathlib import Path
import tempfile
from spire.presentation import Presentation
from spire.presentation.common import IAutoShape, ITable, SlidePicture, PictureShape
from PIL import Image
import io
import ollama

from src.core.file_handlers.base_handler import FileHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PPTXHandler(FileHandler):
    """
    Handler for PowerPoint files (.ppt and .pptx) using Spire.Presentation and a vision model.
    
    Supports:
    - Text extraction from slides (including shapes and paragraphs)
    - Table extraction
    - Image text extraction using a vision model (Gemma-3 via ollama)
    - Both .ppt and .pptx formats
    """

    def __init__(self):
        """Initialize the PPTX handler."""
        self.temp_dir = tempfile.TemporaryDirectory()  # For temporary image storage
        self.vision_model = "gemma3:12b"  # Default vision model; adjust as needed
        self.prompt = "Extract all readable text from these images and format it as structured Markdown."

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PowerPoint file, including text from slides and images via vision model.
        
        :param file_path: Path to the PowerPoint file (.ppt or .pptx)
        :return: Extracted text as a string, including Markdown-formatted image text
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.ppt', '.pptx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .ppt and .pptx are supported.")

        try:
            # Extract text from slides
            slide_text = self._extract_text_from_slides(file_path)
            # Extract text from images using vision model
            image_text = self._extract_text_from_images(file_path)
            # Combine with a separator
            combined_text = f"{slide_text}\n\n=== IMAGE TEXT (Markdown) ===\n{image_text}"
            logger.debug(f"Successfully extracted text from {file_path}")
            return combined_text
        except Exception as e:
            logger.error(f"Error processing PowerPoint file {file_path}: {e}")
            return ""

    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from a PowerPoint file.
        
        :param file_path: Path to the PowerPoint file
        :return: List of tables, where each table is a list of rows,
                 and each row is a list of cell strings
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.ppt', '.pptx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .ppt and .pptx are supported.")

        try:
            return self._extract_tables_from_document(file_path)
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []

    def _extract_text_from_slides(self, file_path: str) -> str:
        """
        Extract text from all slides in the PowerPoint document.
        
        :param file_path: Path to the PowerPoint file
        :return: Extracted text as a string
        """
        presentation = Presentation()
        try:
            presentation.LoadFromFile(file_path)
            text = []
            
            # Loop through all slides
            for slide_idx, slide in enumerate(presentation.Slides, 1):
                slide_text = [f"\n=== Slide {slide_idx} ==="]
                # Loop through shapes in the slide
                for shape in slide.Shapes:
                    if isinstance(shape, IAutoShape):
                        # Extract text from auto shapes (text boxes)
                        for paragraph in shape.TextFrame.Paragraphs:
                            if paragraph.Text:
                                slide_text.append(paragraph.Text.strip())
                text.extend(slide_text)
            
            return '\n'.join(text)
        finally:
            presentation.Dispose()

    def _extract_text_from_images(self, file_path: str) -> str:
        """
        Extract images from the PowerPoint document and extract text using a vision model.
        
        :param file_path: Path to the PowerPoint file
        :return: Text extracted from images as Markdown-formatted string
        """
        presentation = Presentation()
        try:
            presentation.LoadFromFile(file_path)
            image_bytes_list = []
            image_dir = Path(self.temp_dir.name) / "images"
            image_dir.mkdir(exist_ok=True)
            
            # Extract all images from the document
            image_count = 0
            for slide_idx, slide in enumerate(presentation.Slides, 1):
                for shape in slide.Shapes:
                    if isinstance(shape, SlidePicture):
                        img = shape.PictureFill.Picture.EmbedImage.Image
                        img_path = image_dir / f"slide_{slide_idx}_pic_{image_count}.png"
                        img.Save(str(img_path))
                        image_bytes_list.append(self._image_to_bytes(img_path))
                        image_count += 1
                    elif isinstance(shape, PictureShape):
                        img = shape.EmbedImage.Image
                        img_path = image_dir / f"slide_{slide_idx}_pic_{image_count}.png"
                        img.Save(str(img_path))
                        image_bytes_list.append(self._image_to_bytes(img_path))
                        image_count += 1
            
            if not image_bytes_list:
                logger.debug(f"No images found in {file_path}")
                return ""
            
            # Query the vision model with all images at once
            extracted_text = self._query_vision_model(image_bytes_list)
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting images from {file_path}: {e}")
            return ""
        finally:
            presentation.Dispose()

    def _extract_tables_from_document(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from the PowerPoint document.
        
        :param file_path: Path to the PowerPoint file
        :return: List of tables in the format [tables][rows][cells]
        """
        presentation = Presentation()
        try:
            presentation.LoadFromFile(file_path)
            tables = []
            
            # Loop through all slides
            for slide in presentation.Slides:
                for shape in slide.Shapes:
                    if isinstance(shape, ITable):
                        table_data = []
                        # Loop through table rows
                        for row in shape.TableRows:
                            row_data = []
                            # Loop through cells in the row
                            for i in range(row.Count):
                                cell_value = row[i].TextFrame.Text.strip()
                                row_data.append(cell_value)
                            table_data.append(row_data)
                        tables.append(table_data)
            
            logger.debug(f"Extracted {len(tables)} tables from {file_path}")
            return tables
        finally:
            presentation.Dispose()

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