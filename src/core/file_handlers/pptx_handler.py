
from spire.presentation import Presentation, IAutoShape, ITable, SlidePicture, PictureShape

import os
import logging
from typing import List
from pathlib import Path
import tempfile
import asyncio

from PIL import Image
import io
import torch

from src.core.file_handlers.base_handler import FileHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PPTXHandler(FileHandler):
    """
    Handler for PowerPoint files (.ppt and .pptx) using Spire.Presentation
    with robust error handling and fallbacks.
    """

    def __init__(self, model_manager=None):
        """Initialize the PPTX handler."""
        self.model_manager = model_manager
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vision_model = "gemma3:12b"  # Default vision model
        self.prompt = "Extract all readable text from this image and format it as structured Markdown."

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PPT/PPTX file with robust error handling.
        Falls back to a simpler implementation if the advanced features fail.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.ppt', '.pptx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .ppt and .pptx are supported.")

        try:
            # First try with spire.presentation
            try:
               

                loop = asyncio.get_running_loop()
                def sync_extract():
                    presentation = Presentation()
                    try:
                        presentation.LoadFromFile(file_path)
                        text = []
                        for slide_idx, slide in enumerate(presentation.Slides, 1):
                            slide_text = [f"\n=== Slide {slide_idx} ==="]
                            for shape in slide.Shapes:
                                if isinstance(shape, IAutoShape):
                                    for paragraph in shape.TextFrame.Paragraphs:
                                        if paragraph.Text:
                                            slide_text.append(paragraph.Text.strip())
                            text.extend(slide_text)
                        return '\n'.join(text)
                    except Exception as e:
                        logger.error(f"Error in Spire.Presentation processing: {e}")
                        return ""
                    finally:
                        presentation.Dispose()

                text = await loop.run_in_executor(None, sync_extract)
                if text:
                    return text

                # If text extraction failed, try image extraction
                image_text = await self._extract_text_from_images(file_path)
                if image_text:
                    return f"=== IMAGE TEXT ===\n{image_text}"
                    
            except ImportError as e:
                logger.warning(f"Spire.Presentation not available: {e}. Using fallback method.")
                
            # Fallback method 1: Try using comtypes with PowerPoint automation
            try:
                import comtypes.client
                
                def extract_with_comtypes():
                    comtypes.CoInitialize()
                    try:
                        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
                        presentation = powerpoint.Presentations.Open(file_path, WithWindow=False)
                        
                        text_parts = []
                        for i in range(1, presentation.Slides.Count + 1):
                            slide = presentation.Slides.Item(i)
                            text_parts.append(f"=== Slide {i} ===")
                            
                            for shape in slide.Shapes:
                                if shape.HasTextFrame:
                                    if shape.TextFrame.HasText:
                                        text_frame = shape.TextFrame.TextRange.Text
                                        if text_frame:
                                            text_parts.append(text_frame)
                        
                        presentation.Close()
                        powerpoint.Quit()
                        return "\n".join(text_parts)
                    finally:
                        comtypes.CoUninitialize()
                
                text = await loop.run_in_executor(None, extract_with_comtypes)
                if text:
                    return text
            except Exception as e:
                logger.warning(f"Comtypes fallback failed: {e}")
            
            # If all else fails, return a helpful error message
            return "PowerPoint text extraction failed. Please ensure the required libraries are installed correctly."
            
        except Exception as e:
            logger.error(f"Error processing PowerPoint file {file_path}: {e}")
            return f"Error processing PowerPoint file: {str(e)}"

    async def _extract_text_from_images(self, file_path: str) -> str:
        """
        Extract images from a PowerPoint file and process them with a vision model.
        """
        try:
            from spire.presentation import Presentation
            from spire.presentation.common import ShapeType, PictureShape, SlidePicture
            
            presentation = Presentation()
            presentation.LoadFromFile(file_path)
            
            image_dir = Path(self.temp_dir.name) / "images"
            image_dir.mkdir(exist_ok=True)
            
            image_texts = []
            image_count = 0
            
            for slide_idx, slide in enumerate(presentation.Slides, 1):
                for shape in slide.Shapes:
                    if isinstance(shape, SlidePicture) or isinstance(shape, PictureShape):
                        try:
                            img_path = image_dir / f"slide_{slide_idx}_pic_{image_count}.png"
                            if isinstance(shape, SlidePicture):
                                img = shape.PictureFill.Picture.EmbedImage.Image
                            else:  # PictureShape
                                img = shape.EmbedImage.Image
                                
                            img.Save(str(img_path))
                            
                            # Use OCR to extract text from the image
                            if self.model_manager and hasattr(self.model_manager, 'get_trocr_processor'):
                                from PIL import Image
                                with Image.open(img_path) as pil_img:
                                    processor = self.model_manager.get_trocr_processor()
                                    model = self.model_manager.get_trocr_model()
                                    with torch.no_grad():
                                        inputs = processor(images=pil_img, return_tensors="pt").to(model.device)
                                        generated_ids = model.generate(inputs.pixel_values)
                                        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                        if extracted_text.strip():
                                            image_texts.append(f"Image {image_count} (Slide {slide_idx}): {extracted_text}")
                            
                            image_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to process image in slide {slide_idx}: {e}")
            
            presentation.Dispose()
            return "\n".join(image_texts)
            
        except Exception as e:
            logger.error(f"Error extracting images from {file_path}: {e}")
            return ""

    async def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Extract tables from a PPT/PPTX file with error handling.
        """
        try:
            from spire.presentation import Presentation
            from spire.presentation.common import ITable
            
            presentation = Presentation()
            presentation.LoadFromFile(file_path)
            
            tables = []
            for slide in presentation.Slides:
                for shape in slide.Shapes:
                    if isinstance(shape, ITable):
                        table_data = []
                        for row in shape.TableRows:
                            row_data = []
                            for i in range(row.Count):
                                cell_text = row[i].TextFrame.Text.strip()
                                row_data.append(cell_text)
                            if any(row_data):  # Skip empty rows
                                table_data.append(row_data)
                        if table_data:  # Skip empty tables
                            tables.append(table_data)
            
            presentation.Dispose()
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []

    def __del__(self):
        """Clean up temporary directory when the handler is destroyed."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()