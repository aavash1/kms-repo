# src/core/file_handlers/pptx_handler.py
"""
Enhanced PowerPoint handler with hybrid OCR support.
Integrates with your ocr_config.yaml for optimal slide image processing.
"""

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

# Import hybrid OCR system
try:
    from ..services.ocr_service import get_ocr_service
    from ..utils.config_loader import ConfigLoader
    HYBRID_OCR_AVAILABLE = True
except ImportError:
    HYBRID_OCR_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PPTXHandler(FileHandler):
    """
    Enhanced PowerPoint handler with hybrid OCR and smart image processing.
    """

    def __init__(self, model_manager=None, use_smart_ocr=True):
        """Initialize the PPTX handler with smart OCR capabilities."""
        self.model_manager = model_manager
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vision_model = "gemma3:12b"  # Default vision model
        self.prompt = "Extract all readable text from this image and format it as structured Markdown."

        # Hybrid OCR integration
        self.use_smart_ocr = use_smart_ocr and HYBRID_OCR_AVAILABLE
        if self.use_smart_ocr:
            try:
                self.ocr_service = get_ocr_service()
                self.ocr_config = ConfigLoader.load_ocr_config()
                config_status = self.ocr_service.get_config_status()
                self.smart_ocr_available = config_status["tesseract_available"] or config_status["easyocr_available"]
                logger.info(f"PPTX Handler smart OCR enabled: {self.smart_ocr_available}")
            except Exception as e:
                logger.warning(f"Smart OCR initialization failed: {e}, falling back to legacy mode")
                self.use_smart_ocr = False
                self.smart_ocr_available = False

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a PPT/PPTX file with smart OCR strategy."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in ['.ppt', '.pptx']:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format. Only .ppt and .pptx are supported.")

        try:
            # Phase 1: Extract regular text from slides
            regular_text, slide_count = await self._extract_regular_text(file_path)
            total_text_length = len(regular_text)
            
            # Phase 2: Determine OCR strategy for images
            should_process_images, ocr_mode = self._determine_smart_ocr_strategy(
                total_text_length, slide_count, "pptx"
            )
            
            text_parts = [regular_text] if regular_text else []
            
            if should_process_images:
                if self.use_smart_ocr and self.smart_ocr_available:
                    logger.info(f"PPTX has {total_text_length} chars across {slide_count} slides, using smart OCR mode: {ocr_mode}")
                    image_text = await self._extract_text_from_images_smart(file_path, ocr_mode)
                else:
                    # Use original method
                    image_text = await self._extract_text_from_images(file_path)
                
                if image_text:
                    text_parts.append(f"=== SLIDE IMAGES ===\n{image_text}")
            else:
                logger.info(f"PPTX has sufficient text ({total_text_length} chars), skipping image OCR")
            
            return "\n\n".join(text_parts) if text_parts else ""
            
        except Exception as e:
            logger.error(f"Error processing PowerPoint file {file_path}: {e}")
            return f"Error processing PowerPoint file: {str(e)}"

    async def _extract_regular_text(self, file_path: str) -> tuple[str, int]:
        """Extract regular text from PowerPoint slides."""
        try:
            loop = asyncio.get_running_loop()
            
            def sync_extract():
                presentation = Presentation()
                try:
                    presentation.LoadFromFile(file_path)
                    text_parts = []
                    slide_count = len(presentation.Slides)
                    
                    for slide_idx, slide in enumerate(presentation.Slides, 1):
                        slide_text = [f"\n=== Slide {slide_idx} ==="]
                        for shape in slide.Shapes:
                            if isinstance(shape, IAutoShape):
                                for paragraph in shape.TextFrame.Paragraphs:
                                    if paragraph.Text:
                                        slide_text.append(paragraph.Text.strip())
                        text_parts.extend(slide_text)
                    
                    return '\n'.join(text_parts), slide_count
                    
                except Exception as e:
                    logger.error(f"Error in Spire.Presentation text extraction: {e}")
                    return "", 0
                finally:
                    presentation.Dispose()

            return await loop.run_in_executor(None, sync_extract)
            
        except ImportError as e:
            logger.warning(f"Spire.Presentation not available: {e}. Using fallback method.")
            return await self._extract_regular_text_fallback(file_path)

    async def _extract_regular_text_fallback(self, file_path: str) -> tuple[str, int]:
        """Fallback text extraction using comtypes or other methods."""
        try:
            import comtypes.client
            
            def extract_with_comtypes():
                comtypes.CoInitialize()
                try:
                    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
                    presentation = powerpoint.Presentations.Open(file_path, WithWindow=False)
                    
                    text_parts = []
                    slide_count = presentation.Slides.Count
                    
                    for i in range(1, slide_count + 1):
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
                    return "\n".join(text_parts), slide_count
                finally:
                    comtypes.CoUninitialize()
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, extract_with_comtypes)
            
        except Exception as e:
            logger.warning(f"Comtypes fallback failed: {e}")
            return "PowerPoint text extraction failed. Please ensure the required libraries are installed correctly.", 1

    async def _extract_text_from_images_smart(self, file_path: str, ocr_mode: str) -> str:
        """Extract text from PowerPoint images using smart OCR strategy."""
        try:
            # Extract images from presentation
            images, slide_info = await self._extract_presentation_images(file_path)
            
            if not images:
                return ""
            
            # Determine context for better OCR
            context = self._determine_image_context_pptx(file_path, images)
            
            # Choose OCR engine based on context
            if context == "handwritten" and self.model_manager and hasattr(self.model_manager, 'get_trocr_model'):
                # Use TrOCR for handwritten content
                image_texts = await self._extract_with_trocr_pptx(images, slide_info)
            else:
                # Use smart OCR service
                engine = self._choose_ocr_engine_for_context(context)
                extracted_texts = await self.ocr_service.extract_text_from_images(
                    images, ocr_mode, context, engine
                )
                
                # Format with slide information
                image_texts = []
                for i, text in enumerate(extracted_texts):
                    if text and i < len(slide_info):
                        slide_num = slide_info[i]
                        image_texts.append(f"Slide {slide_num}: {text}")
                    elif text:
                        image_texts.append(text)
            
            logger.debug(f"Smart OCR processed {len(image_texts)} images from PPTX")
            return "\n".join(image_texts)
            
        except Exception as e:
            logger.error(f"Smart image extraction from PPTX failed: {e}")
            # Fallback to original method
            return await self._extract_text_from_images(file_path)

    async def _extract_presentation_images(self, file_path: str) -> tuple[List[Image.Image], List[int]]:
        """Extract images from PowerPoint presentation."""
        try:
            from spire.presentation import Presentation
            from spire.presentation.common import ShapeType, PictureShape, SlidePicture
            
            def sync_extract_images():
                presentation = Presentation()
                presentation.LoadFromFile(file_path)
                
                images = []
                slide_info = []
                
                image_dir = Path(self.temp_dir.name) / "images"
                image_dir.mkdir(exist_ok=True)
                
                for slide_idx, slide in enumerate(presentation.Slides, 1):
                    for shape_idx, shape in enumerate(slide.Shapes):
                        if isinstance(shape, SlidePicture) or isinstance(shape, PictureShape):
                            try:
                                img_path = image_dir / f"slide_{slide_idx}_pic_{shape_idx}.png"
                                if isinstance(shape, SlidePicture):
                                    img = shape.PictureFill.Picture.EmbedImage.Image
                                else:  # PictureShape
                                    img = shape.EmbedImage.Image
                                    
                                img.Save(str(img_path))
                                
                                # Load as PIL Image
                                pil_img = Image.open(img_path)
                                images.append(pil_img)
                                slide_info.append(slide_idx)
                                
                            except Exception as e:
                                logger.warning(f"Failed to process image in slide {slide_idx}: {e}")
                
                presentation.Dispose()
                return images, slide_info
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, sync_extract_images)
            
        except Exception as e:
            logger.error(f"Error extracting images from {file_path}: {e}")
            return [], []

    async def _extract_with_trocr_pptx(self, images: List[Image.Image], slide_info: List[int]) -> List[str]:
        """Extract text using TrOCR for handwritten content in presentations."""
        image_texts = []
        
        if not self.model_manager:
            return image_texts
            
        try:
            processor = self.model_manager.get_trocr_processor()
            model = self.model_manager.get_trocr_model()
            
            for idx, (image, slide_num) in enumerate(zip(images, slide_info)):
                try:
                    with torch.no_grad():
                        inputs = processor(images=image, return_tensors="pt").to(model.device)
                        generated_ids = model.generate(inputs.pixel_values)
                        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        if extracted_text.strip():
                            image_texts.append(f"Slide {slide_num} [TrOCR]: {extracted_text}")
                            
                except Exception as e:
                    logger.warning(f"TrOCR failed for image {idx + 1} on slide {slide_num}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"TrOCR processing failed: {e}")
            
        return image_texts

    def _determine_image_context_pptx(self, file_path: str, images: List[Image.Image]) -> str:
        """Determine the context of images in PowerPoint for optimal OCR."""
        try:
            # Since PowerPoint often contains diagrams and screenshots
            # we can make educated guesses based on common presentation patterns
            
            # If there are many images, likely diagrams/charts
            if len(images) > 5:
                return "diagram"
            
            # Check image properties for context clues
            for img in images[:3]:  # Check first few images
                width, height = img.size
                
                # Very wide images might be screenshots
                if width > height * 2:
                    return "screenshot"
                
                # Square-ish images might be diagrams
                if abs(width - height) < min(width, height) * 0.3:
                    return "diagram"
            
            # Default context for presentations
            return "diagram"
            
        except Exception:
            return "general"

    def _determine_smart_ocr_strategy(self, text_length: int, slide_count: int, file_type: str) -> tuple[bool, str]:
        """Determine if smart OCR should be used for PowerPoint."""
        if not self.use_smart_ocr or not self.smart_ocr_available:
            # Use original logic - always process images in presentations
            return True, "full_legacy"
            
        # Smart logic for PowerPoint files
        text_density = text_length / slide_count if slide_count > 0 else 0
        
        if text_length < 500:
            return True, "full"  # Likely image-heavy presentation
        elif text_length < 2000 or text_density < 200:
            return True, "selective"  # Mixed content presentation
        elif text_length < 5000:
            return True, "minimal"  # Text-rich but may have important diagrams
        else:
            return False, "skip"  # Very text-rich presentation

    def _choose_ocr_engine_for_context(self, context: str) -> str:
        """Choose OCR engine based on PowerPoint image context."""
        engine_preferences = {
            "handwritten": "auto",         # Let TrOCR handle this
            "diagram": "tesseract",        # Tesseract good for diagram text
            "screenshot": "tesseract",     # Screenshots usually have clean text
            "korean_text": "easyocr",      # EasyOCR better for Korean
            "general": "auto"
        }
        
        return engine_preferences.get(context, "auto")

    async def _extract_text_from_images(self, file_path: str) -> str:
        """Original method for extracting text from PowerPoint images."""
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
        """Extract tables from a PPT/PPTX file with smart processing."""
        try:
            from spire.presentation import Presentation
            from spire.presentation.common import ITable
            
            def sync_extract_tables():
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
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, sync_extract_tables)
            
        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")
            return []

    def get_handler_status(self) -> dict:
        """Get current handler status for debugging."""
        status = {
            "handler": "Enhanced PPTX Handler",
            "smart_ocr_enabled": self.use_smart_ocr,
            "smart_ocr_available": getattr(self, 'smart_ocr_available', False),
            "spire_presentation_available": True,  # Assume available if imported
            "ml_models_loaded": {
                "trocr": self.model_manager is not None and hasattr(self.model_manager, 'get_trocr_model'),
            }
        }
        
        if self.use_smart_ocr and hasattr(self, 'ocr_service'):
            status["ocr_service_status"] = self.ocr_service.get_config_status()
        
        return status

    def __del__(self):
        """Clean up temporary directory when the handler is destroyed."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()