# src/core/file_handlers/image_handler.py
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
import io
import asyncio
import concurrent.futures
from typing import Optional, List
import time

try:
    import easyocr
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)

class ImageHandler:
    def __init__(self, model_manager=None, languages=['ko', 'en']):
        self.languages = languages
        self.tesseract_lang = "+".join(['eng' if lang == 'en' else 'kor' if lang == 'ko' else lang 
                                        for lang in self.languages])
        self.tesseract_config = f"-l {self.tesseract_lang} --psm 6"
        self.easyocr_langs = self.languages

        self.easyocr_reader = None
        if easyocr is not None:
            try:
                gpu_available = model_manager.get_device() == 'cuda' if model_manager else torch.cuda.is_available()
                self.easyocr_reader = easyocr.Reader(self.easyocr_langs, gpu=gpu_available)
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.easyocr_reader = None

        self.model_manager = model_manager
        if model_manager:
            self.device = model_manager.get_device()
            self.trocr_processor = model_manager.get_trocr_processor()
            self.trocr_model = model_manager.get_trocr_model()
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                r"C:\AI_Models\local_cache\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee",
                local_files_only=True,
                use_fast=True
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                r"C:\AI_Models\local_cache\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee",
                local_files_only=True
            ).to(self.device)
            self.trocr_model.eval()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image to improve OCR accuracy."""
        # Convert to grayscale
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        # Reduce noise
        denoised = cv2.fastNlMeansDenoising(gray)
        # Thresholding to binarize the image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # Make these class methods rather than local functions
    def _ocr_with_tesseract(self, image: np.ndarray) -> str:
        """Run Tesseract OCR."""
        try:
            # Convert to PIL Image for pytesseract
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            result = pytesseract.image_to_string(
                pil_image, 
                config=self.tesseract_config,
                lang=self.tesseract_lang
            )
            return result
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return ""

    def _ocr_with_easyocr(self, image: np.ndarray) -> str:
        """Run EasyOCR."""
        if self.easyocr_reader is None:
            return ""
            
        try:
            ocr_results = self.easyocr_reader.readtext(image, detail=0)
            return " ".join(ocr_results)
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return ""

    def _ocr_with_trocr(self, image: np.ndarray) -> str:
        """Run TrOCR."""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            with torch.no_grad():
                inputs = self.trocr_processor(images=pil_image, return_tensors="pt").to(self.trocr_model.device)
                generated_ids = self.trocr_model.generate(inputs.pixel_values)
                text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return text
        except Exception as e:
            logger.error(f"TrOCR failed: {str(e)}")
            return ""

    def choose_engine(self, engine_preference="auto"):
        """Choose the OCR engine based on preference or availability."""
        engine_preference = engine_preference.lower()
        if engine_preference in ["tesseract", "easyocr", "trocr"]:
            if engine_preference == "easyocr" and self.easyocr_reader is None:
                logger.warning("EasyOCR not available, falling back to tesseract")
                return "tesseract"
            if engine_preference == "trocr" and self.trocr_model is None:
                logger.warning("TrOCR not available, falling back to tesseract")
                return "tesseract"
            return engine_preference

        try:
            _ = pytesseract.get_tesseract_version()
            return "tesseract"
        except Exception:
            if self.easyocr_reader is not None:
                return "easyocr"
            elif self.trocr_model is not None:
                return "trocr"
            raise RuntimeError("No OCR engine available.")

    def _enhance_image_for_ocr(self, image):
        """Apply enhancements to improve OCR quality."""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Increase contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            img = enhancer.enhance(1.5)
            # Sharpen
            img = img.filter(ImageFilter.SHARPEN)
            return img
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _process_blurry_image(self, img):
        """Apply specialized processing for blurry images."""
        try:
            # Make sure we're working with numpy array
            if isinstance(img, Image.Image):
                img = np.array(img)
                
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Apply deblurring techniques
            # 1. Unsharp masking
            gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
            unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            
            # 2. Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 3. Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            return opening
        except Exception as e:
            logger.error(f"Error processing blurry image: {e}")
            return img

    async def process_image(self, image_path: str, engine: str = "auto") -> list:
        """Process an image file and extract text segments with bounding boxes."""
        if not isinstance(image_path, str):
            image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        preprocessed = self.preprocess_image(image)
        selected_engine = self.choose_engine(engine)

        # Use ThreadPoolExecutor instead of directly running in event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if selected_engine == "tesseract":
                future = executor.submit(self._ocr_with_tesseract, preprocessed)
            elif selected_engine == "easyocr":
                future = executor.submit(self._ocr_with_easyocr, preprocessed)
            else:  # trocr
                future = executor.submit(self._ocr_with_trocr, preprocessed)
                
            # Convert future to awaitable
            result = await asyncio.wrap_future(future)
            
            if not result:
                # Try vision model if standard OCR fails
                try:
                    from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
                    vision_extractor = GraniteVisionExtractor()
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                    result = await vision_extractor.extract_text_from_bytes(image_bytes)
                except Exception as e:
                    logger.error(f"Vision model extraction failed: {e}")
                    
            return result if result else ""

    async def extract_text_from_memory(self, image_bytes: bytes, ocr_engine: str = "auto", timeout_seconds: int = 30) -> Optional[str]:
        """
        Extract text from image bytes using OCR with fallback to vision models for difficult images.
        
        Args:
            image_bytes: Raw bytes of the image.
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', 'trocr', or 'auto').
            timeout_seconds: Maximum time in seconds to wait for OCR to complete.
                
        Returns:
            Extracted text, or None if extraction fails.
        """
        try:
            # Convert image bytes to a numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to decode image for OCR.")
                return None

            # Pre-process image to improve OCR accuracy
            enhanced_img = self._enhance_image_for_ocr(img)
            blurry_img = self._process_blurry_image(img)
            
            # Choose OCR engine
            engine = self.choose_engine(ocr_engine)
            
            # Create a future for timeout handling
            result = None
            
            # Use ThreadPoolExecutor instead of multiprocessing
            with concurrent.futures.ThreadPoolExecutor() as executor:
                try:
                    if engine == "tesseract":
                        future = executor.submit(self._ocr_with_tesseract, enhanced_img)
                    elif engine == "easyocr" and self.easyocr_reader:
                        future = executor.submit(self._ocr_with_easyocr, enhanced_img)
                    elif engine == "trocr" and self.trocr_model:
                        future = executor.submit(self._ocr_with_trocr, enhanced_img)
                    else:
                        future = executor.submit(self._ocr_with_tesseract, enhanced_img)
                    
                    # Wait for result with timeout
                    result = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"OCR timed out after {timeout_seconds} seconds")
                    # Try with blurry image processing if standard OCR times out
                    try:
                        future = executor.submit(self._ocr_with_tesseract, blurry_img)
                        result = future.result(timeout=timeout_seconds)
                    except Exception:
                        result = None
                except Exception as e:
                    logger.error(f"OCR processing error: {e}")
                    result = None
            
            # If standard OCR failed, try vision model as fallback if available
            if (not result or not result.strip()) and hasattr(self, 'model_manager'):
                try:
                    # Try to get a GraniteVisionExtractor instance
                    from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
                    vision_extractor = GraniteVisionExtractor()
                    logger.info("Standard OCR failed, falling back to vision model")
                    text = await vision_extractor.extract_text_from_bytes(image_bytes, timeout_seconds)
                    if text and text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"Vision model fallback failed: {e}")
            
            return result.strip() if result else None

        except Exception as e:
            logger.error(f"Error extracting text from image: {e}", exc_info=True)
            return None

    async def extract_text(self, image_path: str, engine: str = "auto", timeout_seconds: int = 30) -> str:
        """Extract text from an image file with a timeout."""
        # Read file into memory and use extract_text_from_memory
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return await self.extract_text_from_memory(image_bytes, engine, timeout_seconds) or ""
        except Exception as e:
            logger.error(f"Error in extract_text: {e}")
            return ""

    @staticmethod
    def reconstruct_aligned_text(results: list, y_threshold: int = 10) -> str:
        """Reconstruct text by aligning segments based on their y-coordinates."""
        if not results:
            return ""
        
        if isinstance(results, str):
            return results
            
        lines = []
        for seg in results:
            if isinstance(seg, dict) and 'bounding_box' in seg:
                x, y, w, h = seg['bounding_box']
                found_line = False
                for line in lines:
                    if abs(line['y'] - y) < y_threshold:
                        line['segments'].append(seg)
                        found_line = True
                        break
                if not found_line:
                    lines.append({'y': y, 'segments': [seg]})
        
        lines.sort(key=lambda l: l['y'])
        reconstructed_lines = []
        for line in lines:
            segments = sorted(line['segments'], key=lambda s: s['bounding_box'][0])
            line_text = " ".join(seg['text'] for seg in segments if seg['text'])
            if line_text.strip():
                reconstructed_lines.append(line_text)
        return "\n".join(reconstructed_lines)