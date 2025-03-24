import fitz  # PyMuPDF
import pdfplumber
import cv2
import numpy as np
import logging
import easyocr
import layoutparser as lp
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from src.core.config import load_ocr_config
from src.core.ocr.tesseract_wrapper import TesseractOCR
from .base_handler import FileHandler
import torch
import warnings
import os
import tempfile
warnings.filterwarnings("ignore", category=FutureWarning)
import regex as re
from src.core.utils.post_processing import (
    preprocess_image, 
    clean_extracted_text,
    fix_hyphenation,
    preserve_line_breaks
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

class PDFHandler(FileHandler):
    def __init__(self, model_manager):
        self.poppler_path = load_ocr_config()['poppler']['path']
        self.ocr = TesseractOCR()
        self.reader = easyocr.Reader(['en', 'ko'], gpu=True)
        self.tesseract_layout_config = '--psm 11'
        self.status_codes = []
        
        # Use ModelManager to access models and device
        self.model_manager = model_manager
        self.device = model_manager.get_device()
        self.handwritten_processor = model_manager.get_trocr_processor()
        self.handwritten_model = model_manager.get_trocr_model()
        self.bert_tokenizer = model_manager.get_klue_tokenizer()
        self.bert_model = model_manager.get_klue_bert()

    def get_status_codes(self):
        """Return the list of status codes found in the last processed document."""
        return self.status_codes
    
    def process_text_with_bert(self, text):
        """
        Process extracted text through KLUE BERT to get embeddings or analyze text.
        """
        try:
            if not isinstance(text, str):
                logger.error(f"Expected string input for BERT, got {type(text)}: {text}")
                return None
            if not text.strip():
                logger.debug("Empty text passed to BERT, skipping")
                return None
            
            with torch.no_grad():
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state
                return embeddings
        except Exception as e:
            logger.error(f"BERT processing failed: {str(e)}", exc_info=True)
            return None
    
    def extract_text(self, file_path):
        combined_text = []
        all_status_codes = set()

        if not isinstance(file_path, str):
            logger.error(f"Invalid file_path type: {type(file_path)}. Expected string.")
            return ""

        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return ""

        # Try pdfplumber first for better text extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    logger.debug(f"pdfplumber extracted from page {page_num}: {text!r}")
                    if text and isinstance(text, str) and text.strip():
                        formatted_text = self._format_text_section(page_num, "Native Text (pdfplumber)", text)
                        combined_text.append(formatted_text)
                        page_status_codes = extract_status_codes(text)
                        all_status_codes.update(page_status_codes)
                        embedding = self.process_text_with_bert(text)
                        if embedding is not None:
                            logger.debug(f"BERT embedding generated for page {page_num}")
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed for {file_path}: {str(e)}", exc_info=True)

        # Fall back to fitz if pdfplumber fails
        if not combined_text:
            try:
                with fitz.open(file_path) as doc:
                    for page_num, page in enumerate(doc, 1):
                        text = page.get_text("text")
                        logger.debug(f"fitz extracted from page {page_num}: {text!r}")
                        if text and isinstance(text, str) and text.strip():
                            formatted_text = self._format_text_section(page_num, "Native Text (fitz)", text)
                            combined_text.append(formatted_text)
                            page_status_codes = extract_status_codes(text)
                            all_status_codes.update(page_status_codes)
                            embedding = self.process_text_with_bert(text)
                            if embedding is not None:
                                logger.debug(f"BERT embedding generated for page {page_num}")
            except Exception as e:
                logger.error(f"fitz text extraction failed for {file_path}: {e}", exc_info=True)

        # Fall back to image-based extraction if no text was extracted
        if not combined_text:
            try:
                images = convert_from_path(file_path, poppler_path=self.poppler_path)
                for idx, img in enumerate(images, start=1):
                    printed_text = self._layout_aware_ocr(img)
                    logger.debug(f"OCR printed text from page {idx}: {printed_text!r}")
                    if printed_text.strip():
                        formatted_printed = self._format_text_section(idx, "Printed OCR Text", printed_text)
                        combined_text.append(formatted_printed)
                        ocr_status_codes = extract_status_codes(printed_text)
                        all_status_codes.update(ocr_status_codes)

                    handwritten_text = self._extract_handwritten_text(img)
                    logger.debug(f"OCR handwritten text from page {idx}: {handwritten_text!r}")
                    if handwritten_text.strip():
                        formatted_handwritten = self._format_text_section(idx, "Handwritten OCR Text", handwritten_text)
                        combined_text.append(formatted_handwritten)
                        hw_status_codes = extract_status_codes(handwritten_text)
                        all_status_codes.update(hw_status_codes)
            except Exception as e:
                logger.error(f"Image processing failed for {file_path}: {e}", exc_info=True)

        self.status_codes = list(all_status_codes)

        if not combined_text:
            logger.warning(f"No text was successfully extracted from the document: {file_path}")
            return ""

        final_text = "\n".join(combined_text)
        logger.debug(f"Final combined text: {final_text!r}")
        return final_text
 
    def _extract_handwritten_text(self, image):
        try:
            # Preprocess image for better results
            enhanced_image = self._preprocess_handwritten_image(image)
            # Process image with CUDA optimization
            with torch.no_grad():  # Disable gradient calculation for inference
                inputs = self.handwritten_processor(
                    enhanced_image, 
                    return_tensors="pt",
                    image_scale=True,
                    do_align=True
                ).to(self.device)
                
                generated_ids = self.handwritten_model.generate(
                    inputs.pixel_values,
                    max_length=256,  # Increased for longer text
                    num_beams=5,     # Increased beam search
                    early_stopping=True,
                    temperature=0.5,  # Added temperature for better sampling
                    top_k=50,        # Added top_k sampling
                    top_p=0.95,      # Added nucleus sampling
                    repetition_penalty=1.2,  # Prevent repetitions
                    length_penalty=1.0,
                    no_repeat_ngram_size=3
                )
                
                text = self.handwritten_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]

                # Additional post-processing for handwritten text
                cleaned_text = self._post_process_handwritten(text)
            
            # Clear CUDA cache if needed
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Handwritten text extraction failed: {str(e)}", exc_info=True)
            return ""

    def _enhance_handwritten_image(self, image):
        """Enhanced image preprocessing specifically for handwritten text."""
        # Convert to grayscale first
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast more aggressively for handwriting
        image = ImageEnhance.Contrast(image).enhance(2.5)
        
        # Increase sharpness
        image = ImageEnhance.Sharpness(image).enhance(2.5)
        
        # Additional preprocessing steps
        img_array = np.array(image)
        
        # Adaptive thresholding
        img_array = cv2.adaptiveThreshold(
            img_array, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Noise removal
        kernel = np.ones((1,1), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(img_array)

    def _post_process_handwritten(self, text):
        """Post-process extracted handwritten text."""
        if not text:
            return text
        
        # Remove repeated characters (common in handwriting recognition)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            
        # Clean up common OCR mistakes
        replacements = {
            r'["""]': '"',            # Normalize quotes
            r'[\''']': "'",           # Normalize apostrophes
            r'1l|Il': '11',          # Fix one-one confusion
            r'(?<!\d)1(?!\d)': 'I',  # Single 1 to I when not part of number
            r'(?<=\d)O|o(?=\d)': '0' # Fix O/0 confusion in numbers
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove extra whitespace while preserving Korean characters
        text = re.sub(r'(?<![가-힣])\s+(?![가-힣])', ' ', text)
        
        return text.strip()
    
    def _enhance_image(self, image):
        # Enhance contrast and remove noise for better OCR
        image = ImageEnhance.Contrast(image).enhance(2.0)  # Increase contrast
        image = ImageEnhance.Sharpness(image).enhance(2.0)  # Sharpen image
        return image

    def extract_tables(self, file_path):
        tables = []
        
        # Native table extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    tables.extend(page.extract_tables())
        except Exception as e:
            logger.error(f"Native table extraction failed: {e}")

        # Image-based table detection
        try:
            images = convert_from_path(file_path, poppler_path=self.poppler_path)
            for img in images:
                tables.extend(self._detect_image_tables(img))
        except Exception as e:
            logger.error(f"Image table detection failed: {e}")

        return tables

    def _layout_aware_ocr(self, image):
        try:
            enhanced_image = self._enhance_image(image)
            np_image = np.array(enhanced_image)
        
            # Use PSM from config
            config = load_ocr_config()['tesseract']
            custom_config = f'--psm {config["psm"]}'
        
            ocr_data = self.ocr.get_ocr_data(np_image, config=custom_config)
            combined_text = self._combine_blocks(ocr_data)
        
            return combined_text
        except Exception as e:
            logger.error(f"Layout-aware OCR failed: {e}")
            return ""

    def _detect_image_tables(self, image):
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Detect lines
            horizontal = cv2.erode(cv2.getStructuringElement(cv2.MORPH_RECT, (40,1)), thresh)
            vertical = cv2.erode(cv2.getStructuringElement(cv2.MORPH_RECT, (1,40)), thresh)
            table_mask = horizontal + vertical

            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return self._process_table_contours(contours, image)
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}", exc_info=True)
            return ""

    def _process_table_contours(self, contours, image):
        tables = []
        np_image = np.array(image)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            table_region = np_image[y:y+h, x:x+w]
            tables.append(self.reader.readtext(table_region, detail=0))
        
        return tables

    def _ocr_images(self, image):
        preprocessed = preprocess_image(image)
        raw_text = self.ocr.extract_text(preprocessed)
        return clean_extracted_text(raw_text)

    def _combine_blocks(self, ocr_data):
        combined_text = []
        current_block = -1
        current_line = ""
        last_top = -1
        min_left = float('inf')
    
        # First pass: Find minimum left position
        for i in range(len(ocr_data['left'])):
            if ocr_data['conf'][i] > 0:
                min_left = min(min_left, ocr_data['left'][i])
    
        # Second pass: Process text blocks
        for i in range(len(ocr_data['block_num'])):
            block_num = ocr_data['block_num'][i]
            word_text = ocr_data['text'][i].strip()
            top = ocr_data['top'][i]
            left = ocr_data['left'][i]
        
            # Skip empty or low-confidence text
            if not word_text or ocr_data['conf'][i] <= 0:
                continue
        
            # Handle new block or line break
            is_new_line = abs(top - last_top) > ocr_data['height'][i] * 0.5
            if block_num != current_block or is_new_line:
                if current_line:
                    combined_text.append(current_line.rstrip())
            
                # Calculate relative indentation
                indent_spaces = max(0, int((left - min_left) / 10))
                current_line = " " * indent_spaces + word_text
                current_block = block_num
            else:
                current_line += f" {word_text}"
        
            last_top = top
    
        # Add final line
        if current_line:
            combined_text.append(current_line.rstrip())
    
        return clean_extracted_text("\n".join(combined_text))

    def __del__(self):
        # Cleanup is handled by ModelManager
        pass

    def _format_text_section(self, page_num, section_type, text):
        if not text.strip():
            return ""

        # For handwritten text that's just "0 0", skip it entirely
        if section_type == "Handwritten OCR Text" and text.strip() in ["0 0", "0", ""]:
            return ""

        header = f"\nPage {page_num}"
        
        lines = []
        formatted_lines = []
        current_indent = 0
        
        in_code_block = False
        code_indent = 0
        
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:
                # Handle indentation for lists and commands
                if cleaned_line[0].isdigit() and '. ' in cleaned_line[:4]:
                    current_indent = 2
                elif cleaned_line.startswith(('$', '#', '>', '-')):
                    current_indent = 4
                elif cleaned_line.startswith('For '):
                    current_indent = 0
                    lines.append('')  # Add blank line before new sections
                    
                lines.append(' ' * current_indent + cleaned_line)
                
                # Reset indent unless it's a continuation
                if not cleaned_line.endswith((':', '-', '>', ',')):
                    current_indent = 0
        
        formatted_text = '\n'.join(lines)
        if formatted_text.strip():
            return header + '\n' + formatted_text + '\n'
        return ""

    def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from a PDF file in memory.

        Parameters:
            file_content (bytes): The content of the PDF file as bytes.

        Returns:
            str: Extracted text from the PDF.
        """
        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Call extract_text, which returns a single string
            text = self.extract_text(temp_file_path)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from memory buffer: {e}", exc_info=True)
            return ""
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    
    def _preprocess_handwritten_image(self, image):
        """
        Enhanced preprocessing specifically for handwritten text recognition.
        """
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply adaptive thresholding
        img_array = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Denoise the image
        img_array = cv2.fastNlMeansDenoising(img_array)
        
        # Increase contrast
        img_array = cv2.normalize(img_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Apply dilation to make handwriting more prominent
        kernel = np.ones((2,2), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        return Image.fromarray(img_array)