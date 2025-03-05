# src/core/ocr/tesseract_wrapper.py
import pytesseract
from ..config import load_ocr_config
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TesseractOCR:
    def __init__(self):
        config = load_ocr_config()['tesseract']
        pytesseract.pytesseract.tesseract_cmd = config['path']
        self.lang = config['lang']
        self.psm = config['psm']
        self.oem = config['oem']
        
    def extract_text(self, image,  config=''):
        # Handles both PIL.Image objects and file paths
        # return pytesseract.image_to_string(
        #     image,
        #     lang=self.lang,
        #     config=f'--psm {self.psm} --oem {self.oem}'
        # )
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            # Combine base config with any additional config
            base_config = f'--psm {self.psm} --oem {self.oem}'
            full_config = f'{base_config} {config}'.strip()
            
            # Extract text
            return pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=full_config
            )
        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {e}")
            return ""

    def get_ocr_data(self, image, config=''):
        # base_config = f'--psm {self.psm} --oem {self.oem}'
        # full_config = f'{base_config} {config}'.strip()
        # return pytesseract.image_to_data(                                                                        
        #     image, 
        #     lang=self.lang,
        #     config=full_config,
        #     output_type=pytesseract.Output.DICT
        # )
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            base_config = f'--psm {self.psm} --oem {self.oem}'
            full_config = f'{base_config} {config}'.strip()
            
            return pytesseract.image_to_data(
                image, 
                lang=self.lang,
                config=full_config,
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"Tesseract OCR data extraction failed: {e}")
            return {}

# #class TesseractOCR:
#     def __init__(self):
#         config = load_ocr_config()['tesseract']
#         pytesseract.pytesseract.tesseract_cmd = config['path']
#         self.lang = config['lang']
#         self.psm = config['psm']
#         self.oem = config['oem']
        
#     def extract_text(self, image):
#         if isinstance(image, str):
#             return pytesseract.image_to_string(
#                 image,
#                 lang=self.config['lang'],
#                 config=f'--psm {self.config["psm"]} --oem {self.config["oem"]}'
#             )
#         elif isinstance(image, Image.Image):
#             return pytesseract.image_to_string(
#                 image,
#                 lang=self.lang,
#                 config=f'--psm {self.psm} --oem {self.oem}'
#             )
#         else:
#             raise ValueError("Unsupported image input.")