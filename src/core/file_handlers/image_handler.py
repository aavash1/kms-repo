# image_handler.py

import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np
import torch
try:
    import easyocr
except ImportError:
    easyocr = None  # If EasyOCR is not installed, we'll handle that in code.

class ImageHandler:
    def __init__(self, languages=['ko', 'en']):
        """
        Initialize the ImageHandler with specified languages for OCR.
        languages: list of language codes (EasyOCR codes or Tesseract codes) 
                   e.g., ['ko','en'] for Korean and English.
        """
        self.languages = languages
        # Prepare language strings for each OCR engine
        # For Tesseract, join with '+', e.g. 'kor+eng'
        self.tesseract_lang = "+".join(['eng' if lang == 'en' else 'kor' if lang == 'ko' else lang 
                                        for lang in self.languages])
        # For EasyOCR, just use the list as is (ensuring easyocr library uses same codes)
        self.easyocr_langs = self.languages

        # Initialize EasyOCR reader if possible (lazily or here)
        self.easyocr_reader = None
        if easyocr is not None:
            try:
                # Use GPU if available; for simplicity, use CPU by setting gpu=False
                gpu_available = torch.cuda.is_available()
                self.easyocr_reader = easyocr.Reader(self.easyocr_langs, gpu=gpu_available)
            except Exception as e:
                # If initialization fails (e.g., models not downloaded), handle accordingly
                print(f"[Warning] EasyOCR initialization failed: {e}")
                self.easyocr_reader = None

    def _ocr_with_tesseract(self, image):
        """
        Internal method to perform OCR using Tesseract (pytesseract).
        Returns a list of dicts with 'text', 'bounding_box', and 'metadata' for each detected text element.
        """
        results = []
        # Use pytesseract to get detailed OCR results including bounding boxes
        data = pytesseract.image_to_data(image, lang=self.tesseract_lang, output_type=Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            text = data['text'][i]
            conf_val = data['conf'][i]
            if isinstance(conf_val, str) and conf_val.isdigit():
                conf = int(conf_val)
            elif isinstance(conf_val, (int, float)):
                conf = int(conf_val)
            else:
                conf = -1
            if text and conf != -1:  # Filter out empty text and non-word entries
                x = int(data['left'][i])
                y = int(data['top'][i])
                w = int(data['width'][i])
                h = int(data['height'][i])
                segment = {
                    "text": text,
                    "bounding_box": (x, y, w, h),
                    "metadata": {
                        "engine": "Tesseract",
                        "confidence": conf,
                        "languages": self.tesseract_lang
                    }
                }
                results.append(segment)
        return results

    def _ocr_with_easyocr(self, image):
        """
        Internal method to perform OCR using EasyOCR.
        Returns a list of dicts with 'text', 'bounding_box', and 'metadata' for each detected text element.
        """
        results = []
        if self.easyocr_reader is None:
            # EasyOCR is not available/initialized
            raise RuntimeError("EasyOCR engine is not initialized or not available.")
        # Perform OCR using EasyOCR
        ocr_results = self.easyocr_reader.readtext(image)  # detail=1 by default returns (bbox, text, confidence)
        for bbox, text, conf in ocr_results:
            # Convert bbox (which is a list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) 
            # into a (x, y, w, h) tuple for consistency.
            # We can compute min x, min y, width, height from the bbox points.
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            segment = {
                "text": text,
                "bounding_box": (int(x), int(y), int(w), int(h)),
                "metadata": {
                    "engine": "EasyOCR",
                    "confidence": float(conf),
                    "languages": self.easyocr_langs
                }
            }
            results.append(segment)
        return results

    def choose_engine(self, engine_preference="auto"):
        """
        Decide which OCR engine to use based on preference or availability.
        - If engine_preference is 'tesseract' or 'easyocr', choose that explicitly.
        - If 'auto', select Tesseract if available, otherwise EasyOCR.
        """
        engine_preference = engine_preference.lower()
        if engine_preference in ["tesseract", "easyocr"]:
            return engine_preference
        # Automatic selection: try tesseract first
        try:
            # Check if Tesseract is available by getting its version
            _ = pytesseract.get_tesseract_version()
            return "tesseract"
        except Exception:
            if self.easyocr_reader is not None:
                return "easyocr"
            else:
                # As a fallback, if neither is available (should not happen if properly installed)
                raise RuntimeError("No OCR engine available. Please install Tesseract or EasyOCR.")
    
    def process_image(self, image_path, engine="auto"):
        """
        Process a single image with OCR and return extracted text with bounding boxes and metadata.
        image_path: path to the image file (or could be a pre-loaded image array).
        engine: 'tesseract', 'easyocr', or 'auto' for dynamic selection.
        """
        # Read image using OpenCV
        if not isinstance(image_path, str):
            image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        
        # For Tesseract via PIL, convert if needed (optional step)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pil_image = Image.fromarray(image_rgb)
        # But pytesseract can directly accept numpy arrays (OpenCV images).
        preprocessed = self._preprocess_image(image)
        # Choose engine
        selected_engine = self.choose_engine(engine)
        if selected_engine == "tesseract":
            results = self._ocr_with_tesseract(preprocessed)
        else:  # selected_engine == "easyocr"
            results = self._ocr_with_easyocr(preprocessed)
        return results

    def _preprocess_image(self, image):
        """
        Convert to grayscale, then apply Otsu threshold to binarize.
        Returns the thresholded (binary) image for improved OCR.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's threshold to remove background noise
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh


    @staticmethod
    def reconstruct_aligned_text(results, y_threshold=10):
        """
        Group OCR segments (with bounding_box = (x, y, w, h)) into lines based on similar y values,
        then sort each line's segments by x and join their text.
        y_threshold defines how close in vertical position segments must be to be considered in the same line.
        """
        lines = []
        for seg in results:
            x, y, w, h = seg['bounding_box']
            # Try to find an existing line (group) with a similar y value
            found_line = False
            for line in lines:
                if abs(line['y'] - y) < y_threshold:
                    line['segments'].append(seg)
                    found_line = True
                    break
            if not found_line:
                lines.append({'y': y, 'segments': [seg]})
        
        # Sort the lines based on their y coordinate
        lines.sort(key=lambda l: l['y'])
        
        reconstructed_lines = []
        for line in lines:
            # Sort segments in each line based on the x coordinate
            segments = sorted(line['segments'], key=lambda s: s['bounding_box'][0])
            # Join the text for each segment; you can adjust spacing here as needed.
            line_text = " ".join(seg['text'] for seg in segments)
            reconstructed_lines.append(line_text)
        
        return "\n".join(reconstructed_lines)

    def extract_text(self, image_path, engine="tesseract"):
        """
        Wrapper method for backward compatibility.
        Processes the image and returns the reconstructed aligned text.
        """
        ocr_results = self.process_image(image_path, engine=engine)
        return self.reconstruct_aligned_text(ocr_results)

    # Add this new method to handle file content in memory
    def extract_text_from_memory(self, file_content: bytes) -> str:
        """
        Extract text from image content in memory using OCR.

        Args:
            file_content: Raw bytes of the image content.

        Returns:
            str: Extracted text, or empty string if extraction fails.
        """
        try:
            # Convert bytes to a numpy array for OpenCV
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image from memory")

            # Preprocess the image
            preprocessed = self._preprocess_image(image)

            # Choose engine (default to tesseract for consistency with extract_text)
            selected_engine = self.choose_engine("tesseract")
            if selected_engine == "tesseract":
                results = self._ocr_with_tesseract(preprocessed)
            else:  # selected_engine == "easyocr"
                results = self._ocr_with_easyocr(preprocessed)

            # Reconstruct aligned text
            text = self.reconstruct_aligned_text(results)
            return text if text else ""
        except Exception as e:
            print(f"[Error] Failed to extract text from image in memory: {e}")
            return ""
    
    def get_status_codes(self):
        """
        For image files, there are no status codes. 
        Return an empty list for compatibility.
        """
        return []


