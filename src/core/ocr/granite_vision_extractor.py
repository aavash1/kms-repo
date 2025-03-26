# src/core/ocr/granite_vision_extractor.py
import ollama
import tempfile
import os
from PIL import Image
import asyncio
import logging
import traceback
logger = logging.getLogger(__name__)

class GraniteVisionExtractor:
    """Simple wrapper for using ollama vision models to extract text from images."""
    
    def __init__(self, model_name="llama3.2-vision"):
        """Initialize with model name."""
        self.model_name = model_name
        logger.info(f"Initialized GraniteVisionExtractor with model: {model_name}")
    
    def extract_text_from_file(self, image_path):
        """Extract text from an image file."""
        try:
            # Using improved prompt for better OCR results
            prompt = (
                "You are an advanced OCR system. Extract ALL visible text from this image. "
                "Include everything - text in tables, buttons, headers, data values, etc. "
                "Return ONLY the extracted text without explanations or descriptions. "
                "Preserve line breaks and formatting. If no text is visible, respond with 'No text detected'."
            )
            
            # Ensure the image path exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return ""
                
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            
            # Extract content from response
            if response and 'message' in response and 'content' in response['message']:
                extracted_text = self._clean_response(response['message']['content'])
                logger.debug(f"Extracted text (first 100 chars): {extracted_text[:100]}")
                return extracted_text
            
            logger.warning("No valid response from vision model")
            return ""
            
        except Exception as e:
            logger.error(f"Error in extract_text_from_file: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    async def extract_text_from_bytes(self, image_bytes, timeout_seconds=30):
        """Extract text from image bytes with timeout."""
        temp_file_path = None
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(image_bytes)
                temp_file_path = temp.name
            
            # Run in a separate thread to avoid blocking event loop
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.extract_text_from_file, temp_file_path),
                timeout=timeout_seconds
            )
            
            return result
                
        except asyncio.TimeoutError:
            logger.error(f"Vision extraction timed out after {timeout_seconds} seconds")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from bytes: {e}")
            logger.error(traceback.format_exc())
            return ""
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {e}")
    
    def _clean_response(self, text):
        """Clean the response to remove any explanatory text."""
        if not text:
            return ""
        
        # Remove common prefixes that might appear
        prefixes_to_remove = [
            "Here's the text from the image:",
            "The text in the image reads:",
            "The image contains the following text:",
            "Text extracted from the image:",
            "The text visible in the image is:",
            "I can see the following text in the image:",
            "Text content:",
            "Extracted text:",
            "The text says:",
            "Here is the text in the image:",
            "The image shows:",
            "Here is the exact text from the image:"
        ]
        
        # Remove prefixes
        cleaned_text = text
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].lstrip()
        
        # Remove markdown code blocks if present
        lines = cleaned_text.split('\n')
        filtered_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if not in_code_block:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()