# src/core/vision/granite_vision_extractor.py

import ollama
import tempfile
import os
from PIL import Image
import io

class GraniteVisionExtractor:
    """Simple wrapper for using ollama vision models to extract text from images."""
    
    def __init__(self, model_name="llama3.2-vision"):
        """Initialize with model name."""
        self.model_name = model_name
    
    def extract_text_from_file(self, image_path):
        """Extract text from an image file."""
        try:
            # Using improved prompt that's more direct and prevents hallucinations
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': 'Extract exactly the text visible in this image. Do not include any commentary, explanation, or summary—return only the text as it appears.',
                    'images': [image_path]
                }]
            )
            
            # Extract content from response
            if response and 'message' in response and 'content' in response['message']:
                # Clean the response to remove any explanatory text
                extracted_text = self._clean_response(response['message']['content'])
                return extracted_text
            return ""
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def extract_text_from_bytes(self, image_bytes):
        """Extract text from image bytes."""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(image_bytes)
                temp_path = temp.name
            
            # Extract text from file
            result = self.extract_text_from_file(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
            return result
            
        except Exception as e:
            print(f"Error extracting text from bytes: {e}")
            return ""
    
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