import ollama
import tempfile
import os
from PIL import Image
import io
import asyncio
import pytesseract

class GraniteVisionExtractor:
    """Simple wrapper for using ollama vision models to extract text from images with pytesseract fallback."""
    
    def __init__(self, model_name="gemma3:4b", fallback_model="granite3.2-vision"):
        self.model_name = model_name if model_name else "gemma3:4b"
        self.fallback_model = fallback_model if fallback_model else "granite3.2-vision"
        self.current_model = self.model_name
        self.available_models = self._get_available_models()
        
        try:
            if self.model_name not in self.available_models:
                try:
                    ollama.pull(self.model_name)
                except Exception:
                    self.current_model = self.fallback_model
            
            if self.current_model == self.fallback_model and self.fallback_model not in self.available_models:
                try:
                    ollama.pull(self.fallback_model)
                    self.available_models = self._get_available_models()
                except Exception:
                    raise RuntimeError(f"Failed to initialize vision model: could not pull {self.fallback_model}")
        
        except Exception:
            if self.current_model != self.fallback_model:
                self.current_model = self.fallback_model
                if self.fallback_model not in self.available_models:
                    try:
                        ollama.pull(self.fallback_model)
                        self.available_models = self._get_available_models()
                    except Exception:
                        raise RuntimeError(f"Failed to initialize vision model: could not pull {self.fallback_model}")
            else:
                raise RuntimeError(f"Failed to initialize vision model: {self.model_name} and fallback {self.fallback_model} unavailable")
    
    def _get_available_models(self):
        """Get a list of available models from Ollama."""
        try:
            response = ollama.list()
            models = response.get("models", [])
            available_models = set()
            for model in models:
                if isinstance(model, dict) and "name" in model:
                    model_name = model["name"]
                    if ":" in model_name:
                        base_name = model_name.split(":")[0]
                        available_models.add(base_name)
                    available_models.add(model_name)
            return available_models
        except Exception:
            return set()
    
    async def extract_text_from_file(self, image_path, model_name=None):
        try:
            model_name = model_name or self.current_model
            
            if model_name not in self.available_models:
                try:
                    await asyncio.to_thread(ollama.pull, model_name)
                    self.available_models = self._get_available_models()
                except Exception:
                    if model_name == self.fallback_model:
                        return await self._extract_with_pytesseract(image_path)
                    elif model_name == self.model_name:
                        return await self.extract_text_from_file(image_path, self.fallback_model)
            
            prompt = (
                "You are an advanced OCR system. Extract ALL visible text from this image. "
                "Include everything - text in tables, buttons, headers, data values, etc. "
                "Return ONLY the extracted text without explanations or descriptions. "
                "Preserve line breaks and formatting. If no text is visible, respond with 'No text detected'."
            )
            
            if not os.path.exists(image_path):
                return ""
                
            try:
                timeout = 120 if "gemma3" in model_name or "llama3" in model_name else 60
                
                try:
                    from ollama import AsyncClient
                    client = AsyncClient()
                    response = await asyncio.wait_for(
                        client.chat(
                            model=model_name,
                            messages=[{
                                'role': 'user',
                                'content': prompt,
                                'images': [image_path]
                            }]
                        ),
                        timeout=timeout
                    )
                except ImportError:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            ollama.chat,
                            model=model_name,
                            messages=[{
                                'role': 'user',
                                'content': prompt,
                                'images': [image_path]
                            }]
                        ),
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                if model_name == self.fallback_model:
                    return await self._extract_with_pytesseract(image_path)
                else:
                    return await self.extract_text_from_file(image_path, self.fallback_model)
            except Exception:
                if model_name == self.fallback_model:
                    return await self._extract_with_pytesseract(image_path)
                else:
                    return await self.extract_text_from_file(image_path, self.fallback_model)
            
            if response and 'message' in response and 'content' in response['message']:
                extracted_text = self._clean_response(response['message']['content'])
                if extracted_text and extracted_text != "No text detected":
                    return extracted_text
            
            if model_name != self.fallback_model:
                return await self.extract_text_from_file(image_path, self.fallback_model)
            
            return await self._extract_with_pytesseract(image_path)
            
        except Exception:
            if model_name != self.fallback_model:
                return await self.extract_text_from_file(image_path, self.fallback_model)
            return await self._extract_with_pytesseract(image_path)
    
    async def _extract_with_pytesseract(self, image_path):
        """Helper method to extract text using pytesseract."""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng')
            cleaned_text = self._clean_response(text)
            if cleaned_text.strip():
                return cleaned_text
            return "No text detected"
        except Exception:
            return "No text detected"
    
    async def extract_text_from_bytes(self, image_bytes, timeout_seconds=120):
        temp_file_path = None
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format.lower() if image.format else 'jpg'
            suffix = f'.{image_format}'
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(image_bytes)
                temp_file_path = temp.name
            
            model_timeout = 120 if "gemma3" in self.current_model else 60
            result = await asyncio.wait_for(
                self.extract_text_from_file(temp_file_path),
                timeout=max(timeout_seconds, model_timeout)
            )
            
            return result
                
        except asyncio.TimeoutError:
            if self.current_model != self.fallback_model:
                try:
                    result = await asyncio.wait_for(
                        self.extract_text_from_file(temp_file_path, self.fallback_model),
                        timeout=60
                    )
                    return result
                except (asyncio.TimeoutError, Exception):
                    return await self._extract_with_pytesseract(temp_file_path)
            else:
                return await self._extract_with_pytesseract(temp_file_path)
        except Exception:
            return ""
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
    
    def _clean_response(self, text):
        if not text:
            return ""
        
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
        
        cleaned_text = text
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].lstrip()
        
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