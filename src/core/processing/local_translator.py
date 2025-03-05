# File: src/core/processing/local_translator.py

import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LocalMarianTranslator:
    def __init__(self, model_path=None, device="cuda"):
        if model_path is None:
            #model_path = r"C:\AI_Models\local_cache\models--Helsinki-NLP--opus-mt-tc-big-en-ko\snapshots\ae8606b7b29a495f31ce679cee2007f536a3a5ce"
            model_path=r"C:\AI_Models\local_cache\models--QuoQA-NLP--KE-T5-En2Ko-Base\merged_model"

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model from local cache
        print(f"Loading tokenizer and model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def split_into_segments(self, text):
        """Split text into logical segments while preserving structure."""
        if not text:
            return []

        # Split on double newlines to preserve paragraph structure
        paragraphs = text.split('\n\n')
        segments = []
        
        for paragraph in paragraphs:
            # Handle bullet points and numbered items
            if re.match(r'^(VMware item \d+:|[-•])', paragraph.strip()):
                segments.append(paragraph.strip())
            else:
                # Split long paragraphs into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
                current_segment = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    if current_length + sentence_tokens > 200:  # Conservative token limit
                        if current_segment:
                            segments.append(' '.join(current_segment))
                        current_segment = [sentence]
                        current_length = sentence_tokens
                    else:
                        current_segment.append(sentence)
                        current_length += sentence_tokens
                
                if current_segment:
                    segments.append(' '.join(current_segment))
        
        return segments

    def translate_segment(self, segment):
        """Translate a single segment with enhanced control."""
        inputs = self.tokenizer([segment], return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                repetition_penalty=2.5,  # Increased repetition penalty
                no_repeat_ngram_size=4,  # Prevent 4-gram repetitions
                early_stopping=True,
                temperature=0.7  # Add some randomness to prevent repetition
            )
        
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to remove common repetition patterns
        translation = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1', translation)
        return translation

    def translate_text(self, text):
        """Translate text with improved segment handling."""
        if not text or not isinstance(text, str):
            return text

        # Split into manageable segments
        segments = self.split_into_segments(text)
        
        # Translate each segment
        translations = []
        for segment in segments:
            translation = self.translate_segment(segment)
            translations.append(translation)
        
        # Join segments while preserving structure
        result = '\n\n'.join(translations)
        
        # Final cleanup of any remaining repetitions
        result = re.sub(r'(\b[가-힣]+\b)(\s+\1\b)+', r'\1', result)
        return result

    def translate_batch(self, texts, batch_size=4):
        """Translate a batch of texts with improved handling."""
        if not texts:
            return []
        
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = []
            
            for text in batch:
                try:
                    translation = self.translate_text(text)
                    batch_translations.append(translation)
                except Exception as e:
                    print(f"Error translating text: {e}")
                    batch_translations.append(text)
            
            translations.extend(batch_translations)
        
        return translations
