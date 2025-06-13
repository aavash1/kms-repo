# src/core/models/model_manager.py
import os
import torch
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from langchain_ollama import OllamaEmbeddings  # Add langchain_ollama for embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize ModelManager with lazy loading - only set up paths and device."""
        if self._initialized:
            logger.debug("ModelManager already initialized, skipping initialization")
            return
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"ModelManager: Using device: {self.device}")

        # Paths to model snapshots from environment variables
        self.trocr_snapshot = os.environ.get(
            "TROCR_MODEL_PATH",
            r"C:\AI_Models\local_cache\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee"
        )
        self.klue_snapshot = os.environ.get(
            "KLUE_MODEL_PATH",
            r"C:\AI_Models\local_cache\models--klue--bert-base\snapshots\77c8b3d707df785034b4e50f2da5d37be5f0f546"
        )
        self.marian_snapshot = os.environ.get(
            "MARIAN_MODEL_PATH",
            r"C:\AI_Models\local_cache\models--QuoQA-NLP--KE-T5-En2Ko-Base\merged_model"
        )

        # Initialize lazy loading flags - models will be loaded on first access
        self._trocr_processor = None
        self._trocr_model = None
        self._klue_tokenizer = None
        self._klue_bert = None
        self._marian_tokenizer = None
        self._marian_model = None
        self._embedding_model = None

        self._initialized = True
        logger.info("ModelManager: Initialized with lazy loading (models will load on first access)")

    def _load_trocr_models(self):
        """Load TrOCR processor and model on demand."""
        if self._trocr_processor is None or self._trocr_model is None:
            try:
                logger.info("Loading TrOCR models from local cache...")
                self._trocr_processor = TrOCRProcessor.from_pretrained(
                    self.trocr_snapshot,
                    local_files_only=True,
                    use_fast=True
                )
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    self.trocr_snapshot,
                    local_files_only=True
                ).to(self.device)
                self._trocr_model.eval()
                logger.info("TrOCR models loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load TrOCR models: {str(e)}")
                raise RuntimeError(f"TrOCR model initialization failed: {str(e)}")

    def _load_klue_models(self):
        """Load KLUE BERT tokenizer and model on demand."""
        if self._klue_tokenizer is None or self._klue_bert is None:
            try:
                logger.info("Loading KLUE BERT from local cache...")
                self._klue_tokenizer = AutoTokenizer.from_pretrained(
                    self.klue_snapshot,
                    local_files_only=True
                )
                self._klue_bert = AutoModel.from_pretrained(
                    self.klue_snapshot,
                    local_files_only=True
                ).to(self.device)
                self._klue_bert.eval()
                logger.info("KLUE BERT loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load KLUE BERT models: {str(e)}")
                raise RuntimeError(f"KLUE BERT model initialization failed: {str(e)}")

    def _load_marian_models(self):
        """Load Marian MT tokenizer and model on demand."""
        if self._marian_tokenizer is None or self._marian_model is None:
            try:
                logger.info("Loading Marian MT models from local cache...")
                self._marian_tokenizer = AutoTokenizer.from_pretrained(
                    self.marian_snapshot,
                    local_files_only=True
                )
                self._marian_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.marian_snapshot,
                    local_files_only=True
                ).to(self.device)
                self._marian_model.eval()
                logger.info("Marian MT models loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load Marian MT models: {str(e)}")
                raise RuntimeError(f"Marian MT model initialization failed: {str(e)}")

    def _load_embedding_model(self):
        """Load sentence-transformers embedding model with GPU acceleration on demand."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from dotenv import load_dotenv
                load_dotenv()
                
                model_path = os.getenv('EMBEDDING_MODEL_PATH')  
                model_name = os.getenv('EMBEDDING_MODEL_NAME', 'mixedbread-ai/mxbai-embed-large-v1')
                
                logger.info("Loading sentence-transformers embedding model...")
                
                if model_path and os.path.exists(model_path):
                    # Load from local cache
                    self._embedding_model = SentenceTransformer(model_path, device=self.device)
                    logger.info(f"Embedding model loaded from cache: {model_path}")
                else:
                    # Download and cache
                    cache_dir = os.path.dirname(model_path) if model_path else r"C:\AI_Models\local_cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    self._embedding_model = SentenceTransformer(
                        model_name,
                        cache_folder=cache_dir,
                        device=self.device
                    )
                    logger.info(f"Embedding model loaded and cached: {model_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise RuntimeError(f"Embedding model initialization failed: {str(e)}")

    # Getter methods with lazy loading
    def get_trocr_processor(self):
        """Get TrOCR processor, loading it if not already loaded."""
        self._load_trocr_models()
        return self._trocr_processor

    def get_trocr_model(self):
        """Get TrOCR model, loading it if not already loaded."""
        self._load_trocr_models()
        return self._trocr_model

    def get_klue_tokenizer(self):
        """Get KLUE tokenizer, loading it if not already loaded."""
        self._load_klue_models()
        return self._klue_tokenizer

    def get_klue_bert(self):
        """Get KLUE BERT model, loading it if not already loaded."""
        self._load_klue_models()
        return self._klue_bert
    
    def get_marian_tokenizer(self):
        """Get Marian tokenizer, loading it if not already loaded."""
        self._load_marian_models()
        return self._marian_tokenizer
    
    def get_marian_model(self):
        """Get Marian model, loading it if not already loaded."""
        self._load_marian_models()
        return self._marian_model

    def get_embedding_model(self):
        """Get embedding model, loading it if not already loaded."""
        self._load_embedding_model()
        return self._embedding_model

    def get_device(self):
        return self.device

    def cleanup(self):
        """Move models to CPU and free up GPU memory."""
        try:
            if not self._initialized:
                logger.debug("ModelManager not initialized, skipping cleanup")
                return

            # Clean up only loaded models
            if self._trocr_model is not None:
                self._trocr_model.cpu()
                del self._trocr_model
                self._trocr_model = None
            if self._klue_bert is not None:
                self._klue_bert.cpu()
                del self._klue_bert
                self._klue_bert = None
            if self._marian_model is not None:
                self._marian_model.cpu()
                del self._marian_model
                self._marian_model = None
            if self._embedding_model is not None:
                # Move sentence-transformers model to CPU
                if hasattr(self._embedding_model, 'to'):
                    self._embedding_model.to('cpu')
                del self._embedding_model
                self._embedding_model = None
                
            # Clean up tokenizers and processors
            if self._trocr_processor is not None:
                del self._trocr_processor
                self._trocr_processor = None
            if self._klue_tokenizer is not None:
                del self._klue_tokenizer
                self._klue_tokenizer = None
            if self._marian_tokenizer is not None:
                del self._marian_tokenizer
                self._marian_tokenizer = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ModelManager: Cleaned up models and freed GPU memory.")
            
            # Reset initialization flag for potential future reuse
            self._initialized = False
        except Exception as e:
            logger.error(f"ModelManager cleanup failed: {str(e)}")

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is already loaded."""
        model_mapping = {
            'trocr': self._trocr_model is not None,
            'klue': self._klue_bert is not None,
            'marian': self._marian_model is not None,
            'embedding': self._embedding_model is not None
        }
        return model_mapping.get(model_name, False)

    def preload_essential_models(self):
        """Preload only the most essential models (embedding model for startup)."""
        logger.info("Preloading essential models...")
        # Only load embedding model as it's needed for vector operations
        self._load_embedding_model()
        logger.info("Essential models preloaded")