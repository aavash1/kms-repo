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
        """Initialize all models used in the project."""
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

        # Initialize models
        self._load_trocr_models()
        self._load_klue_models()
        self._load_marian_models()
        self._load_embedding_model()  # Add embedding model initialization

        self._initialized = True
        logger.info("ModelManager: All models initialized successfully")

    def _load_trocr_models(self):
        """Load TrOCR processor and model."""
        try:
            logger.info("Loading TrOCR models from local cache...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                self.trocr_snapshot,
                local_files_only=True,
                use_fast=True
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                self.trocr_snapshot,
                local_files_only=True
            ).to(self.device)
            self.trocr_model.eval()
            logger.info("TrOCR models loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load TrOCR models: {str(e)}")
            raise RuntimeError(f"TrOCR model initialization failed: {str(e)}")

    def _load_klue_models(self):
        """Load KLUE BERT tokenizer and model."""
        try:
            logger.info("Loading KLUE BERT from local cache...")
            self.klue_tokenizer = AutoTokenizer.from_pretrained(
                self.klue_snapshot,
                local_files_only=True
            )
            self.klue_bert = AutoModel.from_pretrained(
                self.klue_snapshot,
                local_files_only=True
            ).to(self.device)
            self.klue_bert.eval()
            logger.info("KLUE BERT loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load KLUE BERT models: {str(e)}")
            raise RuntimeError(f"KLUE BERT model initialization failed: {str(e)}")

    def _load_marian_models(self):
        """Load Marian MT tokenizer and model for translation."""
        try:
            logger.info("Loading Marian MT models from local cache...")
            self.marian_tokenizer = AutoTokenizer.from_pretrained(
                self.marian_snapshot,
                local_files_only=True
            )
            self.marian_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.marian_snapshot,
                local_files_only=True
            ).to(self.device)
            self.marian_model.eval()
            logger.info("Marian MT models loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Marian MT models: {str(e)}")
            raise RuntimeError(f"Marian MT model initialization failed: {str(e)}")

    def _load_embedding_model(self):
        """Load the embedding model using langchain_ollama."""
        try:
            logger.info("Loading embedding model (Ollama mxbai-embed-large)...")
            self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
            logger.info("Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}")

    # Getter methods
    def get_trocr_processor(self):
        return self.trocr_processor

    def get_trocr_model(self):
        return self.trocr_model

    def get_klue_tokenizer(self):
        return self.klue_tokenizer

    def get_klue_bert(self):
        return self.klue_bert
    
    def get_marian_tokenizer(self):
        return self.marian_tokenizer
    
    def get_marian_model(self):
        return self.marian_model

    def get_embedding_model(self):
        """Return the embedding model for generating embeddings."""
        if not hasattr(self, 'embedding_model'):
            raise AttributeError("Embedding model not initialized. Call _load_embedding_model first.")
        return self.embedding_model

    def get_device(self):
        return self.device

    def cleanup(self):
        """Move models to CPU and free up GPU memory."""
        try:
            if not self._initialized:
                logger.debug("ModelManager not initialized, skipping cleanup")
                return

            if hasattr(self, 'trocr_model'):
                self.trocr_model.cpu()
                del self.trocr_model
            if hasattr(self, 'klue_bert'):
                self.klue_bert.cpu()
                del self.klue_bert
            if hasattr(self, 'marian_model'):
                self.marian_model.cpu()
                del self.marian_model
            if hasattr(self, 'embedding_model'):
                del self.embedding_model  # OllamaEmbeddings doesn't need GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ModelManager: Cleaned up models and freed GPU memory.")
            
            # Reset initialization flag for potential future reuse
            self._initialized = False
        except Exception as e:
            logger.error(f"ModelManager cleanup failed: {str(e)}")