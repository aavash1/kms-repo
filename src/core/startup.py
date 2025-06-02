# src/core/startup.py
import os
import logging
import chromadb
import ollama
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from src.core.file_handlers.doc_handler import AdvancedDocHandler
from src.core.file_handlers.pdf_handler import PDFHandler
from src.core.file_handlers.hwp_handler import HWPHandler
from src.core.file_handlers.image_handler import ImageHandler
from src.core.file_handlers.msg_handler import MSGHandler
from src.core.file_handlers.excel_handler import ExcelHandler
from src.core.file_handlers.pptx_handler import PPTXHandler
from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
from src.core.models.model_manager import ModelManager
from src.core.processing.local_translator import LocalMarianTranslator
from src.core.services.query_service import QueryService
from src.core.services.ingest_service import IngestService
from src.core.services.chat_vector_manager import get_chat_vector_manager
from src.core.services.file_utils import (
    CHROMA_DIR,
    load_documents_to_chroma,
    set_globals,
    get_personal_vector_store
)

from contextlib import asynccontextmanager
from fastapi import FastAPI

# Initialize logger
logger = logging.getLogger(__name__)

_components = None

def get_components():
    """Dependency to access pre-initialized components."""
    global _components
    if _components is None:
        raise RuntimeError("Application components not initialized. Startup failed.")
    return _components

def create_prompt_template() -> str:
    template = """당신은 DSTI Chatbot UI 어시스턴트입니다. 다음 규칙에 따라 응답하세요:

### 기본 규칙 ###
1. **모든 응답은 한국어로 작성**
2. **적절한 제목 사용** 
3. **1인칭 대명사 사용 금지**

### 문서 기반 질문 시 ###
- {context}가 제공되었을 때만 적용:
  1. 문서 내용을 요약하여 설명
  2. 직접 인용은 최소화
  3. 표/코드/수식은 원본 형식 유지

### 일반 대화 시 ###
- 자연스러운 대화 유지
- 간결하고 친절한 어조 사용

대화 내역:
{chat_history}

문서 컨텍스트:
{context}

사용자 질문:
{query}

어시스턴트 응답 (한국어):
"""
    return ChatPromptTemplate.from_template(template)

def check_ollama_availability():
    """
    Check if Ollama is running and the required models are available.
    Now with flexible model matching and non-blocking behavior.
    """
    try:
        response = ollama.list()
        logger.debug(f"Ollama list response: {response}")
        if "models" not in response:
            logger.warning("Ollama response does not contain 'models' key.")
            return False
            
        models = response["models"]
        if not isinstance(models, list):
            logger.warning("Ollama 'models' response is not a list.")
            return False
            
        # Get full model names (including versions)
        available_models = []
        for model in models:
            if isinstance(model, dict) and "model" in model:
                model_name = model["model"]
                # Remove version tags like ":latest" for comparison
                base_name = model_name.split(":")[0]
                available_models.append(base_name)
        
        # Flexible required models matching
        core_models = ["mxbai-embed-large", "gemma3", "mistral"]
        
        missing_models = []
        for required in core_models:
            found = any(available.startswith(required) for available in available_models)
            if not found:
                missing_models.append(required)
        
        if missing_models:
            logger.warning(f"Some models not found: {missing_models}")
            logger.info(f"Available models: {available_models}")
        else:
            logger.info("All required Ollama models are available.")
            
        return len(missing_models) == 0
        
    except Exception as e:
        logger.warning(f"Ollama check failed: {str(e)}")
        return False

async def initialize_chromadb():
    """Initialize ChromaDB collection asynchronously."""
    try:
        # Run ChromaDB operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _init_chroma():
            persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                chroma_coll = persistent_client.get_collection(name="netbackup_docs")
                logger.info("Found existing ChromaDB collection")
            except Exception:
                chroma_coll = persistent_client.create_collection(
                    name="netbackup_docs",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new ChromaDB collection")
            
            count = chroma_coll.count()
            logger.info(f"ChromaDB collection initialized with {count} documents")
            return persistent_client, chroma_coll
            
        persistent_client, chroma_coll = await loop.run_in_executor(None, _init_chroma)
        
        # Update global state
        from src.core.services.file_utils import _state
        _state.chromadb_collection = chroma_coll
        
        return persistent_client, chroma_coll
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise RuntimeError(f"ChromaDB initialization failed: {e}")

async def initialize_heavy_components():
    """Initialize heavy components that can be loaded lazily."""
    def _init_vision_extractor():
        return GraniteVisionExtractor(model_name="gemma3:4b", fallback_model="granite3.2-vision")
    
    loop = asyncio.get_event_loop()
    granite_vision_extractor = await loop.run_in_executor(None, _init_vision_extractor)
    return granite_vision_extractor

class LazyComponentLoader:
    """Lazy loader for components that don't need to be initialized at startup."""
    def __init__(self):
        self._granite_vision_extractor = None
        self._html_handler = None
        self._translator = None
    
    @property
    def granite_vision_extractor(self):
        if self._granite_vision_extractor is None:
            logger.info("Lazy loading GraniteVisionExtractor...")
            self._granite_vision_extractor = GraniteVisionExtractor(
                model_name="gemma3:4b", 
                fallback_model="granite3.2-vision"
            )
        return self._granite_vision_extractor
    
    @property
    def html_handler(self):
        if self._html_handler is None:
            logger.info("Lazy loading HTMLContentHandler...")
            model_manager = ModelManager()  # Reuse singleton pattern
            self._html_handler = HTMLContentHandler(model_manager=model_manager)
        return self._html_handler
    
    @property
    def translator(self):
        if self._translator is None:
            logger.info("Lazy loading LocalMarianTranslator...")
            model_manager = ModelManager()
            self._translator = LocalMarianTranslator(model_manager=model_manager)
        return self._translator

async def startup_event():
    """
    Initialize all components and services with optimized async loading.
    """
    print("Starting up: initializing handlers and Chroma collection...")
    global _components
    
    start_time = asyncio.get_event_loop().time()

    # Step 1: Check Ollama availability (non-blocking)
    ollama_available = check_ollama_availability()
    if not ollama_available:
        logger.warning("Ollama models may not be fully available. Continuing startup...")

    # Step 2: Initialize core components
    model_manager = ModelManager()
    logger.info(f"ModelManager initialized with device: {model_manager.get_device()}")

    try:
        # Step 3: Initialize file handlers (lightweight)
        from src.core.file_handlers.factory import FileHandlerFactory
        FileHandlerFactory.initialize(model_manager)
        logger.info("FileHandlerFactory initialized")
        
        # Step 4: Set environment variables early
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Step 5: Initialize ChromaDB asynchronously
        persistent_client, chroma_coll = await initialize_chromadb()

        # Step 6: Initialize vector components (can be done in parallel)
        async def init_vector_store():
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            
            def custom_relevance_score_fn(distance):
                return 1.0 - distance / 2
            
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain.vectorstores import Chroma

            vector_store = Chroma(
                client=persistent_client,
                embedding_function=embeddings,
                collection_name="netbackup_docs",
                relevance_score_fn=custom_relevance_score_fn,
                collection_metadata={"hnsw:space": "cosine"}
            )
            return vector_store, embeddings

        async def init_workflow_components():
            # Initialize workflow and memory
            workflow = StateGraph(state_schema=MessagesState)
            memory = MemorySaver()
            llm = ChatOllama(model="gemma3:12b", stream=True)
            
            # Initialize retriever and prompt template
            prompt_template = create_prompt_template()
            
            return workflow, memory, llm, prompt_template

        # Run vector store and workflow initialization in parallel
        vector_task = asyncio.create_task(init_vector_store())
        workflow_task = asyncio.create_task(init_workflow_components())
        
        # Wait for both to complete
        (vector_store, embeddings), (workflow, memory, llm, prompt_template) = await asyncio.gather(
            vector_task, workflow_task
        )

        logger.info("Vector store and workflow components initialized")

        # Step 7: Complete workflow setup
        retriever = vector_store.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})
        
        def call_model(state: MessagesState):
            current_query = state["messages"][-1].content if state["messages"] else ""
            context = retriever.invoke(current_query)
            messages = prompt_template.invoke({
                "context": context,
                "chat_history": state["messages"][:-1],
                "query": current_query
            })
            response = llm.invoke(messages)
            return {"messages": response}
        
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        app = workflow.compile(checkpointer=memory)

        # Step 8: Initialize RAG chain
        rag_chain = (
            {"context": retriever, "query": lambda x: x}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # Step 9: Set globals
        if not set_globals(chroma_coll=chroma_coll, rag=app, vect_store=vector_store, 
                          prompt=prompt_template, workflow=workflow, memory=memory):
            raise RuntimeError("Failed to set global state")

        from src.core.services.file_utils import get_global_prompt
        if not get_global_prompt():
            raise RuntimeError("Global prompt verification failed")

        # Step 10: Initialize services with lazy loading
        lazy_loader = LazyComponentLoader()
        
        # Use lazy translator for QueryService
        query_service = QueryService(
            translator=lazy_loader.translator, 
            rag_chain=app, 
            global_prompt=prompt_template
        )
        
        # Load pre-trained RL policy if available (non-blocking)
        policy_path = "policy_network.pth"
        if os.path.exists(policy_path):
            try:
                query_service.load_policy(policy_path)
                logger.info(f"Loaded pre-trained RL policy from {policy_path}")
            except Exception as e:
                logger.warning(f"Failed to load RL policy: {e}")

        ingest_service = IngestService(model_manager=model_manager)
        
        # Step 11: Initialize and start the chat vector manager (lightweight)
        chat_vector_manager = get_chat_vector_manager()
        chat_vector_manager.start()

        # Step 12: Package initialized components
        initialized_components = {
            'model_manager': model_manager,
            'chroma_collection': chroma_coll,
            'vector_store': vector_store,
            'rag_chain': rag_chain,
            'query_service': query_service,
            'ingest_service': ingest_service,
            'file_handler_factory': FileHandlerFactory,
            'document_handlers': {
                'granite_vision': lazy_loader.granite_vision_extractor,  # Lazy loaded
                'html': lazy_loader.html_handler  # Lazy loaded
            },
            'workflow': workflow,
            'memory': memory,
            'chat_vector_manager': chat_vector_manager,
            'lazy_loader': lazy_loader  # Provide access to lazy loader
        }

        end_time = asyncio.get_event_loop().time()
        elapsed_time = end_time - start_time
        
        print(f"Startup complete: All components initialized successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Startup completed in {elapsed_time:.2f} seconds")
        
        _components = initialized_components
        return _components

    except Exception as e:
        error_msg = f"Failed to initialize application: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

# Global service instances for FastAPI
query_service_instance = None
ingest_service_instance = None

def init_service_instances(components):
    """Initialize global service instances for FastAPI"""
    global query_service_instance, ingest_service_instance
    query_service_instance = components['query_service']
    ingest_service_instance = components['ingest_service']

async def startup():
    """FastAPI startup event handler"""
    try:
        components = await startup_event()  # Now async
        init_service_instances(components)
        return components
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

def verify_initialization(components):
    required_components = ['chroma_collection', 'vector_store', 'rag_chain', 'query_service', 
                          'ingest_service', 'file_handler_factory', 'workflow', 'memory']
    for component in required_components:
        if component not in components or components[component] is None:
            raise RuntimeError(f"Required component '{component}' not properly initialized")
    
    factory = components['file_handler_factory']
    required_handlers = ['pdf', 'doc', 'hwp', 'excel', 'pptx']
    for handler in required_handlers:
        if handler not in factory._handlers:
            raise RuntimeError(f"Required document handler '{handler}' not registered in FileHandlerFactory")