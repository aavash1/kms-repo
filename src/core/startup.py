# src/core/startup.py
import os
import logging
import chromadb
import ollama
import torch
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
from src.core.services.file_utils import (
    CHROMA_DIR,
    load_documents_to_chroma,
    set_globals
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

def create_prompt_template():
    """
    Create an improved prompt template that provides better guidance for the LLM
    to generate more relevant and accurate responses.
    """
    template = """system: 당신은 NetBackup 시스템 전문가입니다. 사용자의 질문에 정확하고 관련성 높은 답변을 제공해야 합니다.
    ### 응답 원칙:
    1. 항상 한국어로 답변하세요 (기술 용어는 영어 유지)
    2. 제공된 문서 정보만 사용하세요. 확실하지 않은 정보는 제공하지 마세요.
    3. 답변을 다음 구조로 구성하세요:
    - 문제 이해: 사용자 질문을 명확히 이해했음을 보여주세요
    - 기술 설명: 관련 기술 정보를 간결하게 설명하세요
    - 해결 방안: 구체적인 단계나 명령어를 제공하세요
    - 추가 정보: 필요한 경우 추가 정보를 요청하세요

    ### 기술적 정확성:
    - 명령어, 스크립트, 단계별 절차는 문서에 있는 그대로 정확히 제공하세요
    - 문서에 없는 내용은 추측하지 말고 "제공된 문서에서 이 정보를 찾을 수 없습니다"라고 명시하세요
    - 복잡한 기술 개념은 간결하게 설명하되 정확성을 유지하세요
    - 불확실한 부분은 명확히 "이 부분은 확실하지 않습니다"라고 표시하세요

    ### 예시 답변:
    사용자: NetBackup에서 카탈로그 백업이 실패했습니다. 어떻게 해결할 수 있나요?

    시스템: 
    NetBackup 카탈로그 백업 실패 문제를 해결해 드리겠습니다.

    **문제 이해:**
    카탈로그 백업은 NetBackup의 중요한 구성 요소로, 실패 시 전체 백업 시스템의 복구 능력에 영향을 줍니다.

    **기술 설명:**
    NetBackup 카탈로그는 모든 백업 데이터의 위치와 속성을 포함하는 데이터베이스입니다. 카탈로그 백업 실패의 일반적인 원인은 디스크 공간 부족, 권한 문제, 또는 네트워크 연결 문제입니다.

    **해결 방안:**
    1. 오류 로그 확인: `/usr/openv/netbackup/logs/admin` 디렉토리에서 로그를 확인하세요
    2. 디스크 공간 확인: `df -h` 명령으로 카탈로그가 저장된 볼륨의 공간을 확인하세요
    3. 카탈로그 백업 재시도: 
    4. 문제가 지속되면 NetBackup 서비스 재시작:
        /usr/openv/netbackup/bin/bp.kill_all
        /usr/openv/netbackup/bin/bp.start_all
 
**추가 정보:**
정확한 진단을 위해 발생한 구체적인 오류 메시지나 코드를 알려주실 수 있나요?

### 현재 대화 맥락:
{chat_history}

### 검색된 문서:
{context}

### 사용자 질문:
{query}

### 응답 (한국어로):
"""
    return ChatPromptTemplate.from_template(template)

def check_ollama_availability():
    """
    Check if Ollama is running and the required models are available.

    Raises:
        RuntimeError: If Ollama is not running or required models are not available.
    """
    try:
        response = ollama.list()
        logger.debug(f"Ollama list response: {response}")
        if "models" not in response:
            raise RuntimeError("Ollama response does not contain 'models' key. Ensure Ollama is running and accessible.")
        models = response["models"]
        if not isinstance(models, list):
            raise RuntimeError("Ollama 'models' response is not a list.")
        available_models = [model["name"].split(":")[0] for model in models if isinstance(model, dict) and "name" in model]
        required_models = ["mxbai-embed-large", "deepseek-r1:14b", "mistral"]
        missing_models = [model for model in required_models if model not in available_models]
        if missing_models:
            raise RuntimeError(
                f"The following required Ollama models are not available: {missing_models}. "
                f"Please pull the models using 'ollama pull <model>' (e.g., 'ollama pull mxbai-embed-large')."
            )
        logger.info("Ollama is running and required models are available.")
    except Exception as e:
        logger.error(f"Ollama check failed: {str(e)}")
        raise RuntimeError(
            f"Ollama is not running or models are not available: {str(e)}. "
            "Please ensure Ollama is running (e.g., 'ollama serve') and the required models are pulled."
        )

def startup_event():
    """
    Initialize all components and services.
    """
    print("Starting up: initializing handlers and Chroma collection...")
    global _components

    model_manager = ModelManager()
    logger.info(f"ModelManager initialized with device: {model_manager.get_device()}")

    try:
        from src.core.file_handlers.factory import FileHandlerFactory
        FileHandlerFactory.initialize(model_manager)
        # pdf_handler = PDFHandler(model_manager=model_manager)
        # doc_handler = AdvancedDocHandler(model_manager=model_manager)
        # hwp_handler = HWPHandler(model_manager=model_manager)
        # image_handler = ImageHandler(model_manager=model_manager)
        # msg_handler = MSGHandler(model_manager=model_manager)
        #granite_vision_extractor = GraniteVisionExtractor(model_name="llama3.2-vision")
        granite_vision_extractor = GraniteVisionExtractor(model_name="gemma3:12b")
        #gemma3:12b
        html_handler = HTMLContentHandler(model_manager=model_manager)
        translator = LocalMarianTranslator(model_manager=model_manager)

        # from src.core.file_handlers.factory import FileHandlerFactory
        # FileHandlerFactory.initialize(model_manager)

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Initialize ChromaDB
        try:
            persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                chroma_coll = persistent_client.get_collection(name="netbackup_docs")
                print("Found existing ChromaDB collection")
            except Exception:
                chroma_coll = persistent_client.create_collection(
                    name="netbackup_docs",
                    metadata={"hnsw:space": "cosine"}
                )
                print("Created new ChromaDB collection")

            from src.core.services.file_utils import _state
            _state.chromadb_collection = chroma_coll
            
            count = chroma_coll.count()
            print(f"ChromaDB collection initialized with {count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

        # Initialize embeddings and vector store
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

        logger.info("Skipping document loading during startup. Ensure documents are ingested via API endpoints.")

        # Initialize retriever and prompt template
        retriever = vector_store.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})
        prompt_template = create_prompt_template()

        # Initialize workflow and memory
        workflow = StateGraph(state_schema=MessagesState)
        memory = MemorySaver()
        llm = ChatOllama(model="gemma3:12b", stream=True)

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

        # Initialize RAG chain
        rag_chain = (
            {"context": retriever, "query": lambda x: x}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # Initialize translator
        translator = LocalMarianTranslator()

        # Set globals
        if not set_globals(chroma_coll=chroma_coll, rag=app, vect_store=vector_store, prompt=prompt_template, workflow=workflow, memory=memory):
            raise RuntimeError("Failed to set global state")

        from src.core.services.file_utils import get_global_prompt
        if not get_global_prompt():
            raise RuntimeError("Global prompt verification failed")

        # Initialize services
        query_service = QueryService(translator=translator, rag_chain=app, global_prompt=prompt_template)
        
        # Load pre-trained RL policy if available
        policy_path = "policy_network.pth"
        if os.path.exists(policy_path):
            query_service.load_policy(policy_path)
            logger.info(f"Loaded pre-trained RL policy from {policy_path}")

        ingest_service = IngestService(model_manager=model_manager)

        # Package initialized components
        initialized_components = {
            'model_manager': model_manager,
            'chroma_collection': chroma_coll,
            'vector_store': vector_store,
            'rag_chain': rag_chain,
            'query_service': query_service,
            'ingest_service': ingest_service,
            'file_handler_factory': FileHandlerFactory,  # Store factory class, not instance
            'document_handlers': {  # Optional: pre-instantiate if needed elsewhere
                'granite_vision': granite_vision_extractor,
                'html': html_handler
            },
            'workflow': workflow,
            'memory': memory
        }

        print("Startup complete: All components including RL-enhanced QueryService initialized successfully.")
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
        components = startup_event()
        init_service_instances(components)
        return components
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

# def verify_initialization(components):
#     """Verify that all required components are properly initialized."""
#     required_components = ['chroma_collection', 'vector_store', 'rag_chain', 'query_service', 'ingest_service', 'document_handlers', 'workflow', 'memory']
#     for component in required_components:
#         if component not in components or components[component] is None:
#             raise RuntimeError(f"Required component '{component}' not properly initialized")
#     handlers = components['document_handlers']
#     required_handlers = ['pdf', 'doc', 'hwp']
#     for handler in required_handlers:
#         if handler not in handlers or handlers[handler] is None:
#             raise RuntimeError(f"Required document handler '{handler}' not properly initialized")   
def verify_initialization(components):
    required_components = ['chroma_collection', 'vector_store', 'rag_chain', 'query_service', 'ingest_service', 'file_handler_factory', 'workflow', 'memory']
    for component in required_components:
        if component not in components or components[component] is None:
            raise RuntimeError(f"Required component '{component}' not properly initialized")
    factory = components['file_handler_factory']
    required_handlers = ['pdf', 'doc', 'hwp','excel', 'pptx']
    for handler in required_handlers:
        if handler not in factory._handlers:
            raise RuntimeError(f"Required document handler '{handler}' not registered in FileHandlerFactory")