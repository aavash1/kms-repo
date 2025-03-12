# src/core/startup.py
#from langchain.schema.output_parser import StrOutputParser
import os
import logging
import chromadb
import ollama
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


from src.core.processing.local_translator import LocalMarianTranslator
from src.core.services.query_service import QueryService
from src.core.services.ingest_service import IngestService
from src.core.services.file_utils import (
    CHROMA_DIR,
    load_documents_to_chroma,
    set_globals
)

# Initialize logger
logger = logging.getLogger(__name__)

def create_prompt_template():
    """Create a balanced prompt template that encourages natural, technically precise responses"""
    template = """system: NetBackup 시스템 전문가로서 사용자와 자연스러운 대화를 통해 기술적 지원을 제공합니다.

    대화 원칙:
    1. 불필요한 격식이나 설명 없이 바로 핵심을 다룹니다
    2. 문제나 질문의 맥락을 파악하고 필요한 경우 구체적인 상황을 물어봅니다
    3. 답변은 다음 요소를 자연스럽게 포함합니다:
       - 현재 상황 이해/확인
       - 관련된 기술적 설명
       - 실행 가능한 해결 방안
       - 필요한 경우 추가 정보 요청
    
    기술적 정확성:
    - 제공된 문서 정보만 사용하세요. 문서에 없는 내용은 지어내지 마세요.
    - 명령어, 스크립트, 단계별 절차는 문서에 있는 그대로 정확히 제공하세요.
    - 기술 용어, 명령어, 제품명은 영문 유지
    - 나머지는 자연스러운 한국어로 설명
    - 불확실한 부분은 명확히 확인 요청

    예시 답변 구조:
    "현재 상황을 이해했습니다. [관련 기술 설명]
    
    이 문제를 해결하려면 [해결 방안]이 필요합니다.
    
    혹시 [추가 정보]를 알려주실 수 있나요? 그러면 더 구체적인 도움을 드릴 수 있습니다."

    # Remember: ALL responses must be in Korean!

    채팅 기록: {chat_history}
    검색된 문서: {context}
    사용자 질문: {query}

    응답 (반드시 한국어로): """
    return ChatPromptTemplate.from_template(template)

def startup_event():
    """
    Initialize all components and services.
    """
    print("Starting up: initializing handlers and Chroma collection...")

    try:
        # Step 1: Initialize document handlers
        pdf_handler = PDFHandler()
        doc_handler = AdvancedDocHandler()
        hwp_handler = HWPHandler()
        image_handler = ImageHandler()

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Step 2: Initialize ChromaDB
        try:
            persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                # Try to get existing collection
                chroma_coll = persistent_client.get_collection(name="netbackup_docs")
                print("Found existing ChromaDB collection")
            except Exception:
                # If collection doesn't exist, create it
                chroma_coll = persistent_client.create_collection(
                    name="netbackup_docs",
                    metadata={"hnsw:space": "cosine"}
                )
                print("Created new ChromaDB collection")

            # Set the global state immediately
            from src.core.services.file_utils import _state
            _state.chromadb_collection = chroma_coll
            
            count = chroma_coll.count()
            print(f"ChromaDB collection initialized with {count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

        # Step 3: Initialize embeddings and vector store
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

        # Step 4: Load documents if needed
        if chroma_coll.count() == 0:
            print("Chroma collection is empty; loading documents...")
            load_documents_to_chroma(pdf_handler, doc_handler, hwp_handler)
        else:
            print(f"Chroma collection contains {chroma_coll.count()} documents; skipping document ingestion.")

        # Step 5: Initialize retriever and create prompt template
        retriever = vector_store.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.5})
        prompt_template = create_prompt_template()

        workflow = StateGraph(state_schema=MessagesState)
        memory = MemorySaver()
        llm = ChatOllama(model="deepseek-r1:14b", stream=True)

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

        # Step 6: Initialize RAG chain
        rag_chain = (
            {"context": retriever, "query": lambda x: x}
            | prompt_template
            | ChatOllama(model="deepseek-r1:14b", stream=True)
            | StrOutputParser()
        )

        # Step 7: Initialize translator
        translator = LocalMarianTranslator()

        # Step 8: Set globals for backward compatibility
        if not set_globals(chroma_coll=chroma_coll, rag=app, vect_store=vector_store, prompt=prompt_template, workflow=workflow, memory=memory):
            raise RuntimeError("Failed to set global state")

        from src.core.services.file_utils import get_global_prompt

        if not get_global_prompt():
            raise RuntimeError("Global prompt verification failed")

        # Step 9: Initialize services with correct dependencies
        query_service = QueryService(vector_store=vector_store, translator=translator, rag_chain=app, global_prompt=prompt_template)
        ingest_service = IngestService()

        # Step 10: Return initialized components
        initialized_components = {
            'chroma_collection': chroma_coll,
            'vector_store': vector_store,
            'rag_chain': rag_chain,
            'query_service': query_service,
            'ingest_service': ingest_service,
            'document_handlers': {'pdf': pdf_handler, 'doc': doc_handler, 'hwp': hwp_handler, 'image': image_handler},
            'workflow': workflow,
            'memory': memory
        }

        print("Startup complete: All components including LangChain RAG initialized successfully.")
        return initialized_components

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

def verify_initialization(components):
    """Verify that all required components are properly initialized."""
    required_components = ['chroma_collection', 'vector_store', 'rag_chain', 'query_service', 'ingest_service', 'document_handlers', 'workflow', 'memory']
    for component in required_components:
        if component not in components or components[component] is None:
            raise RuntimeError(f"Required component '{component}' not properly initialized")
    handlers = components['document_handlers']
    required_handlers = ['pdf', 'doc', 'hwp']
    for handler in required_handlers:
        if handler not in handlers or handlers[handler] is None:
            raise RuntimeError(f"Required document handler '{handler}' not properly initialized")