# src/api/routes/query.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, Response, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.core.services.query_service import QueryService
from src.core.services.file_utils import (CHROMA_DIR, set_globals, get_vector_store, get_rag_chain, get_global_prompt, get_workflow, get_memory)
from src.core.processing.local_translator import LocalMarianTranslator
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage 
import logging
logger = logging.getLogger(__name__)
router = APIRouter()
import asyncio
from typing import Optional
import uuid
from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import numpy as np
from src.core.startup import get_components

class AsyncTokenStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put_nowait(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put_nowait(None)

    async def stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

def get_query_service():
    """Get QueryService from startup components with fallback validation."""
    try:
        components = get_components()
        query_service = components.get('query_service')
        if query_service:
            return query_service
        
        # Fallback - recreate if not available (shouldn't happen)
        logger.warning("QueryService not found in components, recreating...")
        rag_chain = get_rag_chain()
        if not rag_chain:
            raise RuntimeError("RAG chain not initialized")
        global_prompt = get_global_prompt()
        if not global_prompt:
            raise RuntimeError("Global prompt not initialized")
        return QueryService(
            translator=LocalMarianTranslator(),
            rag_chain=rag_chain,
            global_prompt=global_prompt
        )
    except Exception as e:
        logger.error(f"Error getting QueryService: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str

@router.post("/", summary="Submit a query and get the generated answer")
async def query_endpoint(request: QueryRequest, query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.process_basic_query(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@router.get("/get", summary="Submit a query via GET and get the generated answer")
async def query_get_endpoint(query: str, query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.process_basic_query(query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@router.post("/stream", summary="Submit a query and stream the generated response")
async def query_stream_endpoint(request: QueryRequest, query_service: QueryService = Depends(get_query_service)):
    try:
        streaming_llm, formatted_prompt = await query_service.process_streaming_query(request.query)
        
        async def token_generator():
            async for chunk in streaming_llm.astream(formatted_prompt):
                yield chunk.content

        return StreamingResponse(token_generator(), media_type="text/plain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing streaming query: {e}")

@router.get("/stream-get", summary="Submit a GET query and stream the generated response")
async def query_stream_get_endpoint(
    query: str, 
    conversation_id: Optional[str] = None,
    plain_text: bool = False,
    filter_document_id: Optional[str] = Query(
        default=None,
        description="Restrict search to a single uploaded file id"),
    query_service: QueryService = Depends(get_query_service)):
    """
    Streams chat completion.  
    *If **filter_document_id** is supplied the answer is generated only
    from vectors that belong to that document (private chat-with-file).*
    """
    try:
        #logger.info(f"Received request: query='{query}', conversation_id={conversation_id}")
        logger.info("GET /stream: q=%s cid=%s doc=%s",
                    query, conversation_id, filter_document_id)
        
        # Process the query with RL enhancement
        streaming_llm, messages, conversation_id = await query_service.process_streaming_query(query, conversation_id,plain_text=plain_text, filter_document_id=filter_document_id)
        logger.info(f"Query processed successfully, streaming response for conversation {conversation_id}")
        
        full_response = ""
        #embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        #query_embedding = np.array(embedding_model.embed_query(query))
        #chunks = query_service.get_relevant_chunks(query)
        
        async def token_generator():
            nonlocal full_response
            buffer= ""
            try:
                logger.info(f"Starting token streaming for conversation {conversation_id}")       
                async for chunk in streaming_llm.astream(messages):
                    content = chunk.content
                    if not content or "<think>" in content or "###" in content:
                        continue
                    
                    full_response += content
                    buffer += content
                    
                    # Split buffer at newlines to preserve Markdown structure
                    while '\n' in buffer:
                        # Find the first newline
                        newline_index = buffer.index('\n')
                        # Yield everything up to and including the newline
                        yield buffer[:newline_index + 1]
                        # Keep the rest in the buffer
                        buffer = buffer[newline_index + 1:]
                        await asyncio.sleep(0.02)  # Small delay for smooth streaming
                    
                    # If buffer is too long but no newline, yield it to avoid delays
                    if len(buffer) >= 100:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.02)
                
                # Yield any remaining buffer content
                if buffer:
                    yield buffer
                    
                logger.info(f"Completed streaming response")
            
            except Exception as e:
                logger.error(f"Error during token streaming: {e}", exc_info=True)
                yield f"\n\n오류가 발생했습니다: {str(e)}\n"
        
        return StreamingResponse(
            token_generator(), 
            media_type="text/markdown",  # Changed to support Markdown rendering
            headers={"X-Conversation-ID": conversation_id}
        )
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        async def error_generator():
            yield "오류가 발생했습니다: 벡터 데이터베이스가 비어 있거나 관련 정보가 없을 수 있습니다."
        return StreamingResponse(
            error_generator(),
            media_type="text/plain",
            headers={"X-Conversation-ID": conversation_id or str(uuid.uuid4())}
        )

@router.get("/vectorSimilaritySearch", summary="Perform similarity search within a specific status code")
async def similarity_search_by_vector(
    query: str = None,  # Make query optional
    status_code: str = Query(..., description="The status code to search within"),
    query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.search_by_vector(query, status_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {e}")

@router.post("/refresh-stores", summary="Refresh vector stores to pick up newly added files")
async def refresh_vector_stores(
    query_service: QueryService = Depends(get_query_service)
):
    """Refresh vector stores to pick up newly added files without requiring a server restart."""
    try:
        success = query_service.refresh_stores()
        if success:
            return {"status": "success", "message": "Vector stores refreshed successfully"}
        else:
            return {"status": "error", "message": "Failed to refresh vector stores"}
    except Exception as e:
        logger.error(f"Error refreshing vector stores: {e}", exc_info=True)
        return {"status": "error", "message": f"Error refreshing vector stores: {str(e)}"}

@router.delete("/resetChromaCollection", summary="Reset the ChromaDB collection")
async def reset_collection(response: Response = None):
    try:
        import chromadb
        persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
        persistent_client.delete_collection("netbackup_docs")
        logger.info("Deleted existing ChromaDB collection 'netbackup_docs'")

        chroma_coll = persistent_client.create_collection(
            name="netbackup_docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new ChromaDB collection 'netbackup_docs'")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain.vectorstores import Chroma

        vector_store = Chroma(
            client=persistent_client,
            embedding_function=embeddings,
            collection_name="netbackup_docs",
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        set_globals(
            chroma_coll=chroma_coll,
            rag=get_rag_chain(),
            vect_store=vector_store,
            prompt=get_global_prompt(),
            workflow=get_workflow(),
            memory=get_memory()
        )
        logger.debug("Updated global state after reset")

        # Refresh QueryService to use the new collection
        try:
            # Use the same get_query_service() function that all other routes use
            query_service_instance = get_query_service()
            success = query_service_instance.refresh_stores()
            if success:
                logger.info("Successfully refreshed QueryService stores after collection reset")
            else:
                logger.warning("Failed to refresh QueryService stores after collection reset")
        except Exception as refresh_error:
            logger.error(f"Error refreshing QueryService stores: {refresh_error}")

        # Cache for 1 hour (3600 seconds) since reset is infrequent
        response.headers["Cache-Control"] = "public, max-age=3600"
        return {"message": "Collection 'netbackup_docs' has been deleted and reinitialized."}
    except Exception as e:
        logger.error(f"Error resetting collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error resetting collection: {e}")

# Direct streaming endpoint (unchanged)
@router.get("/direct-stream", summary="Direct streaming without batch processing")
async def direct_stream_endpoint(
    query: str, 
    conversation_id: Optional[str] = None, 
    query_service: QueryService = Depends(get_query_service)
):
    try:
        logger.info(f"Received direct stream request: query='{query}', conversation_id={conversation_id}")
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        if not hasattr(query_service, 'conversation_histories'):
            query_service.conversation_histories = {}
            
        if conversation_id not in query_service.conversation_histories:
            query_service.conversation_histories[conversation_id] = []
            
        conversation_history = query_service.conversation_histories.get(conversation_id, [])

        docs = query_service.vector_store.similarity_search(query)
        context = "\n".join(doc.page_content[:600] for doc in docs)

        history_text = ""
        if conversation_history:
            history_pairs = []
            for i in range(0, len(conversation_history), 2):
                if i+1 < len(conversation_history):
                    user_msg = conversation_history[i].content
                    assistant_msg = conversation_history[i+1].content
                    history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
            history_text = "\n\n".join(history_pairs)
        
        korean_instruction = "당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 명확하게 답변하세요. 기술 용어만 영어로 유지하세요."
        query_with_context = f"대화 기록:\n{history_text}\n\n문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:"
        
        messages = [
            SystemMessage(content=korean_instruction),
            HumanMessage(content=query_with_context)
        ]
        
        query_service.conversation_histories[conversation_id].append(HumanMessage(content=query))
        
        token_handler = AsyncTokenStreamHandler()
        streaming_llm = ChatOllama(
            model="deepseek-r1:14b", 
            streaming=True, 
            callbacks=[token_handler],
            temperature=0.3
        )
        
        full_response = ""
        
        async def token_generator():
            nonlocal full_response
            try:
                logger.info(f"Starting direct token streaming for conversation {conversation_id}")
                token_count = 0
                
                async for chunk in streaming_llm.astream(messages):
                    content = chunk.content
                    if not content or "<think>" in content or "###" in content:
                        continue
                    full_response += content
                    token_count += 1
                    yield content
                
                logger.info(f"Completed direct streaming {token_count} tokens")
                
                if full_response:
                    query_service.conversation_histories[conversation_id].append(
                        AIMessage(content=full_response)
                    )
            
            except Exception as e:
                logger.error(f"Error during direct token streaming: {e}", exc_info=True)
                yield f"\n\n오류가 발생했습니다: {str(e)}\n"
        
        return StreamingResponse(
            token_generator(), 
            media_type="text/plain",
            headers={"X-Conversation-ID": conversation_id}
        )
    
    except Exception as e:
        logger.error(f"Direct streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with direct streaming: {str(e)}")