# src/api/routes/query.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.core.services.query_service import QueryService
from src.core.services.file_utils import CHROMA_DIR, set_globals, get_vector_store, get_rag_chain, get_global_prompt, get_workflow, get_memory
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

class AsyncTokenStreamHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens"""
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
    """Get initialized QueryService instance with dependencies"""
    try:
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
        logger.error(f"Error initializing QueryService: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str

@router.post("/", summary="Submit a query and get the generated answer")
async def query_endpoint(request: QueryRequest,query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.process_basic_query(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@router.get("/get", summary="Submit a query via GET and get the generated answer")
async def query_get_endpoint(query: str,query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.process_basic_query(query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@router.post("/stream", summary="Submit a query and stream the generated response")
async def query_stream_endpoint(request: QueryRequest,query_service: QueryService = Depends(get_query_service)):
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
    query_service: QueryService = Depends(get_query_service)):
    try:
        logger.info(f"Received request: query='{query}', conversation_id={conversation_id}")
        
        # Process the query
        try:
            streaming_llm, messages, conversation_id = await query_service.process_streaming_query(query, conversation_id)
            logger.info(f"Query processed successfully, streaming response for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error in process_streaming_query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
        
        full_response = ""
        
        async def token_generator():
            nonlocal full_response
            
            try:
                logger.info(f"Starting token streaming for conversation {conversation_id}")
                token_count = 0
                
                async for chunk in streaming_llm.astream(messages):
                    content = chunk.content
                    
                    # Skip special tokens or empty content
                    if not content or "<think>" in content or "###" in content:
                        continue
                    
                    # Add to full response and yield
                    full_response += content
                    token_count += 1
                    if token_count % 50 == 0:
                        logger.debug(f"Streamed {token_count} tokens for conversation {conversation_id}")
                    
                    yield content

                    await asyncio.sleep(0.01)
                
                logger.info(f"Completed streaming {token_count} tokens for conversation {conversation_id}")
                
                # # After streaming is complete, save the response to history
                # if hasattr(query_service, 'conversation_histories') and full_response:
                #     if conversation_id in query_service.conversation_histories:
                #         #query_service.conversation_histories[conversation_id].append(
                #         #    AIMessage(content=full_response)
                #         pass
                #         #)
                #         max_history = 10
                #         if len(query_service.conversation_histories[conversation_id]) > max_history:
                #             query_service.conversation_histories[conversation_id] = \
                #                 query_service.conversation_histories[conversation_id][-max_history:]
                    
                    # For backward compatibility
                if hasattr(query_service, 'message_history'):
                    query_service.message_history.append(AIMessage(content=full_response))
            
            except Exception as e:
                logger.error(f"Error during token streaming: {e}", exc_info=True)
                # Return an error message to the client
                yield f"\n\n오류가 발생했습니다: {str(e)}\n"
        
        # Return a streaming response
        return StreamingResponse(
            token_generator(), 
            media_type="text/plain",
            headers={"X-Conversation-ID": conversation_id}
        )
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing streaming query: {e}")

@router.get("/vectorSimilaritySearch", summary="Perform similarity search within a specific status code")
async def similarity_search_by_vector(query: str, status_code: str,query_service: QueryService = Depends(get_query_service)):
    try:
        return await query_service.search_by_vector(query, status_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {e}")

@router.delete("/resetChromaCollection", summary="Reset the ChromaDB collection")
async def reset_collection():
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

        # Update global state with all required components
        set_globals(
            chroma_coll=chroma_coll,
            rag=get_rag_chain(),  # Preserve existing RAG chain
            vect_store=chroma_coll,  # Update vector_store to new collection
            prompt=get_global_prompt(),  # Preserve existing prompt
            workflow=get_workflow(), # Preserve existing workflow
            memory=get_memory() # Preserve existing memory
        )
        logger.debug("Updated global state after reset")

        return {"message": "Collection 'netbackup_docs' has been deleted and reinitialized."}
    except Exception as e:
        logger.error(f"Error resetting collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error resetting collection: {e}")


"""
A direct implementation to bypass batch processing if you continue to have issues.
"""

# Dependency for QueryService
def get_query_service():
    from src.core.services.query_service import QueryService
    from src.core.processing.local_translator import LocalMarianTranslator
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

@router.get("/direct-stream", summary="Direct streaming without batch processing")
async def direct_stream_endpoint(
    query: str, 
    conversation_id: Optional[str] = None, 
    query_service: QueryService = Depends(get_query_service)):
    try:
        logger.info(f"Received direct stream request: query='{query}', conversation_id={conversation_id}")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Initialize or retrieve conversation history
        if not hasattr(query_service, 'conversation_histories'):
            query_service.conversation_histories = {}
            
        if conversation_id not in query_service.conversation_histories:
            query_service.conversation_histories[conversation_id] = []
            
        conversation_history = query_service.conversation_histories.get(conversation_id, [])

        # Get context from vector store
        docs = query_service.vector_store.similarity_search(query)
        
        # Use document context (skipping web search for simplicity)
        context = "\n".join(doc.page_content[:600] for doc in docs)

        # Include conversation history
        history_text = ""
        if conversation_history:
            history_pairs = []
            for i in range(0, len(conversation_history), 2):
                if i+1 < len(conversation_history):
                    user_msg = conversation_history[i].content
                    assistant_msg = conversation_history[i+1].content
                    history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
            
            history_text = "\n\n".join(history_pairs)
        
        # Prepare the system instruction and query with context
        korean_instruction = "당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 명확하게 답변하세요. 기술 용어만 영어로 유지하세요."
        query_with_context = f"대화 기록:\n{history_text}\n\n문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:"
        
        # Create message format
        messages = [
            SystemMessage(content=korean_instruction),
            HumanMessage(content=query_with_context)
        ]
        
        # Add user query to conversation history
        query_service.conversation_histories[conversation_id].append(HumanMessage(content=query))
        
        # Set up streaming
        token_handler = AsyncTokenStreamHandler()
        
        # Create a streaming LLM directly (no batch processing)
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
                    
                    # Skip special tokens or empty content
                    if not content or "<think>" in content or "###" in content:
                        continue
                    
                    # Add to full response and yield
                    full_response += content
                    token_count += 1
                    
                    yield content
                
                logger.info(f"Completed direct streaming {token_count} tokens")
                
                # After streaming is complete, save the response to history
                if full_response:
                    query_service.conversation_histories[conversation_id].append(
                        AIMessage(content=full_response)
                    )
            
            except Exception as e:
                logger.error(f"Error during direct token streaming: {e}", exc_info=True)
                yield f"\n\n오류가 발생했습니다: {str(e)}\n"
        
        # Return a streaming response
        return StreamingResponse(
            token_generator(), 
            media_type="text/plain",
            headers={"X-Conversation-ID": conversation_id}
        )
    
    except Exception as e:
        logger.error(f"Direct streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with direct streaming: {str(e)}")