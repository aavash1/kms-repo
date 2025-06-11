# src/api/routes/query.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, Response, Query, Body, Request, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.core.services.query_service import QueryService
from src.core.services.ingest_service import IngestService

from src.core.services.file_utils import (CHROMA_DIR, set_globals, get_vector_store, get_rag_chain, get_global_prompt, get_workflow, get_memory)
from src.core.processing.local_translator import LocalMarianTranslator
from src.core.auth.auth_middleware import verify_api_key_and_optional_session, optional_session, verify_api_key, verify_api_key_and_member_id
from src.core.startup import get_postgresql_connector
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage 
import logging
logger = logging.getLogger(__name__)
router = APIRouter()
import asyncio
from typing import Optional, List, Dict
import uuid
from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import numpy as np
import secrets
from datetime import datetime, timedelta
import time
from src.core.startup import get_components
import re
import json

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

# @router.get("/stream-get", summary="Submit a GET query and stream the generated response")
# async def query_stream_get_endpoint(
#     query: str, 
#     conversation_id: Optional[str] = None,
#     plain_text: bool = False,
#     filter_document_id: Optional[str] = Query(
#         default=None,
#         description="Restrict search to a single uploaded file id"),
#     query_service: QueryService = Depends(get_query_service)):
#     """
#     Streams chat completion.  
#     *If **filter_document_id** is supplied the answer is generated only
#     from vectors that belong to that document (private chat-with-file).*
#     """
#     try:
#         #logger.info(f"Received request: query='{query}', conversation_id={conversation_id}")
#         logger.info("GET /stream: q=%s cid=%s doc=%s",
#                     query, conversation_id, filter_document_id)
        
#         # Process the query with RL enhancement
#         streaming_llm, messages, conversation_id = await query_service.process_streaming_query(query, conversation_id,plain_text=plain_text, filter_document_id=filter_document_id)
#         logger.info(f"Query processed successfully, streaming response for conversation {conversation_id}")
        
#         full_response = ""
#         #embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
#         #query_embedding = np.array(embedding_model.embed_query(query))
#         #chunks = query_service.get_relevant_chunks(query)
        
#         async def token_generator():
#             nonlocal full_response
#             buffer= ""
#             try:
#                 logger.info(f"Starting token streaming for conversation {conversation_id}")       
#                 async for chunk in streaming_llm.astream(messages):
#                     content = chunk.content
#                     if not content or "<think>" in content or "###" in content:
#                         continue
                    
#                     full_response += content
#                     buffer += content
                    
#                     # Split buffer at newlines to preserve Markdown structure
#                     while '\n' in buffer:
#                         # Find the first newline
#                         newline_index = buffer.index('\n')
#                         # Yield everything up to and including the newline
#                         yield buffer[:newline_index + 1]
#                         # Keep the rest in the buffer
#                         buffer = buffer[newline_index + 1:]
#                         await asyncio.sleep(0.02)  # Small delay for smooth streaming
                    
#                     # If buffer is too long but no newline, yield it to avoid delays
#                     if len(buffer) >= 100:
#                         yield buffer
#                         buffer = ""
#                         await asyncio.sleep(0.02)
                
#                 # Yield any remaining buffer content
#                 if buffer:
#                     yield buffer
                    
#                 logger.info(f"Completed streaming response")
            
#             except Exception as e:
#                 logger.error(f"Error during token streaming: {e}", exc_info=True)
#                 yield f"\n\n오류가 발생했습니다: {str(e)}\n"
        
#         return StreamingResponse(
#             token_generator(), 
#             media_type="text/markdown",  # Changed to support Markdown rendering
#             headers={"X-Conversation-ID": conversation_id}
#         )
    
#     except ValueError as e:
#         logger.error(f"Value error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Streaming error: {e}", exc_info=True)
#         async def error_generator():
#             yield "오류가 발생했습니다: 벡터 데이터베이스가 비어 있거나 관련 정보가 없을 수 있습니다."
#         return StreamingResponse(
#             error_generator(),
#             media_type="text/plain",
#             headers={"X-Conversation-ID": conversation_id or str(uuid.uuid4())}
#         )

@router.get("/stream-get", summary="Submit a GET query and stream the generated response")
async def query_stream_get_endpoint(
    query: str, 
    conversation_id: Optional[str] = None,
    plain_text: bool = False,
    filter_document_id: Optional[str] = Query(
        default=None,
        description="Restrict search to a single uploaded file id"),
    include_member_docs: bool = Query(
        default=True,
        description="Include user's personal documents in search"),
    include_shared_docs: bool = Query(
        default=True, 
        description="Include shared company documents in search"),
    response_format: str = Query(
        default="stream",
        description="Response format: 'stream' for streaming text, 'json' for structured JSON response"),
    include_sources: bool = Query(
        default=False,
        description="Include source documents in JSON response (only for response_format=json)"),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service)):
    """
    Streams chat completion with member-based conversation management.
    SpringBoot controls sessions, FastAPI uses member_id for chat history.
    
    Supports two response formats:
    - stream: Original streaming markdown response (default)
    - json: Structured JSON response with metadata and optional sources
    """
    try:
        start_time = time.time()
        
        # Extract member_id from auth_data
        member_id = auth_data["member_id"]
        user_role = auth_data.get("user_role", "user")
        
        logger.info("GET /stream: q=%s cid=%s doc=%s member=%s format=%s",
                    query, conversation_id, filter_document_id, member_id, response_format)
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Get conversation history based on member_id
        conversation_history = []
        try:
            # Get chat history manager from components
            components = get_components()
            chat_manager = components.get('chat_history_manager')
            
            if chat_manager:
                # Get existing conversation by member_id instead of session_id
                existing_chat = await chat_manager.get_chat_by_member(
                    conversation_id, member_id
                )
                if existing_chat and existing_chat.get("messages"):
                    # Convert to message objects for query service
                    conversation_history = chat_manager.deserialize_messages(existing_chat["messages"])
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            conversation_history = []
        
        # For JSON format, we need to collect the full response first
        if response_format == "json":
            try:
                # Try member-aware processing first
                streaming_llm, messages, conversation_id = await query_service.process_streaming_query_with_member_context(
                    query=query, 
                    conversation_id=conversation_id,
                    member_id=member_id,
                    user_role=user_role,
                    plain_text=plain_text, 
                    filter_document_id=filter_document_id,
                    include_member_docs=include_member_docs,
                    include_shared_docs=include_shared_docs
                )
            except Exception as e:
                logger.warning(f"Member-aware processing failed, falling back to original: {e}")
                # Fallback to original method
                streaming_llm, messages, conversation_id = await query_service.process_streaming_query(
                    query=query, 
                    conversation_id=conversation_id,
                    plain_text=plain_text, 
                    filter_document_id=filter_document_id
                )
            
            # Collect full response for JSON format
            full_response = ""
            sources = []
            
            try:
                async for chunk in streaming_llm.astream(messages):
                    content = chunk.content
                    if not content or "<think>" in content or "###" in content:
                        continue
                    full_response += content
                
                # Get sources if requested
                if include_sources:
                    try:
                        # Try to get sources from vector store
                        docs = query_service.vector_store.similarity_search(query, k=5)
                        sources = [
                            {
                                "document_id": doc.metadata.get("document_id", "unknown"),
                                "document_type": doc.metadata.get("document_type", "unknown"),
                                "source": doc.metadata.get("source", "unknown"),
                                "relevance_score": 1.0,  # ChromaDB doesn't return scores directly
                                "chunk_content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                "metadata": {k: v for k, v in doc.metadata.items() if k not in ["document_id", "document_type", "source"]}
                            }
                            for doc in docs[:3]  # Limit to top 3 sources
                        ]
                    except Exception as e:
                        logger.warning(f"Failed to get sources: {e}")
                        sources = []
                
                # Save conversation history
                processing_time = int((time.time() - start_time) * 1000)
                
                try:
                    components = get_components()
                    chat_manager = components.get('chat_history_manager')
                    
                    if chat_manager:
                        # Add the new Q&A to conversation history
                        updated_messages = conversation_history + [
                            HumanMessage(content=query),
                            AIMessage(content=full_response)
                        ]
                        
                        # Save complete conversation using member_id
                        await chat_manager.save_chat_by_member(
                            conversation_id=conversation_id,
                            messages=updated_messages,
                            member_id=member_id
                        )
                        
                        # Save analytics record with member context
                        await chat_manager.save_qa_interaction_by_member(
                            member_id=member_id,
                            conversation_id=conversation_id,
                            question=query,
                            response=full_response,
                            model_used="gemma3:12b",
                            response_time_ms=processing_time,
                            metadata={
                                "filter_document_id": filter_document_id,
                                "plain_text": plain_text,
                                "user_role": user_role,
                                "include_member_docs": include_member_docs,
                                "include_shared_docs": include_shared_docs,
                                "response_format": response_format
                            }
                        )
                        
                        logger.info(f"Saved conversation and interaction for member {member_id}")
                except Exception as e:
                    logger.error(f"Failed to save conversation: {e}")
                
                # Return structured JSON response
                json_response = {
                    "status": "success",
                    "data": {
                        "response": full_response,
                        "conversation_id": conversation_id,
                        "member_id": member_id,
                        "user_role": user_role,
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "query_metadata": {
                            "original_query": query,
                            "plain_text": plain_text,
                            "filter_document_id": filter_document_id,
                            "include_member_docs": include_member_docs,
                            "include_shared_docs": include_shared_docs
                        },
                        "conversation_context": {
                            "message_count": len(conversation_history),
                            "has_history": len(conversation_history) > 0
                        }
                    }
                }
                
                # Add sources if requested
                if include_sources and sources:
                    json_response["data"]["sources"] = sources
                
                return json_response
                
            except Exception as e:
                logger.error(f"Error during JSON response generation: {e}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"오류가 발생했습니다: {str(e)}",
                    "conversation_id": conversation_id,
                    "member_id": member_id
                }
        
        else:
            # Original streaming implementation (unchanged)
            try:
                streaming_llm, messages, conversation_id = await query_service.process_streaming_query_with_member_context(
                    query=query, 
                    conversation_id=conversation_id,
                    member_id=member_id,
                    user_role=user_role,
                    plain_text=plain_text, 
                    filter_document_id=filter_document_id,
                    include_member_docs=include_member_docs,
                    include_shared_docs=include_shared_docs
                )
            except Exception as e:
                logger.warning(f"Member-aware processing failed, falling back to original: {e}")
                # Fallback to original method
                streaming_llm, messages, conversation_id = await query_service.process_streaming_query(
                    query=query, 
                    conversation_id=conversation_id,
                    plain_text=plain_text, 
                    filter_document_id=filter_document_id
                )
            
            logger.info(f"Query processed successfully for member {member_id}, streaming response for conversation {conversation_id}")
            
            full_response = ""
            
            async def token_generator():
                nonlocal full_response
                buffer = ""
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
                            newline_index = buffer.index('\n')
                            yield buffer[:newline_index + 1]
                            buffer = buffer[newline_index + 1:]
                            await asyncio.sleep(0.02)
                        
                        if len(buffer) >= 100:
                            yield buffer
                            buffer = ""
                            await asyncio.sleep(0.02)
                    
                    if buffer:
                        yield buffer
                    
                    # Save the complete conversation with member_id (unchanged)
                    try:
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        components = get_components()
                        chat_manager = components.get('chat_history_manager')
                        
                        if chat_manager:
                            # Add the new Q&A to conversation history
                            updated_messages = conversation_history + [
                                HumanMessage(content=query),
                                AIMessage(content=full_response)
                            ]
                            
                            # Save complete conversation using member_id
                            await chat_manager.save_chat_by_member(
                                conversation_id=conversation_id,
                                messages=updated_messages,
                                member_id=member_id
                            )
                            
                            # Save analytics record with member context
                            await chat_manager.save_qa_interaction_by_member(
                                member_id=member_id,
                                conversation_id=conversation_id,
                                question=query,
                                response=full_response,
                                model_used="gemma3:12b",
                                response_time_ms=processing_time,
                                metadata={
                                    "filter_document_id": filter_document_id,
                                    "plain_text": plain_text,
                                    "user_role": user_role,
                                    "include_member_docs": include_member_docs,
                                    "include_shared_docs": include_shared_docs,
                                    "response_format": response_format
                                }
                            )
                            
                            logger.info(f"Saved conversation and interaction for member {member_id}")
                    except Exception as e:
                        logger.error(f"Failed to save conversation: {e}")
                    
                    logger.info(f"Completed streaming response")
                
                except Exception as e:
                    logger.error(f"Error during token streaming: {e}", exc_info=True)
                    yield f"\n\n오류가 발생했습니다: {str(e)}\n"
            
            # Build response headers with member context
            headers = {
                "X-Conversation-ID": conversation_id,
                "X-Member-ID": member_id,
                "X-User-Role": user_role
            }
            
            return StreamingResponse(
                token_generator(), 
                media_type="text/markdown",
                headers=headers
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

@router.post("/stream-get-upload")
async def chat_with_file_upload(
    query: str = Form(...),
    file_urls: Optional[List[str]] = Form(default=None),
    document_type: Optional[str] = Form(default=None),
    conversation_id: Optional[str] = Form(default=None),
    document_id: Optional[str] = Form(default=None),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service),
    request: Request = None
):
    """
    Enhanced chat API with conversation context and local chat history persistence.
    
    Behavior:
    1. Load existing conversation context if conversation_id provided
    2. Process files if provided  
    3. Execute query with full conversation context
    4. Save the Q&A exchange to local chat history
    5. Return JSON response for MongoDB
    """
    try:
        member_id = auth_data["member_id"]
        timestamp = datetime.utcnow()
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Generate simple title from query
        title = query[:50] + "..." if len(query) > 50 else query
        
        # Handle file_urls
        if file_urls is None:
            file_urls = []
        
        # ===== NEW: Load existing conversation for context =====
        try:
            components = get_components()
            chat_manager = components.get('chat_history_manager')
            
            # Try to get existing conversation by member_id
            existing_chat = None
            if chat_manager and conversation_id:
                existing_chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
                if existing_chat and existing_chat.get("messages"):
                    # Update title from existing chat if this is a continuation
                    if existing_chat.get("title") and not title.endswith("..."):
                        title = existing_chat["title"]
                    logger.info(f"Loaded existing conversation {conversation_id} with {len(existing_chat['messages'])} messages")
        except Exception as e:
            logger.warning(f"Could not load existing conversation {conversation_id}: {e}")
            existing_chat = None
        # ===== END NEW SECTION =====
        
        # Step 1: Process files if provided (UNCHANGED)
        if file_urls and len(file_urls) > 0:
            logger.info(f"Processing {len(file_urls)} files for instant chat with member {member_id}")
            
            from src.api.routes.ingest import get_ingest_service_with_postgres
            ingest_service = get_ingest_service_with_postgres(request)
            
            if not document_id:
                date_str = timestamp.strftime("%Y%m%d")
                document_id = f"INSTANT_{date_str}_{conversation_id[:8]}_{member_id}"
            
            final_document_type = await get_validated_document_type(document_type, ingest_service)
            
            resolve_data = {
                "document_id": document_id,
                "document_type": final_document_type,
                "content": "",
                "tags": ["instant", "upload", "chat"],
                "custom_metadata": {
                    "member_id": member_id,
                    "conversation_id": conversation_id,
                    "upload_timestamp": timestamp.isoformat(),
                    "chat_type": "instant_upload"
                }
            }
            
            try:
                ingest_result = await ingest_service.process_direct_uploads_with_urls(
                    resolve_data=json.dumps(resolve_data),
                    file_urls=file_urls
                )
                logger.info(f"File processing result: {ingest_result.get('status')}")
            except Exception as e:
                logger.error(f"Failed to process files: {e}")
                document_id = None
        else:
            logger.info(f"No files provided, querying existing knowledge base for member {member_id}")
            document_id = None
        
        # Step 2: Execute query WITH CONTEXT (ENHANCED)
        start_time = time.time()
        
        try:
            # ===== NEW: Use member-aware query with conversation context =====
            streaming_llm, messages, _ = await query_service.process_streaming_query_with_member_context(
                query=query,
                conversation_id=conversation_id,
                member_id=member_id,
                plain_text=False,
                filter_document_id=document_id,
                include_member_docs=True,
                include_shared_docs=not bool(file_urls)
            )
            logger.info(f"Used member-aware processing with context for conversation {conversation_id}")
            # ===== END NEW SECTION =====
        except Exception as e:
            logger.warning(f"Member-aware processing failed, falling back to basic query: {e}")
            # Fallback to basic query processing
            streaming_llm, messages, _ = await query_service.process_streaming_query(
                query=query,
                conversation_id=conversation_id,
                plain_text=False,
                filter_document_id=document_id
            )
        
        # Collect full response (UNCHANGED)
        full_response = ""
        async for chunk in streaming_llm.astream(messages):
            content = chunk.content
            if content and "<think>" not in content and "###" not in content:
                full_response += content
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # ===== NEW: Save conversation to local chat history =====
        try:
            if chat_manager:
                # Prepare updated messages list
                if existing_chat and existing_chat.get("messages"):
                    # Append to existing conversation
                    updated_messages = existing_chat["messages"].copy()
                else:
                    # Start new conversation
                    updated_messages = []
                
                # Add current Q&A exchange
                updated_messages.extend([
                    {"type": "HumanMessage", "content": query},
                    {"type": "AIMessage", "content": full_response}
                ])
                
                # Save complete conversation to local storage
                await chat_manager.save_chat_by_member(
                    conversation_id=conversation_id,
                    messages=updated_messages,
                    member_id=member_id,
                    chat_title=title
                )
                
                logger.info(f"Saved conversation {conversation_id} with {len(updated_messages)} total messages to local storage")
        except Exception as e:
            logger.error(f"Failed to save to local chat history: {e}")
            # Don't fail the main response if chat history save fails
        # ===== END NEW SECTION =====
        
        # Step 3: Build response for MongoDB storage (UNCHANGED)
        response_data = {
            "title": title,
            "conversation_id": conversation_id,
            "response": full_response,
            "timestamp": timestamp.isoformat(),
            "query": query,
            
            # Additional context
            "member_id": member_id,
            "document_id": document_id,
            "file_count": len(file_urls) if file_urls else 0,
            "file_urls": file_urls if file_urls else [],
            "processing_time_ms": processing_time,
            "chat_type": "instant_upload" if file_urls else "text_only",
            "has_files": bool(file_urls and len(file_urls) > 0),
            
            # ===== NEW: Chat context metadata =====
            "is_continuation": bool(existing_chat),
            "total_messages_in_conversation": len(updated_messages) if 'updated_messages' in locals() else 2
            # ===== END NEW SECTION =====
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in chat_with_file_upload: {e}", exc_info=True)
        return {
            "title": "Error",
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "response": f"Sorry, an error occurred: {str(e)}",
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "error": True,
            "member_id": auth_data.get("member_id", "unknown")
        }

async def get_validated_document_type(backend_provided: Optional[str], ingest_service: IngestService) -> str:
    """Simple validation against PostgreSQL category table."""
    try:
        # Get valid types from database using existing method
        valid_types = await ingest_service.get_valid_document_types()
        
        # If backend provided a type, validate it
        if backend_provided and backend_provided in valid_types:
            logger.info(f"Using backend-provided document_type: {backend_provided}")
            return backend_provided
        
        # If backend type invalid or not provided, use safe default
        if backend_provided:
            logger.warning(f"Invalid document_type '{backend_provided}', using fallback")
        
        # Use first available type from database
        if valid_types:
            fallback = valid_types[0]
            logger.info(f"Using fallback document_type: {fallback}")
            return fallback
        
        # Emergency fallback
        return "memo"
        
    except Exception as e:
        logger.error(f"Error validating document_type: {e}")
        return "memo"

## 2. HELPER FUNCTIONS

def generate_auto_title(query: str, max_length: int = 50) -> str:
    """Generate automatic title from query."""
    # Clean up query
    title = query.strip()
    
    # Remove common question words
    title = re.sub(r'^(what is|what are|how do|how to|explain|tell me|can you)\s+', '', title, flags=re.IGNORECASE)
    
    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length].rsplit(' ', 1)[0] + "..."
    
    # Capitalize first letter
    title = title[0].upper() + title[1:] if title else "New Chat"
    
    return title


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


@router.post("/session/create", summary="Create user session (for Spring Boot backend)")
async def create_user_session(
    member_id: str = Body(..., embed=True),
    db_connector = Depends(get_postgresql_connector),  # Use the dependency
    api_key: str = Depends(verify_api_key)
):
    """Create a new user session. Called by Spring Boot backend after user authentication."""
    try:
        if db_connector:
            # Verify member exists
            member_query = "SELECT member_id, member_email, member_nm FROM member WHERE member_id = %s"
            member_result = db_connector.execute_query(member_query, (member_id,))
            
            if not member_result:
                raise HTTPException(status_code=404, detail=f"Member with ID {member_id} not found")
            
            member_data = dict(member_result[0])
            
            # Generate session
            session_id = str(uuid.uuid4())
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            # Deactivate existing sessions
            deactivate_query = """
                UPDATE user_sessions 
                SET is_active = false 
                WHERE member_id = %s AND is_active = true
            """
            db_connector.execute_query(deactivate_query, (member_id,))
            
            # Create new session
            insert_query = """
                INSERT INTO user_sessions 
                (session_id, member_id, member_email, member_nm, session_token, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            db_connector.execute_query(insert_query, (
                session_id, member_id, member_data["member_email"],
                member_data["member_nm"], session_token, expires_at
            ))
            
            logger.info(f"Created session {session_id} for member {member_id}")
            
            return {
                "session_id": session_id,
                "member_id": member_id,
                "expires_at": expires_at.isoformat(),
                "session_token": session_token
            }
        else:
            # No database support, return a temporary session ID for compatibility
            session_id = str(uuid.uuid4())
            logger.warning(f"No database support, created temporary session {session_id}")
            return {
                "session_id": session_id,
                "member_id": member_id,
                "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                "session_token": "temporary"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/member-search", summary="Advanced member-aware search")
async def member_contextual_search(
    query: str,
    document_types: Optional[List[str]] = Query(None, description="Filter by document types"),
    boost_member_docs: bool = Query(True, description="Boost relevance of member's own documents"),
    limit: int = Query(10, description="Number of results to return"),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service)
):
    """
    Advanced member-aware search with filtering and boosting options.
    """
    try:
        member_id = auth_data["member_id"]
        user_role = auth_data.get("user_role", "user")
        
        # Use enhanced similarity search if available, fallback to original
        try:
            result = await query_service.advanced_member_search(
                query=query,
                member_id=member_id,
                user_role=user_role,
                document_types=document_types,
                boost_member_docs=boost_member_docs,
                limit=limit
            )
        except AttributeError:
            # Fallback: use regular similarity search
            logger.warning("advanced_member_search not available, using regular search")
            result = await query_service.perform_similarity_search(query)
            result["member_id"] = member_id
            result["user_role"] = user_role
        
        return result
        
    except Exception as e:
        logger.error(f"Error in member contextual search: {e}", exc_info=True)
        raise HTTPException(500, f"Search failed: {str(e)}")


#######################################

@router.get("/documents", summary="List user's documents")
async def list_user_documents(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service),
    limit: int = Query(50, description="Maximum number of documents to return"),
    document_type: Optional[str] = Query(None, description="Filter by document type")
):
    """
    List all documents owned by the requesting member.
    Useful for frontend to show user's uploaded files.
    """
    try:
        member_id = auth_data["member_id"]
        
        # Build filter
        filter_conditions = {"member_id": member_id}
        if document_type:
            filter_conditions["document_type"] = document_type
        
        documents = {}
        
        # Search both stores
        for store_name, store in [("kb", query_service.kb_store), ("chat", query_service.chat_store)]:
            if not store:
                continue
                
            try:
                docs = store.get(
                    where=filter_conditions,
                    limit=limit * 2  # Get more to account for duplicates
                )
                
                # Group by document_id to avoid duplicates
                for i, chunk_id in enumerate(docs.get("ids", [])):
                    metadata = docs["metadatas"][i] if docs.get("metadatas") else {}
                    doc_id = metadata.get("document_id")
                    
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "document_type": metadata.get("document_type", "unknown"),
                            "upload_timestamp": metadata.get("upload_timestamp"),
                            "file_count": metadata.get("file_count", 1),
                            "filename": metadata.get("filename", "Unknown"),
                            "source_store": store_name,
                            "chunk_count": 1  # We'll count this below
                        }
                    elif doc_id:
                        documents[doc_id]["chunk_count"] += 1
                        
            except Exception as e:
                logger.warning(f"Error listing documents from {store_name} store: {e}")
        
        # Convert to list and sort by upload time
        doc_list = list(documents.values())
        doc_list.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "member_id": member_id,
            "total_documents": len(doc_list),
            "documents": doc_list[:limit]
        }
        
    except Exception as e:
        logger.error(f"Error listing documents for member {member_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/documents/{document_id}", summary="Delete a document and all its chunks")
async def delete_document(
    document_id: str,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service)
):
    """
    Delete a document and all its chunks from vector stores.
    Works with auto-generated document_ids from /ingest-documents.
    """
    try:
        member_id = auth_data["member_id"]
        user_role = auth_data.get("user_role", "user")
        
        logger.info(f"Delete request: document_id={document_id}, member_id={member_id}")
        
        # Verify ownership (enhanced for auto-generated IDs)
        if not await verify_document_ownership_enhanced(document_id, member_id, user_role, query_service):
            raise HTTPException(
                status_code=403, 
                detail="Not authorized to delete this document"
            )
        
        # Get document info before deletion
        doc_info = await get_document_info(document_id, query_service)
        
        # Delete from vector stores
        deleted_chunks = await query_service.delete_document_chunks_enhanced(document_id)
        
        if deleted_chunks > 0:
            logger.info(f"Document {document_id} deleted by member {member_id} - {deleted_chunks} chunks removed")
            return {
                "status": "success",
                "message": f"Document deleted successfully",
                "document_id": document_id,
                "deleted_by": member_id,
                "chunks_deleted": deleted_chunks,
                "document_info": doc_info
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found or already deleted"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete document: {str(e)}"
        )

@router.put("/documents/{document_id}/replace", summary="Replace document content")
async def replace_document_content(
    document_id: str,
    file_urls: List[str] = Form(...),
    document_type: Optional[str] = Form(default=None),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    query_service: QueryService = Depends(get_query_service),
    request: Request = None
):
    """
    Replace a document by deleting old chunks and uploading new files.
    Maintains the same auto-generated document_id.
    """
    try:
        member_id = auth_data["member_id"]
        user_role = auth_data.get("user_role", "user")
        
        logger.info(f"Replace request: document_id={document_id}, member_id={member_id}, files={len(file_urls)}")
        
        # Verify ownership
        if not await verify_document_ownership_enhanced(document_id, member_id, user_role, query_service):
            raise HTTPException(
                status_code=403, 
                detail="Not authorized to update this document"
            )
        
        # Get original document info
        original_info = await get_document_info(document_id, query_service)
        if not original_info:
            raise HTTPException(
                status_code=404,
                detail=f"Original document {document_id} not found"
            )
        
        # Step 1: Delete existing chunks
        deleted_chunks = await query_service.delete_document_chunks_enhanced(document_id)
        logger.info(f"Deleted {deleted_chunks} chunks for document {document_id}")
        
        # Step 2: Re-upload with same document_id
        from src.api.routes.ingest import get_ingest_service_with_postgres
        ingest_service = get_ingest_service_with_postgres(request)
        
        # Use original document_type if not provided
        final_document_type = document_type or original_info.get("document_type", "memo")
        
        # Validate document type
        try:
            valid_types = await ingest_service.get_valid_document_types()
            if final_document_type not in valid_types:
                final_document_type = valid_types[0] if valid_types else "memo"
        except Exception:
            final_document_type = "memo"
        
        # Create resolve_data for replacement (KEEP SAME document_id)
        resolve_data = {
            "document_id": document_id,  # ✅ Keep same ID - this is the key!
            "document_type": final_document_type,
            "content": "",
            "tags": ["updated", "replaced"],
            "custom_metadata": {
                "member_id": member_id,
                "uploaded_by": member_id,
                "updated_timestamp": datetime.utcnow().isoformat(),
                "operation": "replace",
                "original_file_count": original_info.get("chunk_count", 0),
                "new_file_count": len(file_urls)
            }
        }
        
        # Step 3: Process new files with same document_id
        import json
        ingest_result = await ingest_service.process_direct_uploads_with_urls(
            resolve_data=json.dumps(resolve_data),
            file_urls=file_urls
        )
        
        if ingest_result.get("status") == "success":
            return {
                "status": "success",
                "message": f"Document {document_id} replaced successfully",
                "document_id": document_id,
                "original_chunks": deleted_chunks,
                "new_file_count": len(file_urls),
                "updated_by": member_id,
                "document_type": final_document_type,
                "ingest_details": ingest_result
            }
        else:
            # If re-upload failed, we're in a bad state (document deleted but not re-created)
            logger.error(f"Failed to re-upload after deletion for document {document_id}: {ingest_result}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload replacement files. Document {document_id} may need to be re-uploaded. Error: {ingest_result.get('message')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error replacing document {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to replace document: {str(e)}"
        )

# Enhanced helper functions
async def verify_document_ownership_enhanced(
    document_id: str, 
    member_id: str, 
    user_role: str, 
    query_service: QueryService
) -> bool:
    """
    Enhanced ownership verification that works with auto-generated document_ids.
    """
    try:
        # Admin can delete anything
        if user_role in ["admin", "moderator"]:
            return True
        
        # For auto-generated IDs, check if member_id is in the document_id
        if f"_{member_id}_" in document_id or document_id.endswith(f"_{member_id}"):
            logger.debug(f"Document ownership verified by ID pattern: {document_id} contains {member_id}")
            # Still verify in database to be sure
        
        # Check both stores
        for store in [query_service.kb_store, query_service.chat_store]:
            if not store:
                continue
                
            try:
                docs = store.get(
                    where={"$and": [
                        {"document_id": document_id},
                        {"member_id": member_id}
                    ]},
                    limit=1
                )
                
                if docs.get("ids"):
                    return True
                    
            except Exception as e:
                logger.warning(f"Error checking ownership in store: {e}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error verifying document ownership: {e}")
        return False

async def get_document_info(document_id: str, query_service: QueryService) -> Optional[Dict]:
    """Get basic info about a document before deletion/update."""
    try:
        for store_name, store in [("kb", query_service.kb_store), ("chat", query_service.chat_store)]:
            if not store:
                continue
                
            docs = store.get(where={"document_id": document_id}, limit=1)
            if docs.get("ids"):
                metadata = docs["metadatas"][0] if docs.get("metadatas") else {}
                return {
                    "document_id": document_id,
                    "document_type": metadata.get("document_type"),
                    "member_id": metadata.get("member_id"),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "file_count": metadata.get("file_count"),
                    "source_store": store_name,
                    "chunk_count": len(docs["ids"]) if docs.get("ids") else 0
                }
        return None
    except Exception as e:
        logger.warning(f"Error getting document info: {e}")
        return None


########################################