# src/api/routes/chat_routes.py
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.services.chat_history_manager import ChatHistoryManager
from src.core.startup import get_components
import logging
import uuid
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

# Models for request/response
class ChatHistoryItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ChatMessage(BaseModel):
    type: str
    content: str

class ChatDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]

class SaveChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="ID of the conversation. A new one will be generated if not provided.")
    messages: List[Dict[str, Any]]
    title: Optional[str] = None

class DeleteChatResponse(BaseModel):
    success: bool
    message: str

# Dependency to get the chat history manager
def get_chat_history_manager():
    components = get_components()
    if 'chat_history_manager' not in components:
        # Initialize if not already done
        batch_manager = None
        if 'query_service' in components:
            batch_manager = components['query_service'].batch_manager
        
        # Initialize a new ChatHistoryManager if not available
        components['chat_history_manager'] = ChatHistoryManager(batch_manager=batch_manager)
    
    return components['chat_history_manager']

@router.get("/chats", response_model=List[ChatHistoryItem])
async def get_all_chats(
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get all chat histories for display in sidebar."""
    try:
        chats = await chat_manager.get_all_chats()
        return chats
    except Exception as e:
        logger.error(f"Failed to retrieve chat histories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat histories: {str(e)}")

@router.get("/chats/{conversation_id}", response_model=ChatDetail)
async def get_chat_by_id(
    conversation_id: str,
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get a specific chat by ID."""
    try:
        chat = await chat_manager.get_chat(conversation_id)
        if not chat:
            logger.warning(f"Chat with ID {conversation_id} not found")
            raise HTTPException(status_code=404, detail=f"Chat with ID {conversation_id} not found")
        return chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat: {str(e)}")

@router.post("/chats", response_model=ChatDetail)
async def save_chat(
    request: SaveChatRequest = Body(...),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Save a chat session."""
    try:
        # Generate a new conversation_id if not provided
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation ID: {conversation_id}")
        
        # Save the chat
        await chat_manager.save_chat(
            conversation_id, 
            request.messages,
            request.title
        )
        
        # Add a small delay to ensure file is written (temporary fix)
        await asyncio.sleep(0.1)
        
        # Retrieve the saved chat to return
        chat = await chat_manager.get_chat(conversation_id)
        if not chat:
            logger.error(f"Failed to retrieve chat after saving: {conversation_id}")
            raise HTTPException(status_code=500, detail="Failed to save chat")
        
        return chat
    except Exception as e:
        logger.error(f"Failed to save chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

@router.delete("/chats/{conversation_id}", response_model=DeleteChatResponse)
async def delete_chat(
    conversation_id: str,
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Delete a chat by ID."""
    try:
        success = await chat_manager.delete_chat(conversation_id)
        if success:
            logger.info(f"Chat {conversation_id} deleted successfully")
            return {"success": True, "message": f"Chat {conversation_id} deleted successfully"}
        else:
            logger.warning(f"Failed to delete chat {conversation_id}, chat not found or other error")
            return {"success": False, "message": f"Failed to delete chat {conversation_id}"}
    except Exception as e:
        logger.error(f"Error deleting chat {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

@router.put("/chats/{conversation_id}/title", response_model=ChatDetail)
async def update_chat_title(
    conversation_id: str,
    title: str = Body(..., embed=True),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Update the title of a specific chat."""
    try:
        # Get the current chat
        chat = await chat_manager.get_chat(conversation_id)
        if not chat:
            logger.warning(f"Chat with ID {conversation_id} not found")
            raise HTTPException(status_code=404, detail=f"Chat with ID {conversation_id} not found")
        
        # Update the messages with the new title
        chat["title"] = title
        
        # Save the updated chat
        await chat_manager.save_chat(
            conversation_id, 
            chat["messages"],
            title
        )
        
        # Retrieve the saved chat to return
        updated_chat = await chat_manager.get_chat(conversation_id)
        return updated_chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat title: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update chat title: {str(e)}")