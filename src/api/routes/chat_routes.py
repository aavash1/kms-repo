# src/api/routes/chat_routes.py 
from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.services.chat_history_manager import ChatHistoryManager
from src.core.startup import get_components
import logging
import uuid
import asyncio
from src.core.auth.auth_middleware import verify_api_key_and_member_id

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

# Enhanced dependency - file-based only, no database
def get_chat_history_manager():
    """Get chat history manager without database dependencies."""
    components = get_components()
    if 'chat_history_manager' not in components:
        # Get batch manager for title generation if available
        batch_manager = None
        if 'query_service' in components:
            batch_manager = components['query_service'].batch_manager
        
        # Initialize with file-based storage only (no database)
        chat_manager = ChatHistoryManager(
            batch_manager=batch_manager,
            db_connector=None  # Force file-based storage
        )
        components['chat_history_manager'] = chat_manager
        logger.info("Initialized file-based ChatHistoryManager in chat routes")
    
    return components['chat_history_manager']

@router.get("/chats", response_model=List[ChatHistoryItem])
async def get_all_chats(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get all chat histories for the current member (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        logger.info(f"Getting chats for member: {member_id}")
        
        # Try async first, fallback to sync if needed
        try:
            chats = await chat_manager.get_chats_by_member(member_id)
        except Exception as async_error:
            logger.warning(f"Async get_chats failed, trying sync fallback: {async_error}")
            try:
                chats = chat_manager.get_chats_by_member_sync(member_id)
            except Exception as sync_error:
                logger.error(f"Both async and sync methods failed: {sync_error}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve chats: {sync_error}")
        
        # Convert to the expected format
        chat_items = []
        for chat in chats:
            # Handle both summary and full chat formats
            if isinstance(chat, dict):
                chat_items.append(ChatHistoryItem(
                    id=chat.get("id", ""),
                    title=chat.get("title", "Untitled Chat"),
                    created_at=chat.get("created_at", ""),
                    updated_at=chat.get("updated_at", "")
                ))
        
        logger.info(f"Retrieved {len(chat_items)} chats for member {member_id}")
        return chat_items
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve chat histories for member: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat histories: {str(e)}")

@router.get("/chats/{conversation_id}", response_model=ChatDetail)
async def get_chat_by_id(
    conversation_id: str,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get a specific chat by ID and member_id (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        logger.info(f"Getting chat {conversation_id} for member {member_id}")
        
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
        
        if not chat:
            logger.warning(f"Chat {conversation_id} not found for member {member_id}")
            raise HTTPException(status_code=404, detail=f"Chat with ID {conversation_id} not found")
        
        # Convert messages to the expected format
        messages = []
        for msg in chat.get("messages", []):
            if isinstance(msg, dict):
                # Handle different message formats
                role = msg.get("role", "user")
                if "type" in msg:
                    # Convert from type format (HumanMessage -> user)
                    if "human" in msg["type"].lower():
                        role = "user"
                    elif "ai" in msg["type"].lower():
                        role = "assistant"
                
                messages.append({
                    "type": role,
                    "content": msg.get("content", "")
                })
        
        logger.info(f"Retrieved chat {conversation_id} with {len(messages)} messages")
        return ChatDetail(
            id=chat["id"],
            title=chat.get("title", "Untitled Chat"),
            created_at=chat.get("created_at", ""),
            updated_at=chat.get("updated_at", ""),
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat: {str(e)}")

@router.post("/chats", response_model=ChatDetail)
async def save_chat(
    request: SaveChatRequest = Body(...),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Save a chat session using member_id (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        
        # Generate a new conversation_id if not provided
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation ID: {conversation_id}")
        
        logger.info(f"Saving chat {conversation_id} for member {member_id}")
        
        # Convert messages to the format expected by chat manager
        messages = []
        for msg in request.messages:
            if isinstance(msg, dict):
                # Ensure consistent format
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Convert to the internal format the chat manager expects
                message_type = "HumanMessage" if role == "user" else "AIMessage"
                messages.append({
                    "type": message_type,
                    "content": content
                })
        
        # Generate title if not provided
        title = request.title
        if not title and messages:
            first_user_msg = next(
                (msg["content"] for msg in messages if "human" in msg.get("type", "").lower()),
                None
            )
            if first_user_msg:
                title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
            else:
                title = "New Chat"
        
        # Save the chat using member_id (file-based)
        await chat_manager.save_chat_by_member(
            conversation_id=conversation_id, 
            messages=messages,
            member_id=member_id,
            chat_title=title
        )
        
        # Add a small delay to ensure write completion
        await asyncio.sleep(0.1)
        
        # Retrieve the saved chat to return
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
        if not chat:
            logger.error(f"Failed to retrieve chat after saving: {conversation_id}")
            raise HTTPException(status_code=500, detail="Failed to save chat")
        
        # Convert back to response format
        response_messages = []
        for msg in chat.get("messages", []):
            role = "user" if "human" in msg.get("type", "").lower() else "assistant"
            response_messages.append({
                "type": role,
                "content": msg.get("content", "")
            })
        
        logger.info(f"Successfully saved chat {conversation_id} for member {member_id}")
        return ChatDetail(
            id=conversation_id,
            title=title,
            created_at=chat.get("created_at", ""),
            updated_at=chat.get("updated_at", ""),
            messages=response_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

@router.delete("/chats/{conversation_id}", response_model=DeleteChatResponse)
async def delete_chat(
    conversation_id: str,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Delete a chat by ID using member_id (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        logger.info(f"Deleting chat {conversation_id} for member {member_id}")
        
        # Delete using member_id verification (file-based)
        success = await chat_manager.delete_chat_by_member(conversation_id, member_id)
        if success:
            logger.info(f"Chat {conversation_id} deleted successfully for member {member_id}")
            return DeleteChatResponse(
                success=True, 
                message=f"Chat {conversation_id} deleted successfully"
            )
        else:
            logger.warning(f"Failed to delete chat {conversation_id} for member {member_id}")
            return DeleteChatResponse(
                success=False, 
                message=f"Failed to delete chat {conversation_id}"
            )
    except Exception as e:
        logger.error(f"Error deleting chat {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

@router.put("/chats/{conversation_id}/title", response_model=ChatDetail)
async def update_chat_title(
    conversation_id: str,
    title: str = Body(..., embed=True),
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Update the title of a specific chat using member_id (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        logger.info(f"Updating title for chat {conversation_id} to '{title}' for member {member_id}")
        
        # Get the current chat to verify ownership
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
        if not chat:
            logger.warning(f"Chat with ID {conversation_id} not found for member {member_id}")
            raise HTTPException(status_code=404, detail=f"Chat with ID {conversation_id} not found")
        
        # Save the chat with updated title using member_id (file-based)
        await chat_manager.save_chat_by_member(
            conversation_id=conversation_id, 
            messages=chat["messages"],
            member_id=member_id,
            chat_title=title
        )
        
        # Retrieve the updated chat to return
        updated_chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
        
        # Convert messages to response format
        response_messages = []
        for msg in updated_chat.get("messages", []):
            role = "user" if "human" in msg.get("type", "").lower() else "assistant"
            response_messages.append({
                "type": role,
                "content": msg.get("content", "")
            })
        
        logger.info(f"Updated title for chat {conversation_id} to '{title}' for member {member_id}")
        return ChatDetail(
            id=conversation_id,
            title=title,
            created_at=updated_chat.get("created_at", ""),
            updated_at=updated_chat.get("updated_at", ""),
            messages=response_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat title: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update chat title: {str(e)}")

# Simplified analytics endpoint - file-based stats only
@router.get("/analytics/user")
async def get_user_chat_analytics(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Get basic chat analytics for the current user (file-based only)."""
    try:
        member_id = auth_data["member_id"]
        logger.info(f"Getting analytics for member {member_id}")
        
        # Get all chats for this member - try async first, fallback to sync
        try:
            chats = await chat_manager.get_chats_by_member(member_id)
        except Exception as async_error:
            logger.warning(f"Async analytics failed, trying sync: {async_error}")
            chats = chat_manager.get_chats_by_member_sync(member_id)
        
        # Calculate basic stats from file-based data
        total_conversations = len(chats)
        total_messages = 0
        
        # Count total messages across all chats
        for chat_summary in chats:
            try:
                # Get full chat details to count messages
                full_chat = await chat_manager.get_chat_by_member(chat_summary["id"], member_id)
                if full_chat and "messages" in full_chat:
                    total_messages += len(full_chat["messages"])
            except Exception as e:
                logger.warning(f"Error counting messages for chat {chat_summary['id']}: {e}")
        
        return {
            "member_id": member_id,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "storage_type": "file_based",
            "note": "Analytics are based on file storage. Database analytics not available."
        }
        
    except Exception as e:
        logger.error(f"Failed to get user analytics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# Health check endpoint
@router.get("/health")
async def chat_health_check():
    """Health check endpoint for chat service."""
    try:
        chat_manager = get_chat_history_manager()
        
        # Check if storage directory exists
        import os
        storage_exists = os.path.exists(chat_manager.storage_dir)
        
        # Count files in storage
        file_count = 0
        if storage_exists:
            try:
                files = os.listdir(chat_manager.storage_dir)
                file_count = len([f for f in files if f.endswith('.json')])
            except Exception as e:
                logger.warning(f"Could not count files in storage: {e}")
        
        return {
            "status": "healthy",
            "storage_type": "file-based",
            "storage_directory": chat_manager.storage_dir,
            "storage_exists": storage_exists,
            "chat_files_count": file_count,
            "timestamp": "2025-06-09T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-06-09T12:00:00Z"
        }

# Debug endpoint - only enable in development
@router.get("/debug/files")
async def debug_chat_files(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager)
):
    """Debug endpoint to list all chat files for a member."""
    try:
        member_id = auth_data["member_id"]
        
        import os
        files_info = []
        
        if os.path.exists(chat_manager.storage_dir):
            for filename in os.listdir(chat_manager.storage_dir):
                if filename.startswith(f"{member_id}_") and filename.endswith(".json"):
                    file_path = os.path.join(chat_manager.storage_dir, filename)
                    try:
                        stat = os.stat(file_path)
                        files_info.append({
                            "filename": filename,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "readable": os.access(file_path, os.R_OK)
                        })
                    except Exception as e:
                        files_info.append({
                            "filename": filename,
                            "error": str(e)
                        })
        
        return {
            "member_id": member_id,
            "storage_directory": chat_manager.storage_dir,
            "files": files_info
        }
        
    except Exception as e:
        logger.error(f"Debug files failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")