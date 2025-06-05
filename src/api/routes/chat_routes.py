# src/api/routes/chat_routes.py
from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.services.chat_history_manager import ChatHistoryManager
from src.core.startup import get_components
import logging
import uuid
import asyncio
from src.core.auth.auth_middleware import verify_api_key_and_member_id  # Only use member_id auth
from src.core.startup import get_chat_history_manager_with_db

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
def get_chat_history_manager(request: Request = None):
    components = get_components()
    if 'chat_history_manager' not in components:
        # Check if we have database support
        db_connector = None
        if request and hasattr(request.app.state, 'postgresql_db'):
            db_connector = request.app.state.postgresql_db
        elif request and hasattr(request.app.state, 'db_connector'):
            # Could use MariaDB if needed, but PostgreSQL is preferred for sessions
            pass
        
        # Initialize ChatHistoryManager with optional database support
        batch_manager = None
        if 'query_service' in components:
            batch_manager = components['query_service'].batch_manager
        
        from src.core.services.chat_history_manager import ChatHistoryManager
        chat_manager = ChatHistoryManager(
            batch_manager=batch_manager,
            db_connector=db_connector  # This enables database support if available
        )
        components['chat_history_manager'] = chat_manager
    
    return components['chat_history_manager']

# ✅ FIXED - All endpoints now use verify_api_key_and_member_id consistently

@router.get("/chats", response_model=List[ChatHistoryItem])
async def get_all_chats(
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Get all chat histories for the current member."""
    try:
        member_id = auth_data["member_id"]
        chats = await chat_manager.get_all_chats_by_member(member_id=member_id)
        return chats
    except Exception as e:
        logger.error(f"Failed to retrieve chat histories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat histories: {str(e)}")

@router.get("/chats/{conversation_id}", response_model=ChatDetail)
async def get_chat_by_id(
    conversation_id: str,
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Get a specific chat by ID and member_id."""
    try:
        member_id = auth_data["member_id"]
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
        if not chat:
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
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Save a chat session using member_id."""
    try:
        member_id = auth_data["member_id"]
        
        # Generate a new conversation_id if not provided
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation ID: {conversation_id}")
        
        # Save the chat using member_id
        await chat_manager.save_chat_by_member(
            conversation_id=conversation_id, 
            messages=request.messages,
            member_id=member_id,
            chat_title=request.title
        )
        
        # Add a small delay to ensure write completion
        await asyncio.sleep(0.1)
        
        # Retrieve the saved chat to return
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)
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
    auth_data: dict = Depends(verify_api_key_and_member_id),
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Delete a chat by ID using member_id."""
    try:
        member_id = auth_data["member_id"]
        
        # Delete using member_id verification
        success = await chat_manager.delete_chat_by_member(conversation_id, member_id)
        if success:
            logger.info(f"Chat {conversation_id} deleted successfully for member {member_id}")
            return {"success": True, "message": f"Chat {conversation_id} deleted successfully"}
        else:
            logger.warning(f"Failed to delete chat {conversation_id} for member {member_id}")
            return {"success": False, "message": f"Failed to delete chat {conversation_id}"}
    except Exception as e:
        logger.error(f"Error deleting chat {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

# ✅ FIXED - Update chat title now uses member_id authentication
@router.put("/chats/{conversation_id}/title", response_model=ChatDetail)
async def update_chat_title(
    conversation_id: str,
    title: str = Body(..., embed=True),
    auth_data: dict = Depends(verify_api_key_and_member_id),  # ✅ Changed to member_id
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Update the title of a specific chat using member_id."""
    try:
        member_id = auth_data["member_id"]  # ✅ Get member_id instead of session
        
        # Get the current chat to verify ownership
        chat = await chat_manager.get_chat_by_member(conversation_id, member_id)  # ✅ Use member-based method
        if not chat:
            logger.warning(f"Chat with ID {conversation_id} not found for member {member_id}")
            raise HTTPException(status_code=404, detail=f"Chat with ID {conversation_id} not found")
        
        # Save the chat with updated title using member_id
        await chat_manager.save_chat_by_member(  # ✅ Use member-based method
            conversation_id=conversation_id, 
            messages=chat["messages"],
            member_id=member_id,  # ✅ Use member_id
            chat_title=title
        )
        
        # Retrieve the updated chat to return
        updated_chat = await chat_manager.get_chat_by_member(conversation_id, member_id)  # ✅ Use member-based method
        return updated_chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update chat title: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update chat title: {str(e)}")

# ✅ REMOVED SESSION-BASED ENDPOINTS - These are no longer needed with SpringBoot session control

# Optional: Keep analytics endpoint but make it work with member_id
@router.get("/analytics/user")
async def get_user_chat_analytics(
    auth_data: dict = Depends(verify_api_key_and_member_id),  # ✅ Changed to member_id
    chat_manager: ChatHistoryManager = Depends(get_chat_history_manager_with_db)
):
    """Get chat analytics for the current user using member_id."""
    try:
        member_id = auth_data["member_id"]  # ✅ Get member_id directly
        
        if not chat_manager.use_database:
            raise HTTPException(status_code=501, detail="Analytics require database support")
        
        db = chat_manager.db
        
        # Total conversations
        conv_query = """
            SELECT COUNT(*) as total_conversations
            FROM chat_conversations 
            WHERE member_id = %s AND is_active = true
        """
        conv_result = db.execute_query(conv_query, (member_id,))
        total_conversations = dict(conv_result[0])["total_conversations"] if conv_result else 0
        
        # Total interactions in last 30 days
        interact_query = """
            SELECT 
                COUNT(*) as total_interactions,
                AVG(response_time_ms) as avg_response_time,
                SUM(tokens_used) as total_tokens
            FROM chat_history 
            WHERE member_id = %s
            AND created_at > NOW() - INTERVAL '30 days'
        """
        interact_result = db.execute_query(interact_query, (member_id,))
        interact_stats = dict(interact_result[0]) if interact_result else {}
        
        return {
            "member_id": member_id,
            "total_conversations": total_conversations,
            "total_interactions_30d": interact_stats.get("total_interactions", 0),
            "avg_response_time_ms": interact_stats.get("avg_response_time"),
            "total_tokens_30d": interact_stats.get("total_tokens", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user analytics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")