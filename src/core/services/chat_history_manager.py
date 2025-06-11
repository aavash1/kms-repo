# src/core/services/chat_history_manager.py
import os
import json
import time
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime
import uuid
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import aiofiles
import shutil

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Simplified chat history manager with file-based storage only.
    No database dependencies - all operations use member_id-based file storage.
    """
    def __init__(self, storage_dir: str = None, batch_manager=None, db_connector=None):
        """
        Initialize the ChatHistoryManager with file-based storage only.
        
        Args:
            storage_dir: Directory to store chat histories.
            batch_manager: Optional BatchInferenceManager for title generation.
            db_connector: Ignored - forcing file-based storage only.
        """
        # File-based storage only
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "chat_histories")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Force file-based storage (ignore database connector)
        self.db = None
        self.use_database = False
        
        # In-memory cache of conversation histories
        self.active_conversations: Dict[str, List[Any]] = {}
        
        # Lock for thread-safety
        self.lock = asyncio.Lock()
        
        # For title generation
        self.batch_manager = batch_manager
        
        logger.info(f"ChatHistoryManager initialized with file-based storage in: {self.storage_dir}")

    # ===== MEMBER-BASED METHODS (Primary Interface) =====
    
    async def get_chat_by_member(self, conversation_id: str, member_id: str) -> Optional[Dict[str, Any]]:
        """Get chat by conversation_id and member_id from file storage."""
        async with self.lock:
            return await self._get_chat_from_file_by_member(conversation_id, member_id)
    
    async def get_chats_by_member(self, member_id: str) -> List[Dict[str, Any]]:
        """Alias for get_all_chats_by_member - for backward compatibility with chat_routes.py"""
        return await self.get_all_chats_by_member(member_id)

    async def save_chat_by_member(self, conversation_id: str, messages: List[Any], 
                                  member_id: str, chat_title: Optional[str] = None) -> str:
        """Save chat with member_id to file storage."""
        async with self.lock:
            return await self._save_chat_to_file_by_member(conversation_id, messages, member_id, chat_title)
    
    async def get_all_chats_by_member(self, member_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a specific member_id from file storage."""
        return await self._get_all_chats_from_files_by_member(member_id)
    
    async def delete_chat_by_member(self, conversation_id: str, member_id: str) -> bool:
        """Delete a chat by conversation_id and member_id from file storage."""
        async with self.lock:
            return await self._delete_chat_from_file_by_member(conversation_id, member_id)

    # ===== FILE STORAGE IMPLEMENTATION =====
    
    async def _get_chat_from_file_by_member(self, conversation_id: str, member_id: str) -> Optional[Dict[str, Any]]:
        """Get chat from file storage, filtered by member_id."""
        file_path = os.path.join(self.storage_dir, f"{member_id}_{conversation_id}.json")
        
        # Check in-memory cache first
        cache_key = f"{member_id}_{conversation_id}"
        if cache_key in self.active_conversations:
            logger.debug(f"Retrieved chat {conversation_id} for member {member_id} from cache")
            return self.active_conversations[cache_key]
        
        if not os.path.exists(file_path):
            logger.debug(f"Chat file not found: {file_path}")
            return None
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                chat_data = json.loads(content)
                
            # Verify the member_id matches for security
            if chat_data.get("member_id") != member_id:
                logger.warning(f"Member ID mismatch for conversation {conversation_id}")
                return None
                
            # Update cache
            self.active_conversations[cache_key] = chat_data
            logger.debug(f"Loaded chat {conversation_id} for member {member_id} from file")
            return chat_data
            
        except Exception as e:
            logger.error(f"Error loading chat {conversation_id} for member {member_id}: {e}")
            return None
    
    async def _save_chat_to_file_by_member(self, conversation_id: str, messages: List[Any], 
                                           member_id: str, chat_title: Optional[str] = None) -> str:
        """Save chat to file storage with member_id."""
        try:
            # Convert message objects to serializable dictionaries
            if not messages:
                logger.debug(f"No messages to save for conversation {conversation_id}")
                serialized_messages = []
            else:
                serialized_messages = []
                for msg in messages:
                    if isinstance(msg, (SystemMessage, AIMessage, HumanMessage)):
                        serialized_messages.append({
                            "type": msg.__class__.__name__,
                            "content": msg.content
                        })
                    elif isinstance(msg, dict):
                        # Already serialized
                        serialized_messages.append(msg)
                    else:
                        # Convert to dict
                        serialized_messages.append({
                            "type": "Unknown",
                            "content": str(msg)
                        })
            
            # Generate a title if not provided
            title = chat_title
            if not title:
                if messages:
                    title = await self._generate_title(messages)
                else:
                    title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create chat data structure with member_id
            now = datetime.now().isoformat()
            chat_data = {
                "id": conversation_id,
                "member_id": member_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": serialized_messages
            }
            
            # Save to file with member_id prefix
            file_path = os.path.join(self.storage_dir, f"{member_id}_{conversation_id}.json")
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(chat_data, ensure_ascii=False, indent=2))
                await f.flush()
            
            # Update in-memory cache
            cache_key = f"{member_id}_{conversation_id}"
            self.active_conversations[cache_key] = chat_data
            
            logger.info(f"Saved chat {conversation_id} to file for member {member_id} with {len(serialized_messages)} messages")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error saving chat {conversation_id} for member {member_id}: {e}")
            raise
    
    async def _get_all_chats_from_files_by_member(self, member_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a member from file storage."""
        chats = []
        prefix = f"{member_id}_"
        
        try:
            # Scan the storage directory for files with member_id prefix
            if not os.path.exists(self.storage_dir):
                logger.warning(f"Storage directory does not exist: {self.storage_dir}")
                return []
                
            for filename in os.listdir(self.storage_dir):
                if filename.startswith(prefix) and filename.endswith(".json"):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                            chat_data = json.loads(content)
                        
                        # Verify member_id matches
                        if chat_data.get("member_id") == member_id:
                            chats.append({
                                "id": chat_data.get("id"),
                                "title": chat_data.get("title", "Untitled Chat"),
                                "created_at": chat_data.get("created_at"),
                                "updated_at": chat_data.get("updated_at"),
                                "member_id": member_id
                            })
                    except Exception as e:
                        logger.error(f"Error loading chat file {filename}: {e}")
            
            # Sort by updated_at (most recent first)
            chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            logger.info(f"Found {len(chats)} chats for member {member_id}")
            return chats
            
        except Exception as e:
            logger.error(f"Error getting all chats for member {member_id}: {e}")
            return []
    
    async def _delete_chat_from_file_by_member(self, conversation_id: str, member_id: str) -> bool:
        """Delete chat from file storage using member_id."""
        file_path = os.path.join(self.storage_dir, f"{member_id}_{conversation_id}.json")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                
                # Remove from cache if present
                cache_key = f"{member_id}_{conversation_id}"
                if cache_key in self.active_conversations:
                    del self.active_conversations[cache_key]
                    
                logger.info(f"Deleted chat {conversation_id} from file for member {member_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting chat {conversation_id}: {e}")
                return False
        else:
            logger.warning(f"Chat {conversation_id} for member {member_id} not found for deletion")
            return False

    # ===== UTILITY METHODS =====
    
    async def _generate_title(self, messages: List[Any]) -> str:
        """Generate a title for a chat based on its content."""
        # Default title if generation fails
        default_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # If batch manager is not available, return default
        if not self.batch_manager:
            return default_title
        
        # Extract the first user message
        first_messages = messages[:min(len(messages), 4)]
        user_messages = []
        
        for msg in first_messages:
            if isinstance(msg, HumanMessage):
                user_messages.append(msg.content)
            elif isinstance(msg, dict) and msg.get("type") == "HumanMessage":
                user_messages.append(msg.get("content", ""))
        
        if not user_messages:
            return default_title
        
        try:
            # Build prompt to generate title
            prompt = f"""Based on this chat message, generate a concise, descriptive title (maximum 6 words):
            {user_messages[0][:200]}"""
            
            prompt_messages = [
                SystemMessage(content="You generate concise, descriptive chat titles from message content."),
                HumanMessage(content=prompt)
            ]
            
            # Use batch manager to generate title
            response_future = await self.batch_manager.submit_request(
                query="Generate chat title",
                context="",
                messages=prompt_messages,
                conversation_id=str(uuid.uuid4())  # Temporary ID for title generation
            )
            
            try:
                result = await response_future
                # Clean up the title
                title = result.content.strip().strip('"').strip("'")
                # Enforce length limit
                if len(title) > 50:
                    title = title[:50] + "..."
                logger.info(f"Generated title: {title}")
                return title
            except Exception as e:
                logger.error(f"Error in title generation: {e}")
                return default_title
                
        except Exception as e:
            logger.error(f"Failed to generate chat title: {e}")
            return default_title
    
    def serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert message objects to a serializable format."""
        serialized = []
        for msg in messages:
            if isinstance(msg, (SystemMessage, AIMessage, HumanMessage)):
                serialized.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                # Already serialized
                serialized.append(msg)
            else:
                # Convert to dict
                serialized.append({
                    "type": "Unknown",
                    "content": str(msg)
                })
        return serialized
        
    def deserialize_messages(self, serialized_messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert serialized messages back to message objects."""
        messages = []
        for msg in serialized_messages:
            msg_type = msg.get("type")
            content = msg.get("content", "")
            
            if msg_type == "SystemMessage":
                messages.append(SystemMessage(content=content))
            elif msg_type == "AIMessage":
                messages.append(AIMessage(content=content))
            elif msg_type == "HumanMessage":
                messages.append(HumanMessage(content=content))
            else:
                # Handle unknown types as dict
                messages.append({"type": msg_type, "content": content})
                
        return messages

    # ===== NO-OP DATABASE METHODS (For Compatibility) =====
    
    async def save_qa_interaction_by_member(self, member_id: str, conversation_id: str, 
                                           question: str, response: str, model_used: str = None,
                                           tokens_used: int = None, response_time_ms: int = None,
                                           metadata: Dict = None) -> Optional[str]:
        """No-op method - QA interactions not saved in file-based storage."""
        logger.debug(f"QA interaction logging skipped (file-based storage) for member {member_id}")
        return None

    # ===== LEGACY METHODS (For Backward Compatibility) =====
    
    async def save_chat(self, conversation_id: str, messages: List[Any], 
                       chat_title: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Legacy method - redirects to member-based save (requires session validation)."""
        logger.warning("save_chat called with session_id - this method is deprecated")
        # For backward compatibility, use a default member_id if session_id is provided
        member_id = session_id or "legacy_user"
        return await self.save_chat_by_member(conversation_id, messages, member_id, chat_title)
    
    async def get_chat(self, conversation_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Legacy method - redirects to member-based get."""
        logger.warning("get_chat called with session_id - this method is deprecated")
        member_id = session_id or "legacy_user"
        return await self.get_chat_by_member(conversation_id, member_id)