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
    Manages persistent chat histories with automatic chat titles.
    
    Features:
    - Saves chat sessions to disk
    - Automatically generates titles for chats
    - Retrieves all chat histories for display in sidebar
    - Supports deleting chats
    """
    def __init__(self, storage_dir: str = None, batch_manager=None):
        """
        Initialize the ChatHistoryManager.
        
        Args:
            storage_dir: Directory to store chat histories.
            batch_manager: Optional BatchInferenceManager for title generation.
        """
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "chat_histories")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # In-memory cache of conversation histories
        self.active_conversations: Dict[str, List[Any]] = {}
        
        # Lock for thread-safety
        self.lock = asyncio.Lock()
        
        # For title generation
        self.batch_manager = batch_manager
        
    async def save_chat(self, conversation_id: str, messages: List[Any], 
                     chat_title: Optional[str] = None) -> str:
        """
        Save a chat session to disk.
        
        Args:
            conversation_id: Unique ID for the conversation
            messages: List of message objects
            chat_title: Custom title for the chat (generated if None)
            
        Returns:
            The conversation ID
        """
        async with self.lock:
            # Even if no messages, we still need to create a chat entry
            if not messages:
                logger.warning(f"No messages to save for conversation {conversation_id}")
                # Use an empty list instead of returning early
                serialized_messages = []
            else:
                # Convert message objects to serializable dictionaries
                serialized_messages = []
                for msg in messages:
                    if isinstance(msg, (SystemMessage, AIMessage, HumanMessage)):
                        serialized_messages.append({
                            "type": msg.__class__.__name__,
                            "content": msg.content
                        })
                    else:
                        # Handle other message types or raw dictionaries
                        serialized_messages.append(msg)
            
            # Generate a title if not provided
            title = chat_title
            if not title:
                if messages:
                    title = await self._generate_title(messages)
                else:
                    # Default title for empty chats
                    title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create chat data structure
            now = datetime.now().isoformat()
            chat_data = {
                "id": conversation_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": serialized_messages
            }
            
            # Save to file
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(chat_data, ensure_ascii=False, indent=2))
                await f.flush()  # Ensure data is written to disk
            
            # Update in-memory cache
            self.active_conversations[conversation_id] = chat_data
            
            logger.info(f"Saved chat {conversation_id} with title: {title}")
            return conversation_id

    async def get_chat(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chat by its ID.
        """
        # Check in-memory cache first
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Try to load from disk
        file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
        if not os.path.exists(file_path):
            logger.warning(f"Chat {conversation_id} not found")
            return None
        
        try:
            # Use synchronous file operations to ensure file is fully read
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.loads(f.read())
                
            # Update cache
            self.active_conversations[conversation_id] = chat_data
            return chat_data
        except Exception as e:
            logger.error(f"Error loading chat {conversation_id}: {e}")
            return None
        
    async def get_all_chats(self) -> List[Dict[str, Any]]:
            """
            Retrieve all saved chats.
            
            Returns:
                List of chat metadata (id, title, created_at, updated_at)
            """
            chats = []
            
            # Scan the storage directory for chat files
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            chat_data = json.loads(await f.read())
                        
                        # Extract metadata only (not messages)
                        chats.append({
                            "id": chat_data.get("id"),
                            "title": chat_data.get("title", "Untitled Chat"),
                            "created_at": chat_data.get("created_at"),
                            "updated_at": chat_data.get("updated_at")
                        })
                    except Exception as e:
                        logger.error(f"Error loading chat file {filename}: {e}")
            
            # Sort by updated_at (most recent first)
            chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return chats
        
    async def delete_chat(self, conversation_id: str) -> bool:
            """
            Delete a chat.
            
            Args:
                conversation_id: ID of the chat to delete
                
            Returns:
                True if successful, False otherwise
            """
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    # Remove from cache if present
                    if conversation_id in self.active_conversations:
                        del self.active_conversations[conversation_id]
                    logger.info(f"Deleted chat {conversation_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting chat {conversation_id}: {e}")
                    return False
            else:
                logger.warning(f"Chat {conversation_id} not found for deletion")
                return False
        
    async def _generate_title(self, messages: List[Any]) -> str:
            """
            Generate a title for a chat based on its content.
            
            Args:
                messages: List of message objects
                
            Returns:
                Generated title string
            """
            # Default title if generation fails
            default_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # If batch manager is not available, return default
            if not self.batch_manager:
                return default_title
            
            # Extract the first user message (or first few messages)
            first_messages = messages[:min(len(messages), 4)]
            user_messages = [msg.content for msg in first_messages 
                            if isinstance(msg, HumanMessage)]
            
            if not user_messages:
                return default_title
            
            try:
                # Build prompt to generate title
                prompt = f"""Based on these chat messages, generate a concise, descriptive title (maximum 6 words):
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
            """
            Convert message objects to a serializable format.
            
            Args:
                messages: List of message objects
                
            Returns:
                List of serialized message dictionaries
            """
            serialized = []
            for msg in messages:
                if isinstance(msg, (SystemMessage, AIMessage, HumanMessage)):
                    serialized.append({
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    })
                else:
                    # For already serialized messages or other formats
                    serialized.append(msg)
            return serialized
        
    def deserialize_messages(self, serialized_messages: List[Dict[str, Any]]) -> List[Any]:
            """
            Convert serialized messages back to message objects.
            
            Args:
                serialized_messages: List of serialized message dictionaries
                
            Returns:
                List of message objects
            """
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
                    # Handle unknown types
                    logger.warning(f"Unknown message type: {msg_type}")
                    messages.append({"type": msg_type, "content": content})
                    
            return messages