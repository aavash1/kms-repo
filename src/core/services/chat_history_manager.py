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
    Manages persistent chat histories with support for both file storage and PostgreSQL.
    
    Features:
    - Backward compatible with existing file-based storage
    - Optional PostgreSQL database storage for session-based users
    - Automatically generates titles for chats
    - Retrieves all chat histories for display in sidebar
    - Supports deleting chats
    - Session-aware chat management when database is available
    """
    def __init__(self, storage_dir: str = None, batch_manager=None, db_connector=None):
        """
        Initialize the ChatHistoryManager.
        
        Args:
            storage_dir: Directory to store chat histories (file-based).
            batch_manager: Optional BatchInferenceManager for title generation.
            db_connector: Optional PostgreSQL database connector for session support.
        """
        # File-based storage (existing functionality)
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "chat_histories")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Database storage (new functionality)
        self.db = db_connector
        self.use_database = db_connector is not None
        
        # In-memory cache of conversation histories
        self.active_conversations: Dict[str, List[Any]] = {}
        
        # Lock for thread-safety
        self.lock = asyncio.Lock()
        
        # For title generation
        self.batch_manager = batch_manager
        
        if self.use_database:
            logger.info("ChatHistoryManager initialized with PostgreSQL database support")
        else:
            logger.info("ChatHistoryManager initialized with file-based storage only")

    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return session data (only when database is available)."""
        if not self.use_database:
            return None
            
        try:
            query = """
                SELECT session_id, member_id, member_email, member_nm, expires_at
                FROM user_sessions 
                WHERE session_id = %s AND is_active = true AND expires_at > NOW()
            """
            result = self.db.execute_query(query, (session_id,))
            
            if result:
                # Update last_accessed
                self.db.execute_query(
                    "UPDATE user_sessions SET last_accessed = NOW() WHERE session_id = %s",
                    (session_id,)
                )
                return dict(result[0])
            return None
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None

    async def save_chat(self, conversation_id: str, messages: List[Any], 
                       chat_title: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Save a chat session to storage (file or database based on session_id).
        
        Args:
            conversation_id: Unique ID for the conversation
            messages: List of message objects
            chat_title: Custom title for the chat (generated if None)
            session_id: Optional session ID for database storage
            
        Returns:
            The conversation ID
        """
        async with self.lock:
            # If session_id is provided and database is available, use database storage
            if session_id and self.use_database:
                return await self._save_chat_to_database(session_id, conversation_id, messages, chat_title)
            else:
                # Fall back to existing file-based storage
                return await self._save_chat_to_file(conversation_id, messages, chat_title)

    async def _save_chat_to_database(self, session_id: str, conversation_id: str, 
                                   messages: List[Any], chat_title: Optional[str] = None) -> str:
        """Save chat to PostgreSQL database."""
        try:
            # Validate session first
            session_data = await self.validate_session(session_id)
            if not session_data:
                raise ValueError("Invalid or expired session")
            
            member_id = session_data["member_id"]
            
            # Convert message objects to serializable format
            if not messages:
                logger.warning(f"No messages to save for conversation {conversation_id}")
                serialized_messages = []
            else:
                serialized_messages = self.serialize_messages(messages)
            
            # Generate a title if not provided
            title = chat_title
            if not title:
                if messages:
                    title = await self._generate_title(messages)
                else:
                    title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Check if conversation already exists
            existing_query = """
                SELECT conversation_id FROM chat_conversations 
                WHERE conversation_id = %s
            """
            existing = self.db.execute_query(existing_query, (conversation_id,))
            
            if existing:
                # Update existing conversation
                update_conv_query = """
                    UPDATE chat_conversations 
                    SET title = %s, updated_at = NOW()
                    WHERE conversation_id = %s
                """
                self.db.execute_query(update_conv_query, (title, conversation_id))
                
                # Delete existing messages for this conversation
                delete_messages_query = """
                    DELETE FROM chat_messages WHERE conversation_id = %s
                """
                self.db.execute_query(delete_messages_query, (conversation_id,))
                
            else:
                # Create new conversation
                insert_conv_query = """
                    INSERT INTO chat_conversations 
                    (conversation_id, session_id, member_id, title, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """
                self.db.execute_query(insert_conv_query, (conversation_id, session_id, member_id, title))
            
            # Insert all messages
            if serialized_messages:
                for order, msg in enumerate(serialized_messages):
                    insert_msg_query = """
                        INSERT INTO chat_messages 
                        (conversation_id, message_type, content, message_order, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    metadata = json.dumps(msg.get('metadata', {})) if msg.get('metadata') else None
                    self.db.execute_query(insert_msg_query, (
                        conversation_id, 
                        msg['type'], 
                        msg['content'], 
                        order,
                        metadata
                    ))
            
            # Update in-memory cache
            self.active_conversations[conversation_id] = {
                "id": conversation_id,
                "title": title,
                "session_id": session_id,
                "member_id": member_id,
                "messages": serialized_messages
            }
            
            logger.info(f"Saved chat {conversation_id} to database with title: {title} for user {member_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error saving chat to database: {e}")
            # Fall back to file storage if database fails
            logger.warning("Falling back to file storage due to database error")
            return await self._save_chat_to_file(conversation_id, messages, chat_title)

    async def _save_chat_to_file(self, conversation_id: str, messages: List[Any], 
                                chat_title: Optional[str] = None) -> str:
        """Save chat to file storage (existing functionality)."""
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
        
        logger.info(f"Saved chat {conversation_id} to file with title: {title}")
        return conversation_id

    async def get_chat(self, conversation_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chat by its ID from database (if session_id provided) or files.
        """
        # Check in-memory cache first
        if conversation_id in self.active_conversations:
            cached_chat = self.active_conversations[conversation_id]
            # Validate session if provided
            if session_id and cached_chat.get("session_id") != session_id:
                logger.warning(f"Session mismatch for conversation {conversation_id}")
                return None
            return cached_chat
        
        # Try database first if session_id is provided and database is available
        if session_id and self.use_database:
            chat_data = await self._get_chat_from_database(conversation_id, session_id)
            if chat_data:
                return chat_data
        
        # Fall back to file storage
        return await self._get_chat_from_file(conversation_id)

    async def _get_chat_from_database(self, conversation_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get chat from PostgreSQL database."""
        try:
            # Query from database
            conv_query = """
                SELECT c.conversation_id, c.session_id, c.member_id, c.title, 
                       c.created_at, c.updated_at, c.metadata
                FROM chat_conversations c
                WHERE c.conversation_id = %s AND c.session_id = %s AND c.is_active = true
            """
            conv_result = self.db.execute_query(conv_query, (conversation_id, session_id))
            
            if not conv_result:
                return None
            
            conv_data = dict(conv_result[0])
            
            # Get messages for this conversation
            msg_query = """
                SELECT message_type, content, metadata
                FROM chat_messages 
                WHERE conversation_id = %s 
                ORDER BY message_order
            """
            msg_result = self.db.execute_query(msg_query, (conversation_id,))
            
            messages = []
            for msg_row in msg_result:
                msg_dict = dict(msg_row)
                message_data = {
                    "type": msg_dict["message_type"],
                    "content": msg_dict["content"]
                }
                if msg_dict["metadata"]:
                    message_data["metadata"] = json.loads(msg_dict["metadata"])
                messages.append(message_data)
            
            # Build chat data structure
            chat_data = {
                "id": conv_data["conversation_id"],
                "session_id": conv_data["session_id"],
                "member_id": conv_data["member_id"],
                "title": conv_data["title"],
                "created_at": conv_data["created_at"].isoformat() if conv_data["created_at"] else None,
                "updated_at": conv_data["updated_at"].isoformat() if conv_data["updated_at"] else None,
                "messages": messages
            }
            
            # Update cache
            self.active_conversations[conversation_id] = chat_data
            return chat_data
            
        except Exception as e:
            logger.error(f"Error loading chat from database {conversation_id}: {e}")
            return None

    async def _get_chat_from_file(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get chat from file storage (existing functionality)."""
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
        
    async def get_all_chats(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all saved chats from database (if session_id provided) or files.
        """
        if session_id and self.use_database:
            return await self._get_all_chats_from_database(session_id)
        else:
            return await self._get_all_chats_from_files()

    async def _get_all_chats_from_database(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a session from database."""
        try:
            # Validate session first
            session_data = await self.validate_session(session_id)
            if not session_data:
                raise ValueError("Invalid or expired session")
            
            member_id = session_data["member_id"]
            
            # Get all conversations for this user (not just current session)
            query = """
                SELECT conversation_id, title, created_at, updated_at, session_id
                FROM chat_conversations 
                WHERE member_id = %s AND is_active = true
                ORDER BY updated_at DESC
            """
            result = self.db.execute_query(query, (member_id,))
            
            chats = []
            for row in result:
                row_dict = dict(row)
                chats.append({
                    "id": row_dict["conversation_id"],
                    "title": row_dict["title"],
                    "created_at": row_dict["created_at"].isoformat() if row_dict["created_at"] else None,
                    "updated_at": row_dict["updated_at"].isoformat() if row_dict["updated_at"] else None,
                    "session_id": row_dict["session_id"]
                })
            
            return chats
            
        except Exception as e:
            logger.error(f"Error getting all chats from database for session {session_id}: {e}")
            return []

    async def _get_all_chats_from_files(self) -> List[Dict[str, Any]]:
        """Get all chats from file storage (existing functionality)."""
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
        
    async def delete_chat(self, conversation_id: str, session_id: Optional[str] = None) -> bool:
        """
        Delete a chat from database (if session_id provided) or files.
        """
        if session_id and self.use_database:
            return await self._delete_chat_from_database(conversation_id, session_id)
        else:
            return await self._delete_chat_from_file(conversation_id)

    async def _delete_chat_from_database(self, conversation_id: str, session_id: str) -> bool:
        """Delete chat from database."""
        try:
            # Validate session
            session_data = await self.validate_session(session_id)
            if not session_data:
                raise ValueError("Invalid or expired session")
            
            member_id = session_data["member_id"]
            
            # Verify chat belongs to this user
            verify_query = """
                SELECT conversation_id FROM chat_conversations 
                WHERE conversation_id = %s AND member_id = %s
            """
            verify_result = self.db.execute_query(verify_query, (conversation_id, member_id))
            
            if not verify_result:
                logger.warning(f"Chat {conversation_id} not found or doesn't belong to user {member_id}")
                return False
            
            # Soft delete (mark as inactive)
            delete_query = """
                UPDATE chat_conversations 
                SET is_active = false, updated_at = NOW()
                WHERE conversation_id = %s
            """
            self.db.execute_query(delete_query, (conversation_id,))
            
            # Remove from cache if present
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            
            logger.info(f"Deleted chat {conversation_id} from database for user {member_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chat from database {conversation_id}: {e}")
            return False

    async def _delete_chat_from_file(self, conversation_id: str) -> bool:
        """Delete chat from file storage (existing functionality)."""
        file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # Remove from cache if present
                if conversation_id in self.active_conversations:
                    del self.active_conversations[conversation_id]
                logger.info(f"Deleted chat {conversation_id} from file")
                return True
            except Exception as e:
                logger.error(f"Error deleting chat {conversation_id}: {e}")
                return False
        else:
            logger.warning(f"Chat {conversation_id} not found for deletion")
            return False

    async def save_qa_interaction(self, session_id: str, conversation_id: str, 
                                 question: str, response: str, model_used: str = None,
                                 tokens_used: int = None, response_time_ms: int = None,
                                 metadata: Dict = None) -> Optional[str]:
        """
        Save a question-answer interaction to chat_history table for analytics.
        Only works when database is available and session is provided.
        
        Returns:
            The chat_id for the saved interaction, or None if not saved
        """
        if not self.use_database or not session_id:
            return None
            
        try:
            # Validate session
            session_data = await self.validate_session(session_id)
            if not session_data:
                raise ValueError("Invalid or expired session")
            
            member_id = session_data["member_id"]
            chat_id = str(uuid.uuid4())
            
            query = """
                INSERT INTO chat_history 
                (chat_id, conversation_id, session_id, member_id, question, response, 
                 model_used, tokens_used, response_time_ms, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.db.execute_query(query, (
                chat_id, conversation_id, session_id, member_id, question, response,
                model_used, tokens_used, response_time_ms,
                json.dumps(metadata) if metadata else None
            ))
            
            return chat_id
            
        except Exception as e:
            logger.error(f"Error saving QA interaction: {e}")
            return None
        
    async def _generate_title(self, messages: List[Any]) -> str:
        """
        Generate a title for a chat based on its content (existing functionality).
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
        Convert message objects to a serializable format (existing functionality).
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
        Convert serialized messages back to message objects (existing functionality).
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