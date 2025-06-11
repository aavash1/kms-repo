# src/streamlit_app.py
from __future__ import annotations
import streamlit as st
import requests
import uuid
import os
import json
import logging
import datetime
import re

logger = logging.getLogger(__name__)

# ────────────────────────────── Config ────────────────────────────────────
API_BASE = os.getenv("CHATBOT_API_BASE", "http://localhost:8000")
TIMEOUT_S = 600
HEADERS = {"X-API-Key": os.getenv("API_KEY", "demo-key")}
MODEL_ID = "gemma3:12b"

# Authentication configuration
DEFAULT_MEMBER_ID = os.getenv("DEFAULT_MEMBER_ID", "demo_user_001")
ENABLE_MEMBER_AUTH = os.getenv("ENABLE_MEMBER_AUTH", "true").lower() == "true"

# Response format configuration
DEFAULT_RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "stream")

ALLOWED_TYPES = (
    "pdf", "docx", "hwp", "pptx", "msg",
    "xls", "xlsx", "txt", "csv", "png", "jpg", "jpeg"
)

st.set_page_config(page_title="DSTI Chatbot", page_icon="🤖", layout="wide")

# ────────────────────────── Helper Functions ──────────────────────────────
def generate_chat_title(first_message: str, max_length: int = 50) -> str:
    """Generate a chat title from the first user message."""
    if not first_message:
        return "New Chat"
    
    title = first_message.strip()
    title = re.sub(r'^(what is|what are|how do|how to|explain|tell me about|can you)\s+', '', title, flags=re.IGNORECASE)
    
    if len(title) > max_length:
        title = title[:max_length].rsplit(' ', 1)[0] + "..."
    
    title = title[0].upper() + title[1:] if title else "New Chat"
    return title

# ────────────────────────── Authentication helpers ────────────────────────────
def get_or_create_member_session():
    """Get or create a member session - simplified without external session API."""
    if "member_id" not in st.session_state:
        st.session_state.member_id = DEFAULT_MEMBER_ID
    
    # Remove session creation API calls - we only need member_id
    st.session_state.session_token = None
    st.session_state.session_id = None
    
    logger.info(f"Member session initialized: {st.session_state.member_id}")

def get_auth_headers():
    """Get headers with authentication information."""
    headers = HEADERS.copy()
    if ENABLE_MEMBER_AUTH and hasattr(st.session_state, 'member_id') and st.session_state.member_id:
        headers["X-Member-ID"] = st.session_state.member_id
    return headers

def test_api_connectivity():
    """Test if the API server and routes are accessible."""
    connectivity_status = {
        "api_server": False,
        "chat_routes": False,
        "query_routes": False,
        "error_messages": []
    }
    
    # Test chat routes directly (this also confirms API server is running)
    try:
        auth_headers = get_auth_headers()
        response = requests.get(f"{API_BASE}/chat/health", headers=auth_headers, timeout=5)
        if response.status_code == 200:
            connectivity_status["chat_routes"] = True
            connectivity_status["api_server"] = True  # If chat works, API server works
        else:
            connectivity_status["error_messages"].append(f"Chat routes not available: {response.status_code}")
    except Exception as e:
        connectivity_status["error_messages"].append(f"Cannot connect to API server: {str(e)}")
    
    # Test query routes
    try:
        auth_headers = get_auth_headers()
        response = requests.get(f"{API_BASE}/query/stream-get", 
                              params={"query": "test"}, 
                              headers=auth_headers, timeout=5)
        if response.status_code in [200, 400]:  # 400 is ok for test query
            connectivity_status["query_routes"] = True
    except Exception as e:
        connectivity_status["error_messages"].append(f"Query routes not available: {str(e)}")
    
    return connectivity_status

# ────────────────────────── 6 API IMPLEMENTATIONS ───────────────────────────────

# 1. NEW CHAT (using stream-get-upload) - ENHANCED
def _start_new_chat(prompt: str, uploaded_files=None):
    """Start a new chat conversation using stream-get-upload endpoint."""
    try:
        # Generate new conversation_id for new chat
        new_conversation_id = str(uuid.uuid4())
        
        # Prepare form data for stream-get-upload
        form_data = {
            "query": prompt,
            "conversation_id": new_conversation_id,  # New conversation
            "document_type": "memo",  # Default type
        }
        
        # Handle file uploads if provided
        files_to_upload = {}
        if uploaded_files:
            # This would need to be implemented with actual file upload logic
            form_data["file_urls"] = []  # Empty for now, would contain S3 URLs
        
        auth_headers = get_auth_headers()
        
        logger.info(f"Starting new chat with conversation_id: {new_conversation_id}")
        
        # Use stream-get-upload for new chat
        response = requests.post(
            f"{API_BASE}/query/stream-get-upload",
            data=form_data,
            headers=auth_headers,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "conversation_id": new_conversation_id,
                "response": result.get("response", ""),
                "title": result.get("title", generate_chat_title(prompt))
            }
        else:
            logger.error(f"Failed to start new chat: {response.status_code} - {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error starting new chat: {e}")
        return {"success": False, "error": str(e)}

# 2. CONTINUE CHAT (using stream-get-upload) - ENHANCED  
def _continue_chat(prompt: str, conversation_id: str, uploaded_files=None):
    """Continue existing chat conversation using stream-get-upload endpoint."""
    try:
        # Prepare form data for continuing chat
        form_data = {
            "query": prompt,
            "conversation_id": conversation_id,  # Existing conversation
            "document_type": "memo",
        }
        
        # Handle file uploads if provided
        if uploaded_files:
            form_data["file_urls"] = []  # Would contain S3 URLs in real implementation
        
        auth_headers = get_auth_headers()
        
        logger.info(f"Continuing chat with conversation_id: {conversation_id}")
        
        # Use stream-get-upload for continuing chat
        response = requests.post(
            f"{API_BASE}/query/stream-get-upload",
            data=form_data,
            headers=auth_headers,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "conversation_id": conversation_id,
                "response": result.get("response", ""),
                "title": result.get("title", "Chat")
            }
        else:
            logger.error(f"Failed to continue chat: {response.status_code} - {response.text}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error continuing chat: {e}")
        return {"success": False, "error": str(e)}

# 3. LOAD CHAT LIST - ENHANCED (already exists, improved)
def _load_chat_history():
    """API 3: GET /chats - Fetch chat histories with proper error handling."""
    try:
        auth_headers = get_auth_headers()
        
        logger.info(f"Loading chat history from {API_BASE}/chat/chats")
        response = requests.get(
            f"{API_BASE}/chat/chats",
            headers=auth_headers,
            timeout=10
        )
        
        if response.status_code == 200:
            chats = response.json()
            logger.info(f"Successfully loaded {len(chats)} chats from API")
            return chats
        elif response.status_code == 404:
            logger.warning("Chat routes not found - routes may not be registered")
            st.warning("⚠️ Chat history unavailable. Chat routes not found on server.")
            return []
        else:
            logger.error(f"Failed to load chat histories: {response.status_code} - {response.text}")
            st.error(f"Failed to load chat histories: HTTP {response.status_code}")
            return []
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API server at {API_BASE}")
        st.error(f"❌ Cannot connect to API server at {API_BASE}")
        return []
    except requests.exceptions.Timeout:
        logger.error("Request timeout while loading chat history")
        st.error("⏱️ Request timeout - server may be slow")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading chat histories: {e}")
        st.error(f"Error loading chat histories: {str(e)}")
        return []

# 4. LOAD SPECIFIC CHAT - ENHANCED (already exists, improved)
def _load_chat_by_id(conversation_id):
    """API 4: GET /chats/{conversation_id} - Load specific chat with proper error handling."""
    try:
        auth_headers = get_auth_headers()
        response = requests.get(
            f"{API_BASE}/chat/chats/{conversation_id}",
            headers=auth_headers,
            timeout=10
        )
        
        if response.status_code == 200:
            chat_detail = response.json()
            logger.info(f"Loaded chat {conversation_id} with {len(chat_detail.get('messages', []))} messages")
            return chat_detail
        elif response.status_code == 404:
            logger.warning(f"Chat {conversation_id} not found")
            return None
        else:
            logger.error(f"Failed to load chat {conversation_id}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading chat {conversation_id}: {e}")
        return None

# 5. DELETE CHAT - ENHANCED (already exists, improved)
def _delete_chat(conversation_id):
    """API 5: DELETE /chats/{conversation_id} - Delete chat with proper error handling."""
    try:
        auth_headers = get_auth_headers()
        response = requests.delete(
            f"{API_BASE}/chat/chats/{conversation_id}",
            headers=auth_headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"Chat {conversation_id} deleted successfully")
                # Show success message to user
                st.success(f"✅ Chat '{result.get('message', 'Chat')}' deleted")
                return True
            else:
                logger.warning(f"Failed to delete chat {conversation_id}")
                st.error(f"❌ {result.get('message', 'Failed to delete chat')}")
                return False
        else:
            logger.error(f"Failed to delete chat {conversation_id}: {response.status_code}")
            st.error(f"❌ Failed to delete chat: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting chat {conversation_id}: {e}")
        st.error(f"❌ Error deleting chat: {str(e)}")
        return False

# 6. RENAME CHAT - FIXED IMPLEMENTATION
def _rename_chat(conversation_id, new_title):
    """API 6: PUT /chats/{conversation_id}/title - Rename chat title."""
    try:
        auth_headers = get_auth_headers()
        
        # Validate title
        if not new_title or not new_title.strip():
            st.error("❌ Title cannot be empty")
            return False
        
        new_title = new_title.strip()
        if len(new_title) > 100:
            new_title = new_title[:100] + "..."
            st.info(f"📝 Title truncated to: {new_title}")
        
        # FIXED: Send as embedded JSON body to match Body(..., embed=True)
        logger.debug(f"Sending rename request for {conversation_id} with title: '{new_title}'")
        response = requests.put(
            f"{API_BASE}/chat/chats/{conversation_id}/title",
            json={"title": new_title},  # Embed in object for Body(..., embed=True)
            headers={**auth_headers, "Content-Type": "application/json"},
            timeout=10
        )
        
        logger.debug(f"Rename response: {response.status_code} - {response.text[:200] if response.text else 'No content'}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Chat {conversation_id} renamed to '{new_title}'")
            st.success(f"✅ Chat renamed to '{new_title}'")
            return True
        elif response.status_code == 404:
            logger.warning(f"Chat {conversation_id} not found for rename")
            st.error("❌ Chat not found")
            return False
        elif response.status_code == 422:
            logger.error(f"Invalid data format for rename: {response.text}")
            st.error("❌ Invalid title format")
            return False
        else:
            logger.error(f"Failed to rename chat {conversation_id}: {response.status_code} - {response.text}")
            st.error(f"❌ Failed to rename chat: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error renaming chat {conversation_id}: {e}")
        st.error(f"❌ Error renaming chat: {str(e)}")
        return False

# ────────────────────────── LEGACY API helpers (for backward compatibility) ───────────────────────────────
def _stream_query(prompt: str):
    """Legacy streaming query - keeping for compatibility."""
    params = {
        "query": prompt,
        "conversation_id": st.session_state.active_cid,
        "plain_text": "true",
        "response_format": "stream",
        "include_member_docs": "true",
        "include_shared_docs": "true"
    }
    if st.session_state.file_doc_id:
        params["filter_document_id"] = st.session_state.file_doc_id
    
    auth_headers = get_auth_headers()
    
    return requests.get(
        f"{API_BASE}/query/stream-get",
        params=params,
        headers=auth_headers,
        stream=True,
        timeout=TIMEOUT_S
    )

def _json_query(prompt: str):
    """Legacy JSON query - keeping for compatibility."""
    params = {
        "query": prompt,
        "conversation_id": st.session_state.active_cid,
        "plain_text": "false",
        "response_format": "json",
        "include_sources": "true",
        "include_member_docs": "true",
        "include_shared_docs": "true"
    }
    if st.session_state.file_doc_id:
        params["filter_document_id"] = st.session_state.file_doc_id
    
    auth_headers = get_auth_headers()
    
    return requests.get(
        f"{API_BASE}/query/stream-get",
        params=params,
        headers=auth_headers,
        timeout=TIMEOUT_S
    )

def _save_chat(conversation_id, messages, title=None):
    """Save chat with proper error handling - mainly for backup saves."""
    try:
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({
                    "role": getattr(msg, 'role', 'unknown'),
                    "content": getattr(msg, 'content', str(msg))
                })
        
        data = {
            "conversation_id": conversation_id,
            "messages": formatted_messages,
            "title": title
        }
        
        auth_headers = get_auth_headers()
        response = requests.post(
            f"{API_BASE}/chat/chats",
            json=data,
            headers=auth_headers,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Chat {conversation_id} saved successfully")
            return True
        elif response.status_code == 404:
            logger.warning("Chat save endpoint not found")
            return False
        else:
            logger.error(f"Failed to save chat {conversation_id}: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving chat {conversation_id}: {e}")
        return False

# ─────────────────────────── Session state ───────────────────────────────
def _init_state():
    s = st.session_state
    
    # Initialize authentication first
    get_or_create_member_session()
    
    if "sidebar_open" not in s:
        s.sidebar_open = True
    
    if "api_status_checked" not in s:
        s.api_status_checked = False
        s.api_connectivity = None
    
    if "conversations" not in s:
        # Try to load conversations from API
        try:
            api_chats = _load_chat_history()
            if api_chats:
                s.conversations = {}
                for chat in api_chats:
                    s.conversations[chat["id"]] = {
                        "title": chat["title"],
                        "messages": [],
                        "created_at": chat["created_at"],
                        "updated_at": chat["updated_at"],
                        "loaded": False,
                        "message_count": chat.get("message_count", 0)  # NEW: Use message count from API
                    }
                logger.info(f"Initialized with {len(s.conversations)} conversations from API")
            else:
                logger.info("No conversations loaded from API")
                s.conversations = {}
        except Exception as e:
            logger.error(f"Failed to initialize conversations: {e}")
            s.conversations = {}
    
    if "active_cid" not in s:
        # ALWAYS start with a new chat for first-time users or fresh sessions
        # This ensures users land on the welcome screen instead of an old conversation
        cid = str(uuid.uuid4())
        s.conversations[cid] = {
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "loaded": True,
            "message_count": 0
        }
        s.active_cid = cid
        logger.info(f"Started with new chat: {cid} (fresh session)")
        
        # Note: Existing conversations are still loaded in sidebar for access
        # but the active view shows the welcome screen for new conversations
    
    s.setdefault("file_doc_id", None)
    s.setdefault("file_name", None)
    s.setdefault("conversation_files", {})
    s.setdefault("response_format", DEFAULT_RESPONSE_FORMAT)
    s.setdefault("rename_mode", {})  # NEW: For rename functionality
    if s.active_cid not in s.conversation_files:
        s.conversation_files[s.active_cid] = []

_init_state()

def active_conv() -> dict:
    """Get the active conversation, ensuring messages are loaded."""
    s = st.session_state
    if not hasattr(s, 'active_cid') or s.active_cid not in s.conversations:
        cid = str(uuid.uuid4())
        s.conversations[cid] = {
            "title": "New Chat", 
            "messages": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "loaded": True,
            "message_count": 0
        }
        s.active_cid = cid
    
    conv = s.conversations[s.active_cid]
    if not conv.get("loaded", False) and not conv.get("messages"):
        chat_detail = _load_chat_by_id(s.active_cid)
        if chat_detail and chat_detail.get("messages"):
            conv["messages"] = chat_detail["messages"]
            conv["loaded"] = True
    
    return conv

# ───────────────────────────── CSS / JS (enhanced for rename functionality) ───────────────────────────────────────
st.markdown(
    """
    <style>
    header, footer {visibility:hidden;}
    .sidebar-hidden > div[data-testid="stSidebar"] {transform:translateX(-18rem);}
    .sidebar-hidden .main .block-container{padding-left:1rem !important;}
    #toggle{position:fixed;top:10px;left:12px;font-size:24px;cursor:pointer;z-index:1000;}
    .chat-col{max-width:760px;margin:auto;}
    .stChatInputContainer {
        position: fixed !important;
        bottom: 50px !important;
        background: white !important;
        padding: 1rem !important;
        padding-right: 3rem !important;
        margin-left: 15rem !important;
        width: calc(100% - 15rem) !important;
        z-index: 999 !important;
        border-top: 1px solid #eee !important;
    }
    .sidebar-hidden .stChatInputContainer {
        margin-left: 0 !important;
        width: 100% !important;
    }
    [data-testid="stVerticalBlock"] {
        padding-bottom: 130px !important;
    }
    [data-testid="stChatMessage"] {
        padding: 0 !important;
        margin: 0 0 1rem 0 !important;
        display: flex !important;
    }
    [data-testid="stChatMessage"] .stMarkdown {
        padding: 12px 16px !important;
        border-radius: 15px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,.1) !important;
        max-width: 80% !important;
        position: relative !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="UserIcon"]) {
        justify-content: flex-end !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="UserIcon"]) .stMarkdown {
        background: #fdecea !important;
        color: #611a15 !important;
        border-bottom-right-radius: 0 !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="UserIcon"]) .stMarkdown::after {
        content: "" !important;
        position: absolute !important;
        bottom: 0 !important;
        right: -10px !important;
        width: 20px !important;
        height: 20px !important;
        background: #fdecea !important;
        border-bottom-left-radius: 15px !important;
        z-index: -1 !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="BotIcon"]) {
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="BotIcon"]) .stMarkdown {
        background: #fff9db !important;
        color: #665200 !important;
        border-bottom-left-radius: 0 !important;
    }
    [data-testid="stChatMessage"]:has(svg[data-testid="BotIcon"]) .stMarkdown::after {
        content: "" !important;
        position: absolute !important;
        bottom: 0 !important;
        left: -10px !important;
        width: 20px !important;
        height: 20px !important;
        background: #fff9db !important;
        border-bottom-right-radius: 15px !important;
        z-index: -1 !important;
    }
    div[data-testid="stFileUploader"] {
        position: fixed !important;
        bottom: 10px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 400px !important;
        z-index: 998 !important;
    }
    .member-info {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
    }
    .api-status {
        background: #e8f4fd;
        border: 1px solid #b8daff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
    }
    .api-status.error {
        background: #f8d7da;
        border-color: #f5c6cb;
    }
    /* NEW: Chat item styling with message count */
    .chat-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px solid #eee;
    }
    .chat-title {
        flex: 1;
        font-size: 0.9rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .message-count {
        font-size: 0.7rem;
        color: #666;
        margin-right: 0.5rem;
    }
    </style>
    <span id="toggle">☰</span>
    <script>
    const t = window.parent.document.getElementById("toggle");
    t.onclick = ()=>{document.body.classList.toggle("sidebar-hidden");};
    </script>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar (ENHANCED with all 6 API integrations) ────────────────────────────────────
with st.sidebar:
    # API Status Check
    if not st.session_state.api_status_checked:
        with st.spinner("Checking API connectivity..."):
            st.session_state.api_connectivity = test_api_connectivity()
            st.session_state.api_status_checked = True
    
    # Show API status
    if st.session_state.api_connectivity:
        status = st.session_state.api_connectivity
        if status["api_server"] and status["chat_routes"] and status["query_routes"]:
            st.markdown(
                """
                <div class="api-status">
                    ✅ API Connected<br>
                    ✅ Chat Routes Available<br>
                    ✅ Query Routes Available
                </div>
                """,
                unsafe_allow_html=True
            )
        elif status["api_server"]:
            st.markdown(
                """
                <div class="api-status error">
                    ✅ API Connected<br>
                    ❌ Some Routes Unavailable<br>
                    <small>Limited functionality</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="api-status error">
                    ❌ API Server Unavailable<br>
                    <small>Check if server is running</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Show member info
    if ENABLE_MEMBER_AUTH and hasattr(st.session_state, 'member_id') and st.session_state.member_id:
        st.markdown(
            f"""
            <div class="member-info">
                👤 Member: {st.session_state.member_id}<br>
                📡 Auth: Member-based
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Response format selector
    response_format = st.selectbox(
        "🔧 Response Format:",
        ["stream", "json"],
        index=0 if st.session_state.response_format == "stream" else 1,
        help="Stream: Real-time response\nJSON: Structured response with metadata"
    )
    st.session_state.response_format = response_format
    
    # API 1: NEW CHAT BUTTON (Enhanced to use stream-get-upload)
    if st.button("➕ New chat", use_container_width=True):
        nid = str(uuid.uuid4())
        st.session_state.conversations[nid] = {
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "loaded": True,
            "message_count": 0
        }
        st.session_state.update(active_cid=nid, file_doc_id=None, file_name=None)
        st.rerun()

    # API 3 & 4: CHAT HISTORY NAVIGATION (Enhanced with rename functionality)
    if st.session_state.conversations:
        st.markdown("### 💬 Chat History")
        
        for cid, conv in list(st.session_state.conversations.items()):
            # Check if this chat is in rename mode
            is_renaming = st.session_state.rename_mode.get(cid, False)
            
            if is_renaming:
                # RENAME MODE: Show text input (FIXED - added proper label)
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    new_title = st.text_input(
                        "Chat Title", 
                        value=conv["title"], 
                        key=f"rename_input_{cid}",
                        label_visibility="collapsed",  # Hide label but provide for accessibility
                        placeholder="Enter new chat title..."
                    )
                with col2:
                    if st.button("✅", key=f"save_{cid}", help="Save"):
                        # API 6: RENAME CHAT
                        if _rename_chat(cid, new_title):
                            st.session_state.conversations[cid]["title"] = new_title
                            st.session_state.rename_mode[cid] = False
                            st.rerun()
                with col3:
                    if st.button("❌", key=f"cancel_{cid}", help="Cancel"):
                        st.session_state.rename_mode[cid] = False
                        st.rerun()
            else:
                # NORMAL MODE: Show chat title with buttons
                col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
                with col1:
                    # Show title with message count
                    title_with_count = conv["title"]
                    msg_count = conv.get("message_count", len(conv.get("messages", [])))
                    if msg_count > 0:
                        title_with_count += f" ({msg_count})"
                    
                    # API 4: LOAD SPECIFIC CHAT
                    if st.button(title_with_count, key=f"nav_{cid}"):
                        if not conv.get("messages"):
                            chat_detail = _load_chat_by_id(cid)
                            if chat_detail:
                                st.session_state.conversations[cid]["messages"] = chat_detail["messages"]
                                st.session_state.conversations[cid]["loaded"] = True
                        
                        st.session_state.active_cid = cid
                        st.rerun()
                
                with col2:
                    # API 6: RENAME BUTTON
                    if st.button("✏️", key=f"edit_{cid}", help="Rename"):
                        st.session_state.rename_mode[cid] = True
                        st.rerun()
                
                with col3:
                    # API 5: DELETE CHAT
                    if st.button("🗑️", key=f"del_{cid}", help="Delete"):
                        if _delete_chat(cid):
                            st.session_state.conversations.pop(cid, None)
                            st.session_state.rename_mode.pop(cid, None)
                            
                            # Handle active conversation cleanup
                            if cid == st.session_state.active_cid:
                                if st.session_state.conversations:
                                    st.session_state.active_cid = next(iter(st.session_state.conversations))
                                else:
                                    new_cid = str(uuid.uuid4())
                                    st.session_state.conversations[new_cid] = {
                                        "title": "New Chat",
                                        "messages": [],
                                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                        "loaded": True,
                                        "message_count": 0
                                    }
                                    st.session_state.active_cid = new_cid
                            st.rerun()
                
                with col4:
                    # Show active indicator
                    if cid == st.session_state.active_cid:
                        st.markdown("🔵")
    else:
        st.markdown("### 💬 No Chat History")
        st.write("Start a conversation to see your chat history here.")

# ─────────────────────── Chat history / welcome ───────────────────────────
chat_col = st.container()
with chat_col:
    st.markdown("<div class='chat-col'>", unsafe_allow_html=True)

    if not active_conv()["messages"]:
        st.markdown(
            "<h1 style='text-align:center;'>DSTI Chatbot Assistant</h1>"
            "<p style='text-align:center;'>Ask me anything about DSTI documentation "
            "— or drop a file to chat about its content.</p>",
            unsafe_allow_html=True,
        )
    else:
        for m in active_conv()["messages"]:
            # Handle both 'role' and 'type' fields for compatibility
            role = m.get("role") or m.get("type", "user")
            # Convert type format to role format if needed
            if "human" in role.lower():
                role = "user"
            elif "ai" in role.lower():
                role = "assistant"
            
            with st.chat_message(role):
                st.markdown(m.get("content", ""))

    st.markdown("</div>", unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Type your question and press Enter…")

# File uploader
uploaded_file = st.file_uploader("Upload a file to chat about its content", type=ALLOWED_TYPES)

# Handle file uploads
if uploaded_file is not None:
    with st.spinner("Processing file..."):
        # Basic file embedding logic (simplified)
        st.success(f"📎 File {uploaded_file.name} uploaded successfully!")

# Handle user input (ENHANCED to use new chat APIs)
if prompt:
    # Get current conversation
    conv = active_conv()
    
    # Determine if this is a new chat or continuing existing
    is_new_chat = len(conv["messages"]) == 0
    
    if is_new_chat:
        # API 1: NEW CHAT - Use stream-get-upload for new conversations
        with st.spinner("Starting new conversation..."):
            result = _start_new_chat(prompt, uploaded_file)
            
            if result["success"]:
                # Update conversation with response from stream-get-upload
                conv["messages"].extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result["response"]}
                ])
                
                # Update title if provided
                if result.get("title") and result["title"] != "New Chat":
                    conv["title"] = result["title"]
                    st.session_state.conversations[st.session_state.active_cid]["title"] = result["title"]
                
                # Update conversation ID if changed
                if result.get("conversation_id") and result["conversation_id"] != st.session_state.active_cid:
                    old_cid = st.session_state.active_cid
                    new_cid = result["conversation_id"]
                    
                    # Move conversation data to new ID
                    st.session_state.conversations[new_cid] = st.session_state.conversations.pop(old_cid)
                    st.session_state.active_cid = new_cid
                
                st.rerun()
            else:
                st.error(f"❌ Failed to start new chat: {result.get('error', 'Unknown error')}")
    
    else:
        # API 2: CONTINUE CHAT - Use stream-get-upload for existing conversations
        with st.spinner("Continuing conversation..."):
            result = _continue_chat(prompt, st.session_state.active_cid, uploaded_file)
            
            if result["success"]:
                # Update conversation with response from stream-get-upload
                conv["messages"].extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result["response"]}
                ])
                
                # Update timestamp
                st.session_state.conversations[st.session_state.active_cid]["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                st.session_state.conversations[st.session_state.active_cid]["message_count"] = len(conv["messages"])
                
                st.rerun()
            else:
                st.error(f"❌ Failed to continue chat: {result.get('error', 'Unknown error')}")
                
                # Fallback to legacy streaming if stream-get-upload fails
                st.warning("⚠️ Falling back to legacy query method...")
                
                # Add user message immediately
                active_conv()["messages"].append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    placeholder = st.empty().markdown("⌛ *thinking…*")

                try:
                    if st.session_state.response_format == "json":
                        resp = _json_query(prompt)
                        if resp.status_code == 200:
                            result = resp.json()
                            if result.get("status") == "success":
                                answer = result["data"]["response"]
                                placeholder.markdown(answer)
                                
                                # Show metadata in JSON mode
                                sources = result["data"].get("sources", [])
                                if sources:
                                    with st.expander("📚 Sources"):
                                        for i, source in enumerate(sources[:3]):
                                            st.write(f"**Source {i+1}:** {source.get('document_type', 'Unknown')}")
                                            st.write(f"**Content:** {source.get('chunk_content', 'N/A')[:200]}...")
                                            st.write("---")
                            else:
                                answer = f"⚠️ Error: {result.get('message', 'Unknown error')}"
                                placeholder.markdown(answer)
                        else:
                            answer = f"⚠️ Error: Server returned status {resp.status_code}"
                            placeholder.markdown(answer)
                    else:
                        # Streaming response
                        resp = _stream_query(prompt)
                        if resp.status_code == 200:
                            parts = []
                            for chunk in resp.iter_content(decode_unicode=True):
                                if chunk:
                                    chunk_text = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                                    parts.append(chunk_text)
                                    placeholder.markdown("".join(parts))
                            answer = "".join(parts)
                        else:
                            answer = f"⚠️ Error: Server returned status {resp.status_code}"
                            placeholder.markdown(answer)
                        
                except Exception as exc:
                    answer = f"⚠️ Error: {exc}"
                    placeholder.markdown(answer)

                active_conv()["messages"].append({"role": "assistant", "content": answer})
                
                # Save chat to server (graceful failure)
                conv = active_conv()
                _save_chat(st.session_state.active_cid, conv["messages"], conv["title"])
                
                # Update timestamp
                st.session_state.conversations[st.session_state.active_cid]["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                st.session_state.conversations[st.session_state.active_cid]["message_count"] = len(conv["messages"])
                
                st.rerun()