# src/streamlit_app.py
from __future__ import annotations
import streamlit as st
import requests
import uuid
import os
import json
import logging
import datetime  # Import the datetime module instead of from datetime import datetime

logger = logging.getLogger(__name__)

# ────────────────────────────── Config ────────────────────────────────────
API_BASE = os.getenv("CHATBOT_API_BASE", "http://localhost:8000")
TIMEOUT_S = 600
HEADERS = {"X-API-Key": os.getenv("API_KEY", "demo-key")}
MODEL_ID = "gemma3:12b"

ALLOWED_TYPES = (
    "pdf", "docx", "hwp", "pptx", "msg",
    "xls", "xlsx", "txt", "csv", "png", "jpg", "jpeg"
)

st.set_page_config(page_title="DSTI Chatbot", page_icon="🤖", layout="wide")

# ────────────────────────── API helpers ───────────────────────────────────
def _stream_query(prompt: str):
    params = {
        "query": prompt,
        "conversation_id": st.session_state.active_cid,
        "plain_text": "true",
        "model": MODEL_ID,
    }
    if st.session_state.file_doc_id:
        params["filter_document_id"] = st.session_state.file_doc_id
    return requests.get(
        f"{API_BASE}/query/stream-get",
        params=params,
        headers=HEADERS,
        stream=True,
        timeout=TIMEOUT_S
    )

def _embed_file(upload):
    try:
        files = {"file": (upload.name, upload.getvalue(), upload.type)}
        # Add timestamp and conversation ID to metadata
        metadata = {
            "document_type": "filechat",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "conversation_id": st.session_state.active_cid
        }
        data = {"metadata": json.dumps(metadata)}
        
        response = requests.post(
            f"{API_BASE}/ingest/upload-chat",
            files=files,
            data=data,
            headers=HEADERS,
            timeout=TIMEOUT_S
        )

        if response.status_code != 200:
            logger.error(f"Server returned status {response.status_code}: {response.text}")
            st.error(f"Failed to upload file: Server returned status {response.status_code}")
            return False

        try:
            result = response.json()
            logger.debug(f"Received response: {result}")
            if result.get("status") == "success":
                file_id = result.get("id")
                st.session_state.update(file_doc_id=file_id, file_name=upload.name)
                
                # Add file to current conversation's file list
                if st.session_state.active_cid not in st.session_state.conversation_files:
                    st.session_state.conversation_files[st.session_state.active_cid] = []
                
                # Check if the file is already in the conversation files (by name)
                file_already_exists = any(
                    file_info["name"] == upload.name 
                    for file_info in st.session_state.conversation_files[st.session_state.active_cid]
                )
                
                # Only add the file to the list if it's not already there
                if not file_already_exists:
                    st.session_state.conversation_files[st.session_state.active_cid].append({
                        "id": file_id,
                        "name": upload.name,
                        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    
                    # Add system message about file upload
                    active_conv()["messages"].append({
                        "role": "assistant", 
                        "content": f"📄 File *{upload.name}* has been uploaded and processed. You can now ask questions about its content."
                    })

                     # Call the refresh_stores API to ensure the file is immediately available for querying
                try:
                    refresh_response = requests.post(
                        f"{API_BASE}/query/refresh-stores",
                        headers=HEADERS,
                        timeout=10  # Short timeout for refresh
                    )
                    
                    if refresh_response.status_code == 200:
                        logger.info("Vector stores refreshed successfully")
                        # Success message is shown below, no need for additional message
                    else:
                        logger.warning(f"Failed to refresh vector stores: {refresh_response.status_code}")
                        # We don't show warning here since the file was uploaded successfully
                except Exception as refresh_error:
                    logger.error(f"Error refreshing vector stores: {refresh_error}")
                    # We continue without showing error since the file was uploaded successfully
                
                st.success(f"📎 *{upload.name}* embedded — ask away!")
                return True
                
            elif result.get("status") == "error":
                error_message = result.get("message", "Unknown error")
                
                # Special case: if the error message actually indicates success
                if isinstance(error_message, str) and "successfully" in error_message.lower():
                    # Handle as success
                    file_id = result.get("id") or upload.name
                    st.session_state.update(file_doc_id=file_id, file_name=upload.name)
                    
                    # Add file to current conversation's file list
                    if st.session_state.active_cid not in st.session_state.conversation_files:
                        st.session_state.conversation_files[st.session_state.active_cid] = []
                    
                    # Check if the file is already in the conversation files (by name)
                    file_already_exists = any(
                        file_info["name"] == upload.name 
                        for file_info in st.session_state.conversation_files[st.session_state.active_cid]
                    )
                    
                    # Only add the file to the list if it's not already there
                    if not file_already_exists:
                        st.session_state.conversation_files[st.session_state.active_cid].append({
                            "id": file_id,
                            "name": upload.name,
                            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        })
                        
                        # Add system message about file upload
                        active_conv()["messages"].append({
                            "role": "assistant", 
                            "content": f"📄 File *{upload.name}* has been uploaded and processed. You can now ask questions about its content."
                        })

                    # Call the refresh_stores API to ensure the file is immediately available for querying
                    try:
                        refresh_response = requests.post(
                            f"{API_BASE}/query/refresh-stores",
                            headers=HEADERS,
                            timeout=10  # Short timeout for refresh
                        )
                        
                        if refresh_response.status_code == 200:
                            logger.info("Vector stores refreshed successfully")
                            # Success message is shown below, no need for additional message
                        else:
                            logger.warning(f"Failed to refresh vector stores: {refresh_response.status_code}")
                            # We don't show warning here since the file was uploaded successfully
                    except Exception as refresh_error:
                        logger.error(f"Error refreshing vector stores: {refresh_error}")
                    
                    st.success(f"📎 *{upload.name}* embedded — ask away!")
                    return True
                else:
                    # It's a genuine error
                    results = result.get("results", [])
                    if results:
                        error_message = results[0].get("message", error_message)
                    logger.error(f"File upload failed: {error_message}")
                    st.error(f"Failed to upload file: {error_message}")
                    return False
            else:
                logger.error(f"Unexpected response status: {result}")
                st.error("Failed to upload file: Unexpected server response")
                return False
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from server: {response.text}")
            st.error(f"Failed to process server response: Invalid JSON format")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to server failed: {e}")
        st.error(f"Failed to connect to server: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
        st.error(f"Unexpected error: {str(e)}")
        return False

def _load_chat_history():
    """Fetch all chat histories from the API."""
    try:
        response = requests.get(
            f"{API_BASE}/chat_routes/chats",
            headers=HEADERS,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            chats = response.json()
            return chats
        else:
            logger.error(f"Failed to load chat histories: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error loading chat histories: {e}")
        return []

def _load_chat_by_id(conversation_id):
    """Fetch a specific chat by ID."""
    try:
        response = requests.get(
            f"{API_BASE}/chat_routes/chats/{conversation_id}",
            headers=HEADERS,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to load chat {conversation_id}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error loading chat {conversation_id}: {e}")
        return None

def _save_chat(conversation_id, messages, title=None):
    """Save a chat to the server."""
    try:
        data = {
            "conversation_id": conversation_id,
            "messages": messages,
            "title": title
        }
        
        response = requests.post(
            f"{API_BASE}/chat_routes/chats",
            json=data,
            headers=HEADERS,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            logger.info(f"Chat {conversation_id} saved successfully")
            return True
        else:
            logger.error(f"Failed to save chat {conversation_id}: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error saving chat {conversation_id}: {e}")
        return False

def _delete_chat(conversation_id):
    """Delete a chat by ID."""
    try:
        response = requests.delete(
            f"{API_BASE}/chat_routes/chats/{conversation_id}",
            headers=HEADERS,
            timeout=TIMEOUT_S
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"Chat {conversation_id} deleted successfully")
                return True
            else:
                logger.warning(f"Failed to delete chat {conversation_id}: {result.get('message')}")
                return False
        else:
            logger.error(f"Failed to delete chat {conversation_id}: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error deleting chat {conversation_id}: {e}")
        return False

# ─────────────────────────── Session state ───────────────────────────────
def _init_state():
    s = st.session_state
    if "sidebar_open" not in s:
        s.sidebar_open = True
    if "conversations" not in s:
        # Try to load conversations from the API first
        api_chats = _load_chat_history()
        if api_chats:
            # Convert API chat list to the expected format for the UI
            s.conversations = {
                chat["id"]: {
                    "title": chat["title"],
                    "messages": [],  # We'll load messages on demand when switching to a chat
                    "created_at": chat["created_at"],
                    "updated_at": chat["updated_at"]
                } for chat in api_chats
            }
        else:
            # Fallback to creating a new chat
            s.conversations = {}
    
    if "active_cid" not in s:
        if s.conversations:
            # Use the most recently updated chat as the active one
            s.active_cid = sorted(
                s.conversations.keys(), 
                key=lambda cid: s.conversations[cid].get("updated_at", ""), 
                reverse=True
            )[0]
            
            # Load the messages for this chat
            chat_detail = _load_chat_by_id(s.active_cid)
            if chat_detail:
                s.conversations[s.active_cid]["messages"] = chat_detail["messages"]
        else:
            # Create a new chat if none exist
            cid = str(uuid.uuid4())
            s.conversations[cid] = {
                "title": f"Chat {len(s.conversations) + 1}",
                "messages": [],
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            s.active_cid = cid
    
    s.setdefault("file_doc_id", None)
    s.setdefault("file_name", None)
    s.setdefault("conversation_files", {})
    if s.active_cid not in s.conversation_files:
        s.conversation_files[s.active_cid] = []
_init_state()

def active_conv() -> dict:
    s = st.session_state
    if not hasattr(s, 'active_cid') or s.active_cid not in s.conversations:
        cid = str(uuid.uuid4())
        chat_num = len(s.conversations) + 1
        s.conversations[cid] = {"title": f"Chat {chat_num}", "messages": []}
        s.active_cid = cid
    return s.conversations[s.active_cid]

# ───────────────────────────── CSS / JS ───────────────────────────────────
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
    </style>
    <span id="toggle">☰</span>
    <script>
    const t = window.parent.document.getElementById("toggle");
    t.onclick = ()=>{document.body.classList.toggle("sidebar-hidden");};
    </script>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────── Sidebar ────────────────────────────────────
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        nid = str(uuid.uuid4())
        chat_num = len(st.session_state.conversations) + 1
        st.session_state.conversations[nid] = {
            "title": f"Chat {chat_num}", 
            "messages": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        st.session_state.update(active_cid=nid, file_doc_id=None, file_name=None)
        # Save new chat to server
        _save_chat(nid, [], f"Chat {chat_num}")

    st.markdown("### Chat History")
    for cid, conv in list(st.session_state.conversations.items()):
        col1, col2 = st.columns([7, 1])
        with col1:
            if st.button(conv["title"], key=f"nav_{cid}"):
                # Load messages for this chat if not already loaded
                if not conv.get("messages"):
                    chat_detail = _load_chat_by_id(cid)
                    if chat_detail:
                        st.session_state.conversations[cid]["messages"] = chat_detail["messages"]
                
                st.session_state.active_cid = cid
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                # Delete from server first
                if _delete_chat(cid):
                    st.session_state.conversations.pop(cid)
                    if cid == st.session_state.active_cid:
                        if st.session_state.conversations:
                            st.session_state.active_cid = next(iter(st.session_state.conversations))
                        else:
                            new_cid = str(uuid.uuid4())
                            st.session_state.conversations[new_cid] = {
                                "title": "Chat 1", 
                                "messages": [],
                                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                            }
                            st.session_state.active_cid = new_cid
                            # Save new chat to server
                            _save_chat(new_cid, [], "Chat 1")
                    st.rerun()
                else:
                    st.error(f"Failed to delete chat {cid}")

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
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    st.markdown("</div>", unsafe_allow_html=True)

# Add the chat input
prompt = st.chat_input("Type your question and press Enter…")

# Create session state values for tracking file processing and UI state
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "file_being_processed" not in st.session_state:
    st.session_state.file_being_processed = False
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "show_success" not in st.session_state:
    st.session_state.show_success = False

# Show success message if needed
if st.session_state.show_success and st.session_state.last_uploaded_filename:
    st.success(f"📎 *{st.session_state.last_uploaded_filename}* embedded — ask away!")
    st.session_state.show_success = False

# File uploader - we don't use on_change callback since we've had issues with it
uploaded_file = st.file_uploader("Upload a file to chat about its content", type=ALLOWED_TYPES, key="file_uploader")

# Handle file uploads with careful state tracking
if uploaded_file is not None and not st.session_state.file_being_processed and not st.session_state.file_processed:
    # Mark as being processed to prevent reprocessing
    st.session_state.file_being_processed = True
    
    with st.spinner("Uploading and processing file..."):
        # Process the file
        success = _embed_file(uploaded_file)
        
        if success:
            # Save the filename for the success message
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.show_success = True
            
        # Mark as processed instead of trying to reset the uploader
        st.session_state.file_processed = True
        st.session_state.file_being_processed = False
        
        # Trigger rerun to refresh UI and show success message
        st.rerun()

# Reset processing flags when uploader is empty
if uploaded_file is None and st.session_state.file_processed:
    st.session_state.file_processed = False

# ───────────────────────── On user submit ─────────────────────────────────
if prompt:
    active_conv()["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty().markdown("⌛ *thinking…*")

    try:
        resp = _stream_query(prompt)
        parts = []
        for chunk in resp.iter_content(decode_unicode=True):
            if chunk:
                parts.append(chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk)
                placeholder.markdown("".join(parts))
        answer = "".join(parts)
    except Exception as exc:
        answer = f"⚠️ Error: {exc}"
        placeholder.markdown(answer)

    active_conv()["messages"].append({"role": "assistant", "content": answer})
    
    # Save the chat to the server
    conv = active_conv()
    _save_chat(st.session_state.active_cid, conv["messages"], conv["title"])
    
    # Update timestamp
    st.session_state.conversations[st.session_state.active_cid]["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    st.rerun()