# src/streamlit_app.py
from __future__ import annotations
import streamlit as st
import requests
import uuid
import os
import json
import logging

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

# ─────────────────────────── Session state ───────────────────────────────
def _init_state():
    s = st.session_state
    if "sidebar_open" not in s:
        s.sidebar_open = True
    if "conversations" not in s:
        s.conversations = {}
    if "active_cid" not in s:
        cid = str(uuid.uuid4())
        s.conversations[cid] = {"title": f"Chat {len(s.conversations) + 1}", "messages": []}
        s.active_cid = cid
    s.setdefault("file_doc_id", None)
    s.setdefault("file_name", None)
_init_state()

def active_conv() -> dict:
    s = st.session_state
    if not hasattr(s, 'active_cid') or s.active_cid not in s.conversations:
        cid = str(uuid.uuid4())
        chat_num = len(s.conversations) + 1
        s.conversations[cid] = {"title": f"Chat {chat_num}", "messages": []}
        s.active_cid = cid
    return s.conversations[s.active_cid]

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
        data = {"metadata": json.dumps({"document_type": "filechat"})}
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
            return

        try:
            result = response.json()
            logger.debug(f"Received response: {result}")
            if result.get("status") == "success":
                st.session_state.update(file_doc_id=result.get("id"), file_name=upload.name)
                st.success(f"📎 *{upload.name}* embedded — ask away!")
            elif result.get("status") == "error":
                error_message = result.get("message", "Unknown error")
                results = result.get("results", [])
                if results:
                    error_message = results[0].get("message", error_message)
                logger.error(f"File upload failed: {error_message}")
                st.error(f"Failed to upload file: {error_message}")
            else:
                logger.error(f"Unexpected response status: {result}")
                st.error("Failed to upload file: Unexpected server response")
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from server: {response.text}")
            st.error(f"Failed to process server response: Invalid JSON format")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to server failed: {e}")
        st.error(f"Failed to connect to server: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
        st.error(f"Unexpected error: {str(e)}")

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
        st.session_state.conversations[nid] = {"title": f"Chat {chat_num}", "messages": []}
        st.session_state.update(active_cid=nid, file_doc_id=None, file_name=None)

    st.markdown("### Chat History")
    for cid, conv in list(st.session_state.conversations.items()):
        col1, col2 = st.columns([7, 1])
        with col1:
            if st.button(conv["title"], key=f"nav_{cid}"):
                st.session_state.active_cid = cid
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                st.session_state.conversations.pop(cid)
                if cid == st.session_state.active_cid:
                    if st.session_state.conversations:
                        st.session_state.active_cid = next(iter(st.session_state.conversations))
                    else:
                        new_cid = str(uuid.uuid4())
                        st.session_state.conversations[new_cid] = {"title": "Chat 1", "messages": []}
                        st.session_state.active_cid = new_cid
                st.rerun()

    st.divider()
    st.markdown("### Info")
    st.caption("This app answers questions based on DSTI documentation.")
    st.caption("© 2025 DSTI Chatbot Assistant")

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

# Add the chat input and file uploader
prompt = st.chat_input("Type your question and press Enter…")
file_uploader = st.file_uploader("Upload a file to chat about its content", type=ALLOWED_TYPES, key="file_uploader")
if file_uploader is not None:
    with st.spinner("Uploading and processing file..."):
        _embed_file(file_uploader)

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
    st.rerun()