# src/streamlit_app.py
import streamlit as st
import requests
import uuid
import json
import os
import time
from datetime import datetime
import re

# Set page configuration
st.set_page_config(
    page_title="NetBackup Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS with bottom input positioning
st.markdown("""
<style>
    /* Base text styling */
    .stMarkdown p {
        margin-bottom: 10px;
        font-weight: normal;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* Headers - less bold and better spacing */
    .stMarkdown h2 {
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 500;  /* Less bold */
        font-size: 18px;
        color: #2c3e50;
    }
    
    /* Code blocks with subtle styling */
    .stMarkdown pre {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        border: 1px solid #eaecef;
    }
    
    /* Inline code with subtle styling */
    .stMarkdown code {
        padding: 2px 5px;
        background-color: #f8f9fa;
        border-radius: 3px;
        font-weight: normal;
        color: #e83e8c;
        font-size: 0.9em;
    }
    
    /* Make bold text less intense */
    .stMarkdown strong {
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* Add subtle dividers */
    .stMarkdown hr {
        margin: 15px 0;
        border: 0;
        height: 1px;
        background-color: #eaecef;
    }
    
    /* More subtle list items */
    .stMarkdown ul li, .stMarkdown ol li {
        margin-bottom: 5px;
        font-weight: normal;
    }
    
    /* Make emojis smaller and less intrusive */
    .stMarkdown p:contains('📋'), 
    .stMarkdown p:contains('🔍'),
    .stMarkdown p:contains('🛠️'),
    .stMarkdown p:contains('📌') {
        font-size: 0.9em;
    }
    
    /* Bottom input positioning */
    .main .block-container {
        padding-bottom: 6rem;
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 2rem);
    }
    
    /* This creates a sticky footer effect for the input */
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    
    /* Ensure content doesn't get hidden behind the sticky input */
    .main-content {
        margin-bottom: 4rem;
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"  # Default URL when running locally

# Load API key from environment or .env file
def get_api_key():
    return os.environ.get("API_KEY", "default-api-key")

# Session state initialization
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status_codes" not in st.session_state:
    st.session_state.status_codes = {}  # Changed from list to dict

# UI Components - Header
st.title("🤖 NetBackup Assistant")
st.markdown("NetBackup 시스템에 관한 질문을 한국어로 해보세요. 기술 정보를 찾고 문제를 해결하는 데 도움을 드립니다.")

# Sidebar
with st.sidebar:
    st.header("설정")
    
    # Status code search
    st.subheader("상태 코드로 검색")
    status_code = st.text_input("검색할 상태 코드를 입력하세요:")
    status_query = st.text_input("상태 코드와 관련된 질문:")
    
    if st.button("상태 코드 검색", key="status_search"):
        if status_code and status_query:
            with st.spinner("검색 중..."):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/query/vectorSimilaritySearch",
                        params={"query": status_query, "status_code": status_code},
                        headers={"X-API-Key": get_api_key()}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.status_codes = result
                        st.success(f"상태 코드 {status_code}에 대한 검색 결과를 찾았습니다!")
                    else:
                        st.error(f"오류 발생: {response.text}")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {str(e)}")

    # New conversation button
    if st.button("새 대화 시작"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # About section
    st.markdown("---")
    st.markdown("### 정보")
    st.markdown("이 앱은 NetBackup 문서를 기반으로 질문에 답변합니다.")
    st.markdown("© 2025 NetBackup Assistant")

# Create two main sections - one for the content (messages) and one for the input
main_content = st.container()
chat_input_container = st.container()

# Display status code search results if available
with main_content:
    if st.session_state.status_codes:
        with st.expander("상태 코드 검색 결과", expanded=True):
            st.markdown(f"### 상태 코드: {st.session_state.status_codes.get('status_code', '')}")
            
            # Summary section
            summary = st.session_state.status_codes.get('summary', '요약 정보가 없습니다.')
            
            # General text cleaning approach
            # 1. Format file paths and command syntax as code for better readability
            summary = re.sub(r'<install_path>[^<>]*', lambda m: f"`{m.group(0)}`", summary)
            summary = re.sub(r'/[a-zA-Z0-9/\._-]+\.log\b', lambda m: f"`{m.group(0)}`", summary)
            
            # 2. Format SQL Server and NetBackup terms consistently
            summary = re.sub(r'\bSQL Server\b', "**SQL Server**", summary)
            summary = re.sub(r'\bNetBackup\b', "**NetBackup**", summary)
            
            st.markdown("#### 요약")
            st.markdown(summary)
            
            # Related documents section
            st.markdown("#### 관련 문서")
            
            # Properly indented inside the status_codes if statement block
            results = st.session_state.status_codes.get('results', [])
            if results:
                for idx, result in enumerate(results):
                    # Create a container for each document with better formatting
                    with st.container():
                        # Create columns for better layout
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Document number and icon
                            st.markdown(f"**관련 문서 {idx+1}**")
                            
                            # Get and display metadata
                            metadata = result.get('metadata', {})
                            
                            # Get filename with fallback
                            filename = result.get('filename', '')
                            if not filename or filename.lower() == 'unknown':
                                filename = metadata.get('source', f"문서 {idx+1}")
                                
                            # Display file information
                            st.markdown(f"**파일명:** {filename}")
                            
                            # Display file type if available
                            file_type = metadata.get('file_type', '')
                            if file_type:
                                st.markdown(f"**파일 유형:** {file_type}")
                            
                            # Display URL as a clickable link if available
                            url = metadata.get('url', '')
                            if url:
                                st.markdown(f"[문서 열기]({url})")
                            
                            # Display creation date if available
                            created = metadata.get('created', '')
                            if created:
                                st.markdown(f"**생성일:** {created}")
                        
                        with col2:
                            # Get snippet
                            snippet = result.get('snippet', '')
                            if not snippet or len(snippet.strip()) == 0:
                                snippet = "문서 내용이 없거나 추출할 수 없습니다."
                            
                            # Format file paths as code
                            snippet = re.sub(r'<install_path>[^<>]*', lambda m: f"`{m.group(0)}`", snippet)
                            snippet = re.sub(r'/[a-zA-Z0-9/\._-]+\.log\b', lambda m: f"`{m.group(0)}`", snippet)
                            
                            # Format NetBackup terms consistently
                            snippet = re.sub(r'\bNetBackup\b', "**NetBackup**", snippet)
                            
                            # Display snippet content
                            st.markdown(snippet)
                        
                        # Add divider between documents
                        st.markdown("---")
            else:
                st.markdown("관련 문서가 없습니다.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Create a placeholder for the chat input at the bottom
with chat_input_container:
    # Add some visual separation 
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    
    # Chat input - using the same implementation but in the bottom container
    prompt = st.chat_input("질문을 입력하세요...")
    
    # Close the container div
    st.markdown("</div>", unsafe_allow_html=True)

# Process user input - keep the existing processing logic
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in the main content area
    with main_content:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Display assistant message with a loading spinner
    with main_content:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            try:
                with st.spinner("답변 생성 중..."):
                    # Make streaming request to API
                    response = requests.get(
                        f"{API_BASE_URL}/query/stream-get",
                        params={
                            "query": prompt, 
                            "conversation_id": st.session_state.conversation_id
                        },
                        headers={"X-API-Key": get_api_key()},
                        stream=True
                    )
                    
                    # Check for errors
                    if response.status_code != 200:
                        st.error(f"API 오류: {response.status_code}")
                        st.session_state.messages.append({"role": "assistant", "content": f"오류가 발생했습니다. 다시 시도해 주세요."})
                        st.stop()
                    
                    # Get new conversation ID if provided in headers
                    if "X-Conversation-ID" in response.headers:
                        st.session_state.conversation_id = response.headers["X-Conversation-ID"]
                    
                    # Process streaming response
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            chunk_text = chunk.decode('utf-8')
                            full_response += chunk_text
                            
                            # Use a simple cursor that's less distracting
                            display_text = full_response + "▌"
                            
                            # Apply minimal spacing improvements for better readability
                            # without excessive formatting
                            improved_display = display_text
                            
                            # Use more subtle formatting for headers - no extra newlines
                            # This prevents the text from jumping around too much
                            improved_display = re.sub(r'(#{1,3}\s+.+)$', r'\1', improved_display, flags=re.MULTILINE)
                            
                            message_placeholder.markdown(improved_display)
                            time.sleep(0.01)  # Very small delay for smoother updates
                            
                    # Final display - more subtle formatting
                    message_placeholder.markdown(full_response)
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                full_response = f"오류가 발생했습니다. 다시 시도해 주세요. 상세: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Rerun the app to update the UI with the new messages
    st.rerun()