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

# Custom CSS (unchanged)
st.markdown("""
<style>
    /* Base text styling */
    .stMarkdown p {
        margin-bottom: 10px;
        font-weight: normal;
        font-size: 16px;
        line-height: 1.5;
    }
    /* ... rest of your CSS remains unchanged ... */
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
    st.session_state.status_codes = {}  # Already a dict, no change needed

# UI Components - Header
st.title("🤖 NetBackup Assistant")
st.markdown("NetBackup 시스템에 관한 질문을 한국어로 해보세요. 기술 정보를 찾고 문제를 해결하는 데 도움을 드립니다.")

# Sidebar
with st.sidebar:
    st.header("설정")
    
    # Status code search
    st.subheader("상태 코드로 검색")
    status_code = st.text_input("검색할 상태 코드를 입력하세요:", key="status_code_input")
    status_query = st.text_input("상태 코드와 관련된 질문 (선택 사항):", key="status_query_input")
    
    if st.button("상태 코드 검색", key="status_search"):
        if status_code:  # Only require status_code
            with st.spinner("검색 중..."):
                try:
                    # Prepare params; query is optional
                    params = {"status_code": status_code}
                    if status_query.strip():  # Only add query if provided
                        params["query"] = status_query
                    
                    response = requests.get(
                        f"{API_BASE_URL}/query/vectorSimilaritySearch",
                        params=params,
                        headers={"X-API-Key": get_api_key()}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.status_codes = result
                        if status_query:
                            st.success(f"상태 코드 {status_code}에 대한 '{status_query}' 검색 결과를 찾았습니다!")
                        else:
                            st.success(f"상태 코드 {status_code}에 대한 전체 요약을 찾았습니다!")
                    else:
                        st.error(f"오류 발생: {response.text}")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {str(e)}")
        else:
            st.error("상태 코드를 입력하세요.")

    # New conversation button (unchanged)
    if st.button("새 대화 시작"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # About section (unchanged)
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
            query = st.session_state.status_codes.get('query', None)
            
            # General text cleaning approach (unchanged)
            summary = re.sub(r'<install_path>[^<>]*', lambda m: f"`{m.group(0)}`", summary)
            summary = re.sub(r'/[a-zA-Z0-9/\._-]+\.log\b', lambda m: f"`{m.group(0)}`", summary)
            summary = re.sub(r'\bSQL Server\b', "**SQL Server**", summary)
            summary = re.sub(r'\bNetBackup\b', "**NetBackup**", summary)
            
            st.markdown(f"#### 요약{' (' + query + ')' if query else ''}")
            st.markdown(summary)
            
            # Related documents section (unchanged)
            st.markdown("#### 관련 문서")
            results = st.session_state.status_codes.get('results', [])
            if results:
                for idx, result in enumerate(results):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**관련 문서 {idx+1}**")
                            metadata = result.get('metadata', {})
                            filename = result.get('filename', '') or metadata.get('source', f"문서 {idx+1}")
                            st.markdown(f"**파일명:** {filename}")
                            file_type = metadata.get('file_type', '')
                            if file_type:
                                st.markdown(f"**파일 유형:** {file_type}")
                            url = metadata.get('url', '')
                            if url:
                                st.markdown(f"[문서 열기]({url})")
                            created = metadata.get('created', '')
                            if created:
                                st.markdown(f"**생성일:** {created}")
                        with col2:
                            snippet = result.get('snippet', '문서 내용이 없거나 추출할 수 없습니다.')
                            snippet = re.sub(r'<install_path>[^<>]*', lambda m: f"`{m.group(0)}`", snippet)
                            snippet = re.sub(r'/[a-zA-Z0-9/\._-]+\.log\b', lambda m: f"`{m.group(0)}`", snippet)
                            snippet = re.sub(r'\bNetBackup\b', "**NetBackup**", snippet)
                            st.markdown(snippet)
                        st.markdown("---")
            else:
                st.markdown("관련 문서가 없습니다.")
    
    # Display chat messages (unchanged)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input (unchanged)
with chat_input_container:
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    prompt = st.chat_input("질문을 입력하세요...")
    st.markdown("</div>", unsafe_allow_html=True)

# Process user input (unchanged)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with main_content:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("답변 생성 중..."):
                    response = requests.get(
                        f"{API_BASE_URL}/query/stream-get",
                        params={
                            "query": prompt,
                            "conversation_id": st.session_state.conversation_id
                        },
                        headers={"X-API-Key": get_api_key()},
                        stream=True
                    )
                    if response.status_code != 200:
                        st.error(f"API 오류: {response.status_code}")
                        st.session_state.messages.append({"role": "assistant", "content": f"오류가 발생했습니다. 다시 시도해 주세요."})
                        st.stop()
                    if "X-Conversation-ID" in response.headers:
                        st.session_state.conversation_id = response.headers["X-Conversation-ID"]
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            chunk_text = chunk.decode('utf-8')
                            full_response += chunk_text
                            display_text = full_response + "▌"
                            improved_display = re.sub(r'(#{1,3}\s+.+)$', r'\1', display_text, flags=re.MULTILINE)
                            message_placeholder.markdown(improved_display)
                            time.sleep(0.01)
                    message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                full_response = f"오류가 발생했습니다. 다시 시도해 주세요. 상세: {str(e)}"
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()