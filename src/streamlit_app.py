# src/streamlit_app.py
import streamlit as st
import requests
import uuid
import json
import os
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="NetBackup Assistant",
    page_icon="🤖",
    layout="wide",
)

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
    st.session_state.status_codes = []

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

# Main chat interface
chat_container = st.container()

# Display status code search results if available
if st.session_state.status_codes:
    with st.expander("상태 코드 검색 결과", expanded=True):
        st.markdown(f"### 상태 코드: {st.session_state.status_codes.get('status_code', '')}")
        st.markdown("#### 요약")
        st.markdown(st.session_state.status_codes.get('summary', '요약 정보가 없습니다.'))
        
        st.markdown("#### 관련 문서")
        results = st.session_state.status_codes.get('results', [])
        if results:
            for idx, result in enumerate(results):
                st.markdown(f"**문서 {idx+1}**: {result.get('filename', '알 수 없음')}")
                st.markdown(f"{result.get('snippet', '')}")
                st.markdown("---")
        else:
            st.markdown("관련 문서가 없습니다.")

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("질문을 입력하세요..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant message with a loading spinner
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
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final display without cursor
                    message_placeholder.markdown(full_response)
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                full_response = f"오류가 발생했습니다. 다시 시도해 주세요. 상세: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})