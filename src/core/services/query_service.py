# src/core/services/query_service.py
import os
import uuid
import asyncio
import ollama
from fastapi import HTTPException
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.core.processing.local_translator import LocalMarianTranslator
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from src.core.services.tavilysearch import TavilySearch
import logging
from src.core.services.file_utils import get_vector_store
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages
from src.core.inference.batch_inference import BatchInferenceManager
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from langchain_ollama import OllamaEmbeddings
from collections import OrderedDict
from typing import Dict, List, Optional
import hashlib
import time

logger = logging.getLogger(__name__)

class AsyncPreGeneratedLLM:
    """A class that mimics the streaming LLM interface but returns a pre-generated response."""
    def __init__(self, result, token_handler, chunk_size):
        self.result = result
        self.token_handler = token_handler
        self.chunk_size = chunk_size
    
    async def astream(self, messages):
        """Stream the pre-generated result."""
        content = self.result.content
        for chunk in self._split_content_into_chunks(content):
            await self.token_handler.queue.put(chunk)
        await self.token_handler.queue.put(None)
        async for token in self.token_handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, content, chunk_size=4):
        if chunk_size <= 1:
            return [char for char in content]
        else:
            words = content.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                chunks.append(chunk)
            return chunks

class AsyncTokenStreamHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens"""
    def __init__(self):
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put_nowait(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put_nowait(None)

    async def stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

class MemoryStore:
    def __init__(self, result, token_handler):
        self.result = result
        self.token_handler = token_handler
    
    async def astream(self, messages):
        content = self.result.content
        for chunk in self._split_content_into_chunks(content):
            await self.token_handler.queue.put(chunk)
        await self.token_handler.queue.put(None)
        async for token in self.token_handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, content, chunk_size=4):
        words = content.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class QueryService:
    def __init__(self, translator, rag_chain, global_prompt):
        if not global_prompt:
            raise ValueError("Global prompt cannot be None")
        self.translator = translator
        self.rag_chain = rag_chain
        self.global_prompt = global_prompt
        self.app = self.rag_chain
        self.llm = ChatOllama(model="gemma3:12b", temperature=0.1, stream=True)
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
              
        # Embedding cache configuration
        self.embedding_cache: OrderedDict = OrderedDict()  # Using OrderedDict for LRU cache
        self.embedding_cache_max_size = 1000  # Maximum number of cached embeddings
        self.embedding_dim = 1024  # Dimension of mxbai-embed-large embeddings
        
        # Cache hit tracking
        self._cache_hits = 0
        self._cache_requests = 0
        
        self.batch_manager = BatchInferenceManager(batch_interval=0.1, max_batch_size=5,model="gemma3:12b")
        self.analysis_batch_manager = BatchInferenceManager(
            batch_interval=0.1,
            max_batch_size=5,
            model="mistral:latest"
        )
        self.conversation_histories = {}

        # RL Components
        self.embedding_dim = 1024  # Adjust if mxbai-embed-large output size differs
        self.policy_net = PolicyNetwork(input_dim=self.embedding_dim + 5 * self.embedding_dim,
                                       hidden_dim=128,
                                       output_dim=5)  # Select from top 5 chunks
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.top_k = 5

        # Conversation history with cleanup tracking
        self.conversation_histories: Dict[str, List] = {}
        self.conversation_last_access: Dict[str, float] = {}  # Track last access time
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour in seconds
        self.max_conversation_age = 24 * 3600  # 24 hours in seconds

        self.tavily_search = False
        self.tavily_enabled = False

        # Initialize workflow
        self.workflow = StateGraph(state_schema=MessagesState)
        self._setup_workflow()

    @property
    def vector_store(self):
        vs = get_vector_store()
        if not vs:
            raise ValueError("Vector store not initialized")
        return vs

    def _hash_content(self, content: str) -> str:
        """Generate a consistent hash for caching embeddings."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cached_embedding(self, content: str) -> Optional[np.ndarray]:
        self._cache_requests += 1
        content_hash = self._hash_content(content)
        if content_hash in self.embedding_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for content: {content[:30]}...")
            return self.embedding_cache[content_hash]
        return None

    def _cache_embedding(self, content: str, embedding: np.ndarray) -> None:
        content_hash = self._hash_content(content)
        if len(self.embedding_cache) >= self.embedding_cache_max_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[content_hash] = embedding
        self.embedding_cache.move_to_end(content_hash)
        logger.debug(f"Cached embedding for content: {content[:30]}...")    

    def get_relevant_chunks(self, query: str) -> List:
        cached_query_embedding = self._get_cached_embedding(query)
        if cached_query_embedding is not None:
            query_embedding = cached_query_embedding
        else:
            query_embedding = np.array(self.embedding_model.embed_query(query))
            self._cache_embedding(query, query_embedding)

        docs = self.vector_store.similarity_search(query, k=self.top_k)
        state = self._get_state(query_embedding, docs)
        action_probs = self.policy_net(torch.tensor(state, dtype=torch.float32))
        action = torch.multinomial(action_probs, 1).item()
        selected_chunks = [docs[action]]
        return selected_chunks if torch.max(action_probs) > 0.5 else docs
    
    def _get_state(self, query_embedding: np.ndarray, chunks: List) -> np.ndarray:
        chunk_embeddings = []
        for chunk in chunks:
            cached_embedding = self._get_cached_embedding(chunk.page_content)
            if cached_embedding is not None:
                chunk_embeddings.append(cached_embedding)
            else:
                embedding = np.array(self.embedding_model.embed_query(chunk.page_content))
                self._cache_embedding(chunk.page_content, embedding)
                chunk_embeddings.append(embedding)

        if len(chunk_embeddings) < self.top_k:
            padding = np.zeros((self.top_k - len(chunk_embeddings), self.embedding_dim))
            chunk_embeddings.extend([padding] * (self.top_k - len(chunk_embeddings)))
        
        chunk_embeddings = np.vstack(chunk_embeddings)
        return np.concatenate((query_embedding, chunk_embeddings.flatten()))
    
    def _setup_workflow(self):
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}
        self.workflow.add_node("model", call_model)
        self.workflow.add_edge(START, "model")
    
    def get_cache_stats(self) -> Dict[str, float]:
        hit_rate = self._cache_hits / (self._cache_requests or 1)
        return {
            "size": len(self.embedding_cache),
            "max_size": self.embedding_cache_max_size,
            "hit_rate": hit_rate,
            "total_requests": self._cache_requests,
            "cache_hits": self._cache_hits
        }
    
    def _cleanup_old_conversations(self):
        """Clean up old conversation histories to prevent memory leaks."""
        current_time = time.time()
        ids_to_remove = [
            conv_id for conv_id, last_access in self.conversation_last_access.items()
            if current_time - last_access > self.max_conversation_age
        ]
        
        for conv_id in ids_to_remove:
            if conv_id in self.conversation_histories:
                del self.conversation_histories[conv_id]
            if conv_id in self.conversation_last_access:
                del self.conversation_last_access[conv_id]
        
        if ids_to_remove:
            logger.info(f"Cleaned up {len(ids_to_remove)} old conversations")
            logger.debug(f"Remaining conversations: {len(self.conversation_histories)}")
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both embeddings have zero norm, returning 0 similarity")
            return 0.0
        return dot_product / (norm1 * norm2)

    def _get_recent_messages(self, conversation_id, max_messages=8):
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = []
        history = self.conversation_histories[conversation_id]
        return history[-max_messages:] if len(history) > max_messages else history
    
    def _format_history_for_prompt(self, messages):
        history_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content
                assistant_msg = messages[i + 1].content
                history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
        return "\n\n".join(history_pairs)        

    async def process_basic_query(self, query: str):
        try:
            if not query.strip():
                raise ValueError("Query text is empty.")
            try:
                docs = self.vector_store.similarity_search(query, k=5)
                context = "\n".join(doc.page_content[:600] for doc in docs)
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                context = ""
            messages = [
                SystemMessage(content="당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 명확하게 답변하세요."),
                HumanMessage(content=f"문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:")
            ]
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Error in process_basic_query: {e}")
            raise

    def update_policy(self, state: np.ndarray, action: int, reward: float):
        """Update RL policy based on reward."""
        self.optimizer.zero_grad()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state_tensor)
        log_prob = torch.log(action_probs[action])
        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()

    def save_policy(self, filepath: str):
        """Save RL policy network state."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load_policy(self, filepath: str):
        """Load RL policy network state."""
        self.policy_net.load_state_dict(torch.load(filepath,weights_only=True))

    async def process_streaming_query(self, query: str, conversation_id: str = None):
        """Process streaming query with optimized embedding usage and conversation cleanup."""
        try:
            # Check if cleanup is needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_conversations()
                self.last_cleanup = current_time

            if not query.strip():
                raise ValueError("Query text is empty.")
            
            conversation_id = conversation_id or str(uuid.uuid4())
            if conversation_id not in self.conversation_histories:
                self.conversation_histories[conversation_id] = []
            
            # Update last access time
            self.conversation_last_access[conversation_id] = time.time()
            
            history = self._get_recent_messages(conversation_id)
            retrieval_query = self._enhance_query(
                f"{self._format_history_for_prompt(history)} {query}"
                if history else query
            )
            
            docs = self.get_relevant_chunks(retrieval_query)
            grouped_docs = self._group_chunks_by_source(docs, query)
            context = "\n\n".join([f"Document: {source}\n{content}" for source, content in grouped_docs])

            if not context.strip():
                no_context_response = AIMessage(content="현재 데이터베이스에 관련 정보가 없습니다.")
                self.conversation_histories[conversation_id].extend([
                    HumanMessage(content=query),
                    no_context_response
                ])
                token_handler = AsyncTokenStreamHandler()
                streaming_llm = AsyncPreGeneratedLLM(no_context_response, token_handler, chunk_size=1)
                return streaming_llm, [], conversation_id

            korean_instruction = """당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 답변하세요.

            답변을 작성할 때 다음 마크다운 형식을 정확히 따르세요:

            1. 주요 섹션 제목은 '## 제목:' 형식으로 작성하세요.
            - 문제 설명은 '## 📋 문제:'
            - 원인 분석은 '## 🔍 원인:'
            - 해결 방안은 '## 🛠️ 해결 방안:'
            - 참고 사항은 '## 📌 참고:'

            2. 중요한 기술 용어는 볼드체(**용어**)로 표시하세요:
            - 예: **NetBackup**, **SQL Server**, **DNS**

            3. 순서가 있는 내용은 번호 목록으로 만드세요:
            1. 첫 번째 단계
            2. 두 번째 단계

            4. 명령어나 파일 경로는 코드 블록으로 표시하세요:
            - 예: `ping 192.168.1.1`

            5. 각 단락 사이에는 빈 줄을 넣어 구분하세요.

            6. 답변은 체계적으로 구조화하고, 사용자가 따라할 수 있는 명확한 단계별 지침을 제공하세요."""
            
            current_messages = [
                SystemMessage(content=korean_instruction),
                HumanMessage(content=f"문맥 정보: {context}\n\n질문: {query}")
            ]
            
            response_future = await self.batch_manager.submit_request(
                query=query,
                context=context,
                messages=current_messages,
                conversation_id=conversation_id
            )
            result = await response_future
            
            self.conversation_histories[conversation_id].extend([
                HumanMessage(content=query),
                AIMessage(content=result.content)
            ])
            
            token_handler = AsyncTokenStreamHandler()
            streaming_llm = AsyncPreGeneratedLLM(result, token_handler, chunk_size=1)
            return streaming_llm, current_messages, conversation_id

        except Exception as e:
            logger.error(f"Error in process_streaming_query: {e}")
            raise
    
    def _enhance_query(self, query):
        if any(term in query.lower() for term in ["how to", "command", "steps", "procedure", "명령어", "단계", "절차"]):
            return f"command procedure steps {query}"
        if any(term in query.lower() for term in ["database", "db", "데이터베이스"]):
            return f"database recovery restore {query}"
        if any(term in query.lower() for term in ["error", "status code", "status", "code", "에러", "상태", "코드"]):
            return f"error status code troubleshooting {query}"
        return query

    def _group_chunks_by_source(self, docs, query):
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        source_groups = {}
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        source_scores = {}
        for source, source_docs in source_groups.items():
            match_count = 0
            for doc in source_docs:
                content = doc.page_content.lower()
                match_count += sum(1 for term in query_terms if term in content)
            source_scores[source] = match_count / len(query_terms) if query_terms else 0
        sorted_sources = sorted(source_groups.keys(), key=lambda s: source_scores.get(s, 0), reverse=True)
        result = []
        for source in sorted_sources[:3]:
            chunks = [doc.page_content for doc in source_groups[source]]
            source_text = "\n".join(chunks)
            result.append((source, source_text))
        return result

    def preprocess_query(query):
        if "how to" in query.lower() or "steps" in query.lower() or "command" in query.lower():
            return f"command steps procedure {query}"
        if "database" in query.lower() or "db" in query.lower():
            return f"database recovery NetBackup command {query}"
        return query

    def clear_conversation(self, conversation_id):
        try:
            if conversation_id in self.conversation_histories:
                del self.conversation_histories[conversation_id]
            if conversation_id in self.conversation_last_access:
                del self.conversation_last_access[conversation_id]
            logger.debug(f"Cleared conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False  
    
    async def perform_similarity_search(self, query: str):
        try:
            docs = self.vector_store.similarity_search_with_score(query, k=3)
            results = [
                {
                    "id": doc.metadata.get("id", "unknown"),
                    "snippet": doc.page_content[:300] + "...",
                    "relevance": float(score)
                } 
                for doc, score in docs
                if score > 0.5
            ]
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return {
                "query": query,
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            logger.error(f"Error in perform_similarity_search: {e}")
            raise

    def _get_snippet_with_keyword(self, content: str, query: str, max_length: int = 500):
        if not query or not content:
            return content[:max_length] + "..."
        query_terms = query.lower().split()
        content_lower = content.lower()
        positions = []
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                positions.append(pos)
        if not positions:
            return content[:max_length] + "..."
        pos = min(positions)
        start = max(0, pos - 200)
        end = min(len(content), pos + 300)
        if start > 0:
            space_before = content.rfind(" ", 0, start)
            if space_before != -1:
                start = space_before + 1
        if end < len(content):
            space_after = content.find(" ", end)
            if space_after != -1:
                end = space_after
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet

    # In src/core/services/query_service.py - Replace the search_by_vector method

    async def search_by_vector(self, query: str, status_code: str):
        """
        Perform a vector similarity search for documents related to a specific status code
        with multiple fallback strategies.
        """
        try:
            logger.info(f"Performing vector similarity search for query='{query}', status_code='{status_code}'")
            
            # Try multiple search strategies
            docs = []
            
            # Strategy 1: Try with filter
            try:
                filter_docs = self.vector_store.similarity_search(
                    f"Status Code {status_code} {query}", 
                    filter={"status_code": status_code},
                    k=5
                )
                if filter_docs:
                    logger.info(f"Found {len(filter_docs)} documents using status_code filter")
                    docs = filter_docs
            except Exception as e:
                logger.warning(f"Error with filtered search: {e}")
            
            # Strategy 2: Try with status code in query but no filter
            if not docs:
                try:
                    query_docs = self.vector_store.similarity_search(
                        f"NetBackup Status Code {status_code} {query}",
                        k=10
                    )
                    # Filter post-query for documents that mention the status code
                    filtered_docs = [
                        doc for doc in query_docs 
                        if f"Status Code {status_code}" in doc.page_content or
                        f"Status Code: {status_code}" in doc.page_content or
                        f"Code {status_code}" in doc.page_content or
                        f"ErrorCode {status_code}" in doc.page_content
                    ]
                    if filtered_docs:
                        logger.info(f"Found {len(filtered_docs)} documents by filtering query results")
                        docs = filtered_docs
                        
                    # If still no docs, take the top query results anyway
                    if not docs and query_docs:
                        logger.info(f"Using top {min(5, len(query_docs))} unfiltered query results as fallback")
                        docs = query_docs[:5]
                except Exception as e:
                    logger.warning(f"Error with unfiltered search: {e}")
            
            logger.info(f"Total documents found: {len(docs)}")
            
            # Process document results with improved metadata
            results = []
            for i, doc in enumerate(docs):
                logger.debug(f"Processing document {i} content preview: {doc.page_content[:100]}")
                
                # Extract metadata
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                logger.debug(f"Document {i} metadata: {metadata}")
                
                # Extract document source information with better fallbacks
                source = metadata.get('source', '')
                if not source:
                    source = metadata.get('filename', '')
                if not source:
                    source = metadata.get('title', '')
                if not source:
                    source = f"NetBackup Document {i+1}"
                    
                logger.debug(f"Document {i} source: {source}")
                
                # Extract document path or URL
                doc_path = metadata.get('path', '')
                doc_url = metadata.get('url', '')
                
                # Extract file type with better detection
                file_type = metadata.get('file_type', '')
                if not file_type and source and '.' in source:
                    ext = source.split('.')[-1].lower()
                    if ext in ['pdf', 'docx', 'doc', 'txt', 'log', 'html', 'kb']:
                        file_type = f"{ext.upper()} 파일"
                
                # Generate a document ID for stable reference
                doc_id = metadata.get('id', '')
                if not doc_id:
                    # Create a stable hash from the content
                    content_hash = hashlib.md5(doc.page_content[:1000].encode()).hexdigest()[:8]
                    doc_id = f"doc-{content_hash}"
                
                # Extract creation date
                created_date = metadata.get('created', '')
                
                # Extract or generate document title
                title = metadata.get('title', '')
                if not title:
                    # Try to extract a title from the first line of content
                    content_lines = doc.page_content.split('\n')
                    if content_lines and len(content_lines[0].strip()) > 0 and len(content_lines[0].strip()) < 100:
                        title = content_lines[0].strip()
                    else:
                        # Use source as title, or status code related title
                        title = source if source else f"Status Code {status_code} 관련 문서"
                
                # Get the most relevant snippet from the document
                snippet = self._get_snippet_with_keyword(doc.page_content, f"{status_code} {query}")
                if not snippet:
                    snippet = doc.page_content[:800] + "..."
                
                # Add to results with enhanced metadata
                results.append({
                    "filename": source,
                    "snippet": snippet,
                    "metadata": {
                        "source": source,
                        "title": title,
                        "file_type": file_type,
                        "url": doc_url,
                        "path": doc_path,
                        "id": doc_id,
                        "created": created_date,
                        "status_code": metadata.get('status_code', status_code)
                    }
                })
                
                logger.debug(f"Added document {i} to results")
            
            logger.info(f"Processed {len(results)} documents for display")
            
            # Generate a summary even if no documents are found
            if docs:
                context = "\n".join(doc.page_content for doc in docs)
                summary_response = await self._generate_analysis(context, query, status_code)
            else:
                # Generate a fallback summary
                summary_response = f"상태 코드 {status_code}에 대한 '{query}' 관련 정보를 찾을 수 없습니다. 다른 검색어로 시도해 보세요."
            
            return {
                "status_code": status_code,
                "query": query,
                "summary": summary_response,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in search_by_vector: {e}", exc_info=True)
            return {
                "status_code": status_code,
                "results": [],
                "summary": f"검색 중 오류가 발생했습니다: {str(e)}",
                "error": str(e)
            }
    
    async def _generate_analysis(self, content: str, query: str, status_code: str):
        try:
            prompt = f"""Analyze the following documents related to status code {status_code}.
            Document content:
            {content}
            Search query: {query}
            Provide a technical analysis with the following structure:
            Problem:
            - Brief description of the issue
            Root Cause:
            - Main causes and implications
            Solution:
            - Recommended actions
            Note: 
            - Keep NetBackup terms, error codes, and commands in English
            - Focus only on information present in the documents
            - Be concise and technical"""
            conversation_id = str(uuid.uuid4())
            messages = [
                SystemMessage(content="You are a NetBackup expert analyzing technical documents."),
                HumanMessage(content=prompt)
            ]
            response_future = await self.analysis_batch_manager.submit_request(
                query=query,
                context=content,
                messages=messages,
                conversation_id=conversation_id
            )
            result = await response_future
            english_summary = result.content
            summary = await asyncio.to_thread(self.translator.translate_text, english_summary)
            return summary
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            try:
                logger.warning(f"Falling back to direct Ollama call for analysis with mistral model")
                response = ollama.chat(
                    model='mistral:latest',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                english_summary = response['message']['content']
                return await asyncio.to_thread(self.translator.translate_text, english_summary)
            except Exception as fallback_error:
                logger.error(f"Error in fallback analysis: {fallback_error}")
                return "AI 분석 중 오류가 발생했습니다. 다시 시도해 주시기 바랍니다."