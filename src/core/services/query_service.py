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
        self.tavily_search = False
        self.tavily_enabled = False
        self.batch_manager = BatchInferenceManager(batch_interval=0.1, max_batch_size=5)
        self.analysis_batch_manager = BatchInferenceManager(
            batch_interval=0.1,
            max_batch_size=5,
            model="mistral:latest"
        )
        self.conversation_histories = {}

        # RL Components
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        self.embedding_dim = 1024  # Adjust if mxbai-embed-large output size differs
        self.policy_net = PolicyNetwork(input_dim=self.embedding_dim + 5 * self.embedding_dim,
                                       hidden_dim=128,
                                       output_dim=5)  # Select from top 5 chunks
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.top_k = 5

    @property
    def vector_store(self):
        vs = get_vector_store()
        if not vs:
            raise ValueError("Vector store not initialized")
        return vs

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both embeddings have zero norm, returning 0 similarity")
            return 0.0
        return dot_product / (norm1 * norm2)

    def _setup_workflow(self):
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}
        self.workflow.add_node("model", call_model)
        self.workflow.add_edge(START, "model")

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

    def get_relevant_chunks(self, query: str):
        """RL-enhanced chunk retrieval while preserving original retrieval logic."""
        query_embedding = np.array(self.embedding_model.embed_query(query))
        docs = self.vector_store.similarity_search(query, k=self.top_k)  # Original retrieval
        
        # RL-based chunk selection
        state = self._get_state(query_embedding, docs)
        action_probs = self.policy_net(torch.tensor(state, dtype=torch.float32))
        action = torch.multinomial(action_probs, 1).item()
        selected_chunks = [docs[action]]  # RL selects one chunk

        # Fallback to all chunks if RL confidence is low
        return selected_chunks if torch.max(action_probs) > 0.5 else docs

    def _get_state(self, query_embedding: np.ndarray, chunks: list) -> np.ndarray:
        """Generate state for RL policy network."""
        chunk_embeddings = [np.array(self.embedding_model.embed_query(chunk.page_content)) for chunk in chunks]
        if len(chunk_embeddings) < self.top_k:
            padding = np.zeros((self.top_k - len(chunk_embeddings), self.embedding_dim))
            chunk_embeddings = chunk_embeddings + [padding] * (self.top_k - len(chunk_embeddings))
        chunk_embeddings = np.vstack(chunk_embeddings)
        return np.concatenate((query_embedding, chunk_embeddings.flatten()))

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
        try:
            if not query.strip():
                raise ValueError("Query text is empty.")
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            if conversation_id not in self.conversation_histories:
                self.conversation_histories[conversation_id] = []
            history = self._get_recent_messages(conversation_id)
            
            retrieval_query = query
            if history:
                user_msgs = [msg for msg in history if isinstance(msg, HumanMessage)][-2:]
                history_text = " ".join([msg.content for msg in user_msgs])
                retrieval_query = f"{history_text} {query}"
            
            retrieval_query = self._enhance_query(retrieval_query)
            
            # Use RL-enhanced retrieval
            docs = self.get_relevant_chunks(retrieval_query)
            grouped_docs = self._group_chunks_by_source(docs, query)
            context = "\n\n".join([f"Document: {source}\n{content}" for source, content in grouped_docs])

            if not context.strip():
                logger.info(f"No documents found for query: {query}")
                no_context_response = AIMessage(content="현재 데이터베이스에 관련 정보가 없습니다. NetBackup 문서를 추가해 주세요.")
                self.conversation_histories[conversation_id].append(HumanMessage(content=query))
                self.conversation_histories[conversation_id].append(no_context_response)
                max_messages = 8
                if len(self.conversation_histories[conversation_id]) > max_messages:
                    self.conversation_histories[conversation_id] = self.conversation_histories[conversation_id][-max_messages:]
                token_handler = AsyncTokenStreamHandler()
                streaming_llm = AsyncPreGeneratedLLM(no_context_response, token_handler, chunk_size=1)
                return streaming_llm, [], conversation_id

            history_for_prompt = self._format_history_for_prompt(history)
            korean_instruction = """당신은 NetBackup 시스템 전문가입니다. 
            반드시 한국어로 명확하게 답변하세요. 기술 용어만 영어로 유지하세요.
            대화의 맥락을 유지하고 이전 대화를 참조하여 일관성 있는 답변을 제공하세요.
            사용자가 이전 질문이나 답변을 언급할 때는 그 맥락을 이해하고 적절히 응답하세요."""
            query_with_context = f"대화 기록:\n{history_for_prompt}\n\n문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:"
            
            current_messages = [
                SystemMessage(content=korean_instruction),
                HumanMessage(content=query_with_context)
            ]
            self.conversation_histories[conversation_id].append(HumanMessage(content=query))
            
            response_future = await self.batch_manager.submit_request(
                query=query,
                context=context,
                messages=current_messages,
                conversation_id=conversation_id
            )
            result = await response_future
            self.conversation_histories[conversation_id].append(AIMessage(content=result.content))

            max_messages = 8
            if len(self.conversation_histories[conversation_id]) > max_messages:
                self.conversation_histories[conversation_id] = self.conversation_histories[conversation_id][-max_messages:]

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

    async def search_by_vector(self, query: str, status_code: str):
        try:
            logger.debug(f"Searching for query: {query} with status code: {status_code}")
            matching_filter = {"status_code": {"$eq": status_code}}
            docs = self.vector_store.similarity_search(query, k=5, filter=matching_filter)
            results = []
            for doc in docs:
                snippet = self._get_snippet_with_keyword(doc.page_content, query)
                if snippet:
                    results.append({
                        "filename": doc.metadata.get("filename", "unknown"),
                        "snippet": snippet,
                        "score": doc.metadata.get("score", 0.0)
                    })
            if not results:
                return {
                    "status_code": status_code,
                    "results": [],
                    "summary": "죄송합니다. 요청하신 검색어와 관련된 문서 내용을 찾을 수 없습니다."
                }
            summary = await self._generate_analysis("\n".join(doc.page_content for doc in docs), query, status_code)
            return {
                "status_code": status_code,
                "results": results,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error in search_by_vector: {e}")
            raise

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