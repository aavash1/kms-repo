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
from src.core.services.file_utils import (get_vector_store, get_personal_vector_store)
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
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import re
import textwrap

logger = logging.getLogger(__name__)

class AsyncTokenStreamHandler(BaseCallbackHandler):
    """Callback handler that pushes streamed tokens into an asyncio.Queue."""
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_llm_new_token(self, token: str, **_) -> None:  
        self.queue.put_nowait(token)

    def on_llm_end(self, *_):  # type: ignore[override]
        self.queue.put_nowait(None)

    async def stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

class AsyncPreGeneratedLLM:
    """A fake-LLM wrapper that mimics the LangChain streaming interface but simply
    yields the already-generated `AIMessage` in small chunks.
    """
    def __init__(
        self,
        result: AIMessage,
        token_handler: AsyncTokenStreamHandler,
        *,
        chunk_size: int = 12,
    ) -> None:
        self._result = result
        self._handler = token_handler
        self._chunk_size = max(chunk_size, 1)
            
    async def astream(self, _messages):
        """Stream the pre-generated result."""
        content = self._result.content
        for chunk in self._split_content_into_chunks(content):
            await self._handler.queue.put(chunk)
        await self._handler.queue.put(None)
        async for token in self._handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i : i + self._chunk_size])
            for i in range(0, len(words), self._chunk_size)
        ]

class MemoryStore:
    def __init__(self, result, token_handler):
        self._result = result
        self._token_handler = token_handler
    
    async def astream(self, _messages):
        content = self._result.content
        for chunk in self._split_content_into_chunks(content):
            await self._token_handler.queue.put(chunk)
        await self._token_handler.queue.put(None)
        async for token in self._token_handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, text, chunk_size=4):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

class QueryService:
    """
    Central RAG+RL service. Supports both
    *knowledge-base* (shared) and *chat_files* (per-chat) collections.
    """
    def __init__(self, translator, rag_chain, global_prompt: str):
        if not global_prompt:
            raise ValueError("Global prompt cannot be None")
        
        # -- Static Components ----
        self.translator = translator
        self.rag_chain = rag_chain
        self.global_prompt = global_prompt
        self.app = self.rag_chain
        
        self.llm = ChatOllama(model="gemma3:12b", temperature=0.1, stream=True)
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        
        self.kb_store       = get_vector_store() #shared knowledge-base
        self.chat_store     = get_personal_vector_store() #new isolated chat store

        # Determine actual embedding dimension
        test_vec= self.embedding_model.embed_query("test")
        self.embedding_dim = len(test_vec)  # Dynamically set embedding dimension
        #logger.info(f"Embedding dimension set to {self.embedding_dim}")


        # # Embedding cache configuration
        self.embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.embedding_cache_max_size = 1_000
        self._cache_hits = 0
        self._cache_requests = 0
        
        self.batch_manager = BatchInferenceManager(batch_interval=0.1, max_batch_size=6,model="gemma3:12b", quantization="Q4_0")
        self.analysis_batch_manager = BatchInferenceManager(
            batch_interval=0.1,
            max_batch_size=6,
            model="gemma3:4b",
            quantization="Q4_0"
        )
        
        # RL Components
        #self.embedding_dim = 1024  # Adjust if mxbai-embed-large output size differs
        self.top_k = 5
        self.policy_net = PolicyNetwork(
            input_dim=self.embedding_dim * (self.top_k + 1),
                                       hidden_dim=128,
                                       output_dim=self.top_k)  # Select from top 5 chunks
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.reward_history: List[Tuple[np.ndarray, int, float]] = []
        

        # Chats & cleanup
        self.conversation_histories: Dict[str, List[AIMessage | HumanMessage]] = {}
        self.conversation_last_access: Dict[str, float] = {}
        self.cleanup_interval = 3600  # 1 h
        self.max_conversation_age = 24 * 3600  # 24 h
        self.last_cleanup = time.time()
        self.read_lock = asyncio.Lock()  
        


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

    def _get_store(self, filter_document_id: Optional[str]) :
        """
        Returns the correct LangChain/Chroma VectorStore.

        * If *filter_document_id* is given we are inside a “chat-with-file”
          session – search only in the private *chat_files* collection.
        * Otherwise fall back to the corporate knowledge-base store.
        """
        return self.chat_store if filter_document_id else self.kb_store  

    
    def _cache_embedding(self, content: str, embedding: np.ndarray) -> None:
        content_hash = self._hash_content(content)
        if len(self.embedding_cache) >= self.embedding_cache_max_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[content_hash] = embedding
        self.embedding_cache.move_to_end(content_hash)
        logger.debug(f"Cached embedding for content: {content[:30]}...")    

    def get_relevant_chunks(self,query: str,*,filter_document_id: Optional[str] = None) -> Tuple[List, List, str]:
        """Search the appropriate store and apply RL post-filtering."""
        store = self._get_store(filter_document_id)                   
        if store is None:
            logger.warning("Vector store is empty or not initialized.")
            return [], [], "Unknown Document"

        try:
            # Embed the translated query with caching
            cached_query_embedding = self._get_cached_embedding(query)
            if cached_query_embedding is not None:
                query_embedding = cached_query_embedding
            else:
                query_embedding = np.array(self.embedding_model.embed_query(query))
                self._cache_embedding(query, query_embedding)

            # Optional metadata filter for per-chat uploads
            md_filter = ({"document_id": filter_document_id}
                     if filter_document_id else None)
            docs = store.similarity_search(query, k=self.top_k, filter=md_filter)
            if not docs:
                return [], [], "Unknown Document"

            # RL-based chunk selection
            state = self._get_state(query_embedding, docs)
            action_probs = self.policy_net(torch.tensor(state, dtype=torch.float32))
            action = torch.multinomial(action_probs, 1).item()
            rl_selected_chunks = [docs[action]] if torch.max(action_probs) > 0.5 else docs

            # Filter for explanatory content (heuristic: longer text, fewer numbers)
            selected_chunks = [
                doc for doc in rl_selected_chunks
                if len(doc.page_content) > 50 and sum(c.isdigit() for c in doc.page_content) / len(doc.page_content) < 0.1
            ] if rl_selected_chunks else []

            # Extract source_name from metadata
            source_name = docs[0].metadata.get("filename", "Unknown Document")

            return docs, selected_chunks, source_name

        except Exception as e:
            logger.error(f"Error in get_relevant_chunks: {e}", exc_info=True)
            return [], [], "Unknown Document"
    
    def _get_state(self, query_embedding: np.ndarray, chunks: List) -> np.ndarray:
        """
        Concatenate the query embedding and *exactly* `top_k` doc-embeddings
        (zero-padded) so the RL network always sees a fixed-size input.
        """
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

    def _get_recent_messages(self, conversation_id, max_messages=8):
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = []
        history = self.conversation_histories[conversation_id]
        return history[-max_messages:] if len(history) > max_messages else history        
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both embeddings have zero norm, returning 0 similarity")
            return 0.0
        return dot_product / (norm1 * norm2)

  
    
    def _format_history_for_prompt(self, messages: List)->str:
        history_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content
                assistant_msg = messages[i + 1].content
                history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
        return "\n\n".join(history_pairs)        

    async def process_basic_query(self, query: str):
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


    def _format_response(
        self,
        content: str,
        *,
        is_korean: bool = True,
        section_labels: Optional[List[str]] = None,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        line_width: int = 90,
        source_name: str = "Unknown Document",
        document_type: str = "unknown"
    ) -> str:
        """
        Clean an LLM answer to render a concise, clear Korean response tailored to document type.

        Parameters
        ----------
        content : str
            Raw model output.
        is_korean : bool, default True
            Selects Korean or English defaults.
        section_labels : list[str], optional
            Ordered headings (first = overview). Overrides document_type defaults.
        keyword_map : dict[str, list[str]], optional
            Mapping from heading to trigger words. Overrides document_type defaults.
        line_width : int, default 90
            Soft-wrap width for paragraphs.
        source_name : str, default "Unknown Document"
            Document source for citation.
        document_type : str, default "unknown"
            Document type (e.g., troubleshooting, contract) for formatting rules.
        """
        content = content.strip()
        if not content:
            return "정보를 찾거나 요약할 수 없습니다."

        # Document-type-specific configurations
        doc_configs = {
            "troubleshooting": {
                "section_labels": ["문제", "원인", "해결 방안", "참고"],
                "keyword_map": {
                    "원인": ["원인", "이유", "문제의 원인", "problem", "cause"],
                    "해결 방안": ["해결", "방안", "해결 방법", "solution", "fix", "resolve"],
                    "참고": ["참고", "추가 정보", "note", "reference"]
                },
                "terms": ["ERROR", "LOG", "CONFIGURATION", "DEBUG"],
                "code_patterns": [r'\b(ping|ssh|grep|awk|telnet)\b', r'(\b[A-Za-z]:\\[^ \n]*?|[^ \n]*?\.log\b)']
            },
            "contract": {
                "section_labels": ["질문", "계약 조항", "의무", "참고"],
                "keyword_map": {
                    "계약 조항": ["조항", "계약", "clause", "contract"],
                    "의무": ["의무", "책임", "obligation", "duty"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["CONTRACT", "CLAUSE", "OBLIGATION", "PARTY"],
                "code_patterns": []
            },
            "memo": {
                "section_labels": ["질문", "내용", "행동", "참고"],
                "keyword_map": {
                    "내용": ["내용", "메모", "memo", "content"],
                    "행동": ["행동", "조치", "action", "step"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["MEMO", "ACTION", "MEETING"],
                "code_patterns": []
            },
            "wbs": {
                "section_labels": ["질문", "작업 내역", "일정", "참고"],
                "keyword_map": {
                    "작업 내역": ["작업", "wbs", "task", "work"],
                    "일정": ["일정", "마일스톤", "schedule", "milestone"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["WBS", "TASK", "MILESTONE", "DELIVERABLE"],
                "code_patterns": []
            },
            "rnr": {
                "section_labels": ["질문", "요구사항", "구현", "참고"],
                "keyword_map": {
                    "요구사항": ["요구", "요구사항", "requirement", "spec"],
                    "구현": ["구현", "실행", "implementation", "execute"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["REQUIREMENT", "FEATURE", "SPECIFICATION"],
                "code_patterns": []
            },
            "proposal": {
                "section_labels": ["질문", "제안 내용", "실행 계획", "참고"],
                "keyword_map": {
                    "제안 내용": ["제안", "내용", "proposal", "objective"],
                    "실행 계획": ["계획", "실행", "plan", "strategy"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["PROPOSAL", "OBJECTIVE", "STRATEGY", "BUDGET"],
                "code_patterns": [r'(\$.*?\$|\$\$.*?\$\$)']
            },
            "presentation": {
                "section_labels": ["질문", "주요 내용", "다음 단계", "참고"],
                "keyword_map": {
                    "주요 내용": ["내용", "주요", "content", "keypoint"],
                    "다음 단계": ["단계", "다음", "step", "next"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": ["PRESENTATION", "SLIDE", "KEYPOINT"],
                "code_patterns": [r'(\$.*?\$|\$\$.*?\$\$)']
            },
            "unknown": {
                "section_labels": ["질문", "내용", "행동", "참고"],
                "keyword_map": {
                    "내용": ["내용", "content"],
                    "행동": ["행동", "action"],
                    "참고": ["참고", "note", "reference"]
                },
                "terms": [],
                "code_patterns": []
            }
        }

        # Use provided section_labels/keyword_map or document_type defaults
        config = doc_configs.get(document_type.lower(), doc_configs["unknown"])
        section_labels = section_labels or config["section_labels"]
        keyword_map = keyword_map or config["keyword_map"]
        terms = config["terms"]
        code_patterns = config["code_patterns"]

        # Split into sentences for specific question detection
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', content) if s.strip()]
        if not sentences:
            return "정보를 찾거나 요약할 수 없습니다."

        # Detect specific question (short content implies direct answer)
        is_specific = len(sentences) <= 2 and len(content) < 100

        if is_specific:
            # Direct answer for specific questions
            formatted = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
            # Bold document-type-specific terms
            for term in terms:
                formatted = re.sub(rf'\b{term}\b', f'**{term}**', formatted, flags=re.IGNORECASE, count=1)
            # Format code/LaTeX
            for pattern in code_patterns:
                formatted = re.sub(pattern, r'`\1`', formatted, flags=re.IGNORECASE)
            return formatted

        # Strip rogue markdown heading tokens
        content = re.sub(r'^\s*#+\s*', '', content, flags=re.MULTILINE)

        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]
        if not paragraphs:
            return "정보를 찾거나 요약할 수 없습니다."

        # Initialize Markdown output
        md = [f"## 📋 {section_labels[0]}:", textwrap.fill(paragraphs[0], line_width), ""]

        buckets = {lbl: [] for lbl in section_labels[1:]}
        current = section_labels[1] if len(section_labels) > 1 else None

        # Pre-compile cue-stripping regex
        cue_regex = {
            sec: re.compile(
                r"^(?:#+\s*)?(" + "|".join(map(re.escape, cues)) + r")\s*[:\-]?\s*", re.IGNORECASE
            )
            for sec, cues in keyword_map.items()
        }

        # Route paragraphs into buckets
        for para in paragraphs[1:]:
            plain = para.lstrip("# ").strip()
            lowered = plain.lower()

            hit = next(
                (sec for sec, cues in keyword_map.items()
                if any(lowered.startswith(c.lower()) for c in cues)),
                None,
            )

            if hit:
                current = hit
                plain = cue_regex[hit].sub("", plain).strip()

            # Normalize bullet markers
            if plain.startswith("* "):
                plain = "- " + plain[2:]

            # Re-wrap regular text; keep lists untouched
            if not plain.startswith(("-", "1.", "2.", "3.")):
                plain = textwrap.fill(plain, line_width)

            if current in buckets:
                buckets[current].append(plain)
            else:
                md.append(plain)  # Unexpected → keep in overview

        # Emit buckets
        for sec in section_labels[1:]:
            md += [f"## 🔍 {sec}:" if sec == section_labels[1] else f"## 🛠️ {sec}:" if sec == section_labels[2] else f"## 📌 {sec}:", ""]
            body = "\n".join(buckets[sec]).strip()
            md.append(body if body else "- 정보가 제공되지 않았습니다.")
            md.append("")

        # Add source reference
        md += [f"**출처**: {source_name}", ""]

        out = "\n".join(md).strip()

        # Bold document-type-specific terms
        for term in terms:
            out = re.sub(rf'\b{term}\b', f'**{term}**', out, flags=re.IGNORECASE, count=1)

        # Format code/LaTeX
        for pattern in code_patterns:
            out = re.sub(pattern, r'`\1`', out, flags=re.IGNORECASE)

        # Preserve LaTeX equations
        out = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', out, flags=re.DOTALL)
        out = re.sub(r'\$(.*?)\$', r'$\1$', out)

        return out
    
    async def process_streaming_query(self, query: str, conversation_id: str = None, *, plain_text: bool = False, filter_document_id: str | None = None ):
        """Process streaming query with optimized embedding usage and conversation cleanup."""
        if not query.strip():
                raise ValueError("Query text is empty.")
        if time.time() - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_conversations()
                self.last_cleanup = time.time()
        
        conversation_id = conversation_id or str(uuid.uuid4())
        hist=self.conversation_histories.setdefault(conversation_id,[])
        self.conversation_last_access[conversation_id] = time.time()

        hist_txt = self._format_history_for_prompt(hist)
        retrieval_query = f"{hist_txt} {query}" if hist_txt else query
        
        async with self.read_lock:
            docs,selected_chunks,source_name = self.get_relevant_chunks(retrieval_query, filter_document_id=filter_document_id)
            grouped_docs = self._group_chunks_by_source(docs, query)
            context = "\n\n".join([f"Document: {source}\n{content}" for source, content in grouped_docs])
            document_type = docs[0].metadata.get("document_type", "unknown") if docs else "unknown"
            source_name = source_name if docs else "Unknown Document"

        if not context.strip():
            no_context_response = AIMessage(content="벡터 데이터베이스가 비어 있거나 관련 정보가 없습니다.")
            hist.extend([
                HumanMessage(content=query),
                no_context_response])
            token_handler = AsyncTokenStreamHandler()
            streaming_llm = AsyncPreGeneratedLLM(no_context_response, token_handler, chunk_size=1)
            return streaming_llm, [], conversation_id

        korean_instruction="""You are a smart, RAG‐powered document analysis assistant that can read and answer questions about any company document type listed in the VALID_DOCUMENT_TYPES environment variable (e.g. troubleshooting, contract, memo, wbs, rnr, proposal, presentation). Always reply in clear, professional English.

            When you generate a response, follow these guidelines:

            1. **Start with a one‐sentence summary** of your answer.  
            2. **Use Markdown**:
            - Use `# Heading` or `## Subheading` for structure.  
            - Use bullet points or numbered lists for steps or examples.  
            - Wrap commands, file paths, or code in backticks: ``like_this``.  

            3. **Be conversational but concise**—write as if you were ChatGPT:  
            - Explain jargon in plain terms.  
            - Offer next steps or tips if relevant.  

            4. **If the user’s question is outside of document analysis**, say, “Sure, let me help with that,” and just answer naturally without forcing the template.

            Environment note:  
            - Document types = `os.getenv("VALID_DOCUMENT_TYPES")`  
            - Use those to decide whether to trigger detailed doc‐analysis style or freeform chat."""
            
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
        try:
            result = await response_future
        except TypeError as e:
            if "can't be used in 'await'" in str(e):
                    # It's already a result, not a future
                result = response_future
            else:
                raise

            # Translate the response to Korean if necessary
        is_korean = any(ord(char) > 127 for char in result.content[:100])  # Check for Korean characters
        if not is_korean:
            result.content = await asyncio.to_thread(self.translator.translate_text, result.content)

            # Post-process the response to enforce formatting
        if not plain_text:
            result.content = self._format_response(
                result.content, 
                is_korean=is_korean,
                source_name=source_name,
                document_type=document_type,
                )

            # RL Update: Calculate reward (e.g., based on response length or user feedback later)
        state = self._get_state(np.array(self.embedding_model.embed_query(retrieval_query)), docs)
        action = docs.index(selected_chunks[0]) if len(selected_chunks) == 1 else 0  # Assuming first chunk selected
        reward = len(result.content) / 1000.0  # Simple heuristic: longer response = better (normalize)
        self.reward_history.append((state, action, reward))
        if len(self.reward_history) >= 10:  # Batch update every 10 queries
            self._batch_update_policy()
            
        self.conversation_histories[conversation_id].extend([
                HumanMessage(content=query),
                AIMessage(content=result.content)])
            
        token_handler = AsyncTokenStreamHandler()
        streaming_llm = AsyncPreGeneratedLLM(result, token_handler, chunk_size=1)
        return streaming_llm, current_messages, conversation_id

    
    def _batch_update_policy(self):
        """Batch update RL policy with accumulated rewards."""
        if not self.reward_history:
            return
        self.optimizer.zero_grad()
        for state, action, reward in self.reward_history:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = self.policy_net(state_tensor)
            log_prob = torch.log(action_probs[action])
            loss = -log_prob * reward
            loss.backward()
        self.optimizer.step()
        logger.info("Updated RL policy with batch of rewards")
        self.reward_history.clear()
    
    def save_policy_periodically(self, filepath: str, interval: int = 3600):
        """Save policy every 'interval' seconds."""
        async def save_loop():
            while True:
                await asyncio.sleep(interval)
                self.save_policy(filepath)
                logger.info(f"Saved RL policy to {filepath}")
        asyncio.create_task(save_loop())
    
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
            async with self.read_lock:  # Ensure consistent read during ingestion
                cached_query_embedding = self._get_cached_embedding(query)
                if cached_query_embedding is not None:
                    query_embedding = cached_query_embedding
                else:
                    query_embedding = np.array(self.embedding_model.embed_query(query))
                    self._cache_embedding(query, query_embedding)
                
                # Use RL for result selection
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
                docs = [doc for doc, _ in docs_with_scores]
                state = self._get_state(query_embedding, docs)
                action_probs = self.policy_net(torch.tensor(state, dtype=torch.float32))
                action = torch.multinomial(action_probs, 1).item()
                selected_docs = [docs[action]] if torch.max(action_probs) > 0.5 else docs[:3]  # Top 3 if uncertain
                
                results = [
                    {
                        "id": doc.metadata.get("id", "unknown"),
                        "snippet": self._get_snippet_with_keyword(doc.page_content, query),
                        "relevance": float(docs_with_scores[docs.index(doc)][1]) if doc in docs else 0.5
                    }
                    for doc in selected_docs
                ]
                results.sort(key=lambda x: x['relevance'], reverse=True)

            # RL Update: Simple reward based on result count
            reward = len(results) / 3.0  # Normalize by max expected results (3)
            self.reward_history.append((state, action, reward))
            if len(self.reward_history) >= 10:
                self._batch_update_policy()

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

    async def search_by_vector(self, query: str | None, status_code: str):
        """
        Perform a vector similarity search for documents related to a specific status code.
        If query is None, summarize all data for the status code, grouped by source.

        Args:
            query: Optional search query string. If None, summarizes all data for the status code.
            status_code: The status code to filter the search by.

        Returns:
            dict: Results including status_code, query (if provided), summary, and grouped document results.
        """
        try:
            logger.info(f"Performing vector similarity search for query={query!r}, status_code='{status_code}'")

            docs = []
            if query:  # Query provided: perform similarity search
                # Strategy 1: Try with filter on error_code_nm
                try:
                    filter_docs = self.vector_store.similarity_search(
                        f"Status Code {status_code} {query}",
                        filter={"error_code_nm": status_code},
                        k=5
                    )
                    if filter_docs:
                        logger.info(f"Found {len(filter_docs)} documents using error_code_nm filter")
                        docs = filter_docs
                    else:
                        logger.debug(f"No documents found with filter {{'error_code_nm': '{status_code}'}} in Strategy 1")
                except Exception as e:
                    logger.warning(f"Error with error_code_nm filtered search: {e}")

                # Strategy 2: Try with document_id or logical_nm for files like Excel
                if not docs:
                    try:
                        metadata_filter = {
                            "$or": [
                                {"document_id": {"$contains": status_code}},
                                {"logical_nm": {"$contains": status_code}}
                            ]
                        }
                        file_docs = self.vector_store.similarity_search(
                            f"NetBackup Status Code {status_code} {query}",
                            filter=metadata_filter,
                            k=5
                        )
                        if file_docs:
                            logger.info(f"Found {len(file_docs)} documents using document_id/logical_nm filter")
                            docs = file_docs
                        else:
                            logger.debug(f"No documents found with document_id/logical_nm filter")
                    except Exception as e:
                        logger.warning(f"Error with metadata filtered search: {e}")

                # Strategy 3: Try with status code in query but no filter
                if not docs:
                    try:
                        query_docs = self.vector_store.similarity_search(
                            f"NetBackup Status Code {status_code} {query}",
                            k=10
                        )
                        filtered_docs = [
                            doc for doc in query_docs
                            if any(
                                phrase in doc.page_content
                                for phrase in [
                                    f"Status Code {status_code}",
                                    f"Status Code: {status_code}",
                                    f"Code {status_code}",
                                    f"ErrorCode {status_code}"
                                ]
                            )
                        ]
                        if filtered_docs:
                            logger.info(f"Found {len(filtered_docs)} documents by filtering query results")
                            docs = filtered_docs
                        elif query_docs:
                            logger.info(f"Using top {min(5, len(query_docs))} unfiltered query results as fallback")
                            docs = query_docs[:5]
                        else:
                            logger.debug(f"No documents found in unfiltered search")
                    except Exception as e:
                        logger.warning(f"Error with unfiltered search: {e}")
            else:  # No query: retrieve all documents for the status code
                try:
                    all_docs = self.vector_store.similarity_search(
                        f"Status Code {status_code}",
                        filter={"error_code_nm": status_code},
                        k=100
                    )
                    if all_docs:
                        logger.info(f"Found {len(all_docs)} documents for status_code {status_code} with error_code_nm")
                        docs = all_docs
                    else:
                        # Try document_id or logical_nm
                        metadata_filter = {
                            "$or": [
                                {"document_id": {"$contains": status_code}},
                                {"logical_nm": {"$contains": status_code}}
                            ]
                        }
                        file_docs = self.vector_store.similarity_search(
                            f"Status Code {status_code}",
                            filter=metadata_filter,
                            k=100
                        )
                        if file_docs:
                            logger.info(f"Found {len(file_docs)} documents for status_code {status_code} with metadata filter")
                            docs = file_docs
                        else:
                            logger.info(f"No documents found for status_code {status_code}")
                except Exception as e:
                    logger.warning(f"Error fetching documents for status_code {status_code}: {e}")

            logger.info(f"Total documents found: {len(docs)}")

            # Group documents by source
            grouped_docs = {}
            for doc in docs:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                # Use logical_nm for files, document_id for text, fallback to 'unknown'
                source_key = metadata.get('logical_nm', metadata.get('document_id', 'unknown'))
                if source_key not in grouped_docs:
                    grouped_docs[source_key] = []
                grouped_docs[source_key].append(doc)

            # Process grouped results
            results = []
            for source_key, doc_group in grouped_docs.items():
                first_doc = doc_group[0]
                metadata = first_doc.metadata if hasattr(first_doc, 'metadata') else {}
                
                # Determine source
                if 'logical_nm' in metadata:
                    source = metadata.get('logical_nm', f"File {source_key}")
                    doc_url = metadata.get('url', '')
                else:
                    source = metadata.get('document_id', "Troubleshooting Report Text")
                    doc_url = ""

                # Combine snippets
                combined_content = " ".join(doc.page_content for doc in doc_group)
                snippet = self._get_snippet_with_keyword(combined_content, f"{status_code} {query or ''}")
                if not snippet:
                    snippet = combined_content[:800] + "..." if len(combined_content) > 800 else combined_content

                # Extract metadata
                file_type = metadata.get('file_type', '')
                if not file_type and '.' in source:
                    ext = source.split('.')[-1].lower()
                    if ext in ['xlsx', 'xls', 'pdf', 'docx', 'doc', 'txt', 'log', 'html', 'kb']:
                        file_type = f"{ext.upper()} 파일"
                
                doc_id = metadata.get('id', '') or f"doc-{hashlib.md5(combined_content[:1000].encode()).hexdigest()[:8]}"
                created_date = metadata.get('created', '')
                title = metadata.get('title', source)

                results.append({
                    "filename": source,
                    "snippet": snippet,
                    "metadata": {
                        "source": source,
                        "title": title,
                        "file_type": file_type,
                        "url": doc_url,
                        "path": metadata.get('path', ''),
                        "id": doc_id,
                        "created": created_date,
                        "status_code": metadata.get('error_code_nm', status_code)
                    }
                })
                logger.debug(f"Added grouped result for source: {source}")

            logger.info(f"Processed {len(results)} grouped documents for display")

            # Generate summary
            if docs:
                context = "\n".join(doc.page_content for doc in doc_group)
                summary_response = await self._generate_analysis(context, query or "", status_code)
            else:
                summary_response = f"상태 코드 {status_code}에 대한 정보를 찾을 수 없습니다." if not query else f"상태 코드 {status_code}에 대한 '{query}' 관련 정보를 찾을 수 없습니다."

            return {
                "status_code": status_code,
                "query": query,
                "summary": summary_response,
                "results": results
            }
        except Exception as e:
            logger.error(f"Unexpected error in search_by_vector: {e}", exc_info=True)
            return {
                "status_code": status_code,
                "query": query,
                "summary": f"검색 중 오류 발생: {str(e)}",
                "results": []
            }
    
    async def _generate_analysis(self, content: str, query: str, status_code: str):
        """
        Generate a technical analysis or summary based on document content.

        Args:
            content: The concatenated content of the documents.
            query: The search query (optional; empty string if not provided).
            status_code: The status code to analyze.

        Returns:
            str: The generated summary or analysis.
        """
        try:
            if query:
                # Query-specific analysis
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
            else:
                # Summary of all data for the status code
                prompt = f"""Summarize all available information related to status code {status_code} based on the following documents.
                Document content:
                {content}
                Provide a concise technical summary with the following structure:
                Overview:
                - General description of issues related to status code {status_code}
                Common Causes:
                - Typical reasons for encountering this status code
                Recommended Actions:
                - General troubleshooting or resolution steps
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
                query=query or f"Summary for status code {status_code}",
                context=content,
                messages=messages,
                conversation_id=conversation_id
            )
            try:
                result = await response_future
            except TypeError as e:
                if "can't be used in 'await'" in str(e):
                    # It's already a result, not a future
                    result = response_future
                else:
                    raise
            
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