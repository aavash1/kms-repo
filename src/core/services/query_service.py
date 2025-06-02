# src/core/services/query_service.py

import os
import uuid
import asyncio
import ollama
import logging
import hashlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import textwrap
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from src.core.processing.local_translator import LocalMarianTranslator
from src.core.services.file_utils import get_vector_store, get_personal_vector_store
from src.core.services.dynamic_query_processor import (
    DynamicQueryProcessor,
    EnhancedRLFeatureExtractor
)
from src.core.inference.batch_inference import BatchInferenceManager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


logger = logging.getLogger(__name__)


class AsyncTokenStreamHandler(BaseCallbackHandler):
    """Callback handler that pushes streamed tokens into an asyncio.Queue."""
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_llm_new_token(self, token: str, **_) -> None:
        self.queue.put_nowait(token)

    def on_llm_end(self, *_):
        self.queue.put_nowait(None)

    async def stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token


class AsyncPreGeneratedLLM:
    """Wraps a pre‐generated AIMessage so we can stream it chunk by chunk."""
    def __init__(self, result: AIMessage, token_handler: AsyncTokenStreamHandler, *, chunk_size: int = 12):
        self._result = result
        self._handler = token_handler
        self._chunk_size = max(chunk_size, 1)

    async def astream(self, _messages):
        content = self._result.content
        words = content.split()
        for i in range(0, len(words), self._chunk_size):
            chunk = " ".join(words[i : i + self._chunk_size])
            # Add a trailing space between chunks (except last)
            if i + self._chunk_size < len(words):
                chunk += " "
            await self._handler.queue.put(chunk)
        await self._handler.queue.put(None)
        async for token in self._handler.stream():
            yield AIMessage(content=token)


class MemoryStore:
    """Same as above but smaller chunk size for memory streaming."""
    def __init__(self, result: AIMessage, token_handler: AsyncTokenStreamHandler):
        self._result = result
        self._handler = token_handler

    async def astream(self, _messages):
        content = self._result.content
        words = content.split()
        for i in range(0, len(words), 4):
            chunk = " ".join(words[i : i + 4])
            await self._handler.queue.put(chunk)
        await self._handler.queue.put(None)
        async for token in self._handler.stream():
            yield AIMessage(content=token)


class PolicyNetwork(nn.Module):
    """A simple 2‐layer policy net with softmax on output."""
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
    Central RAG + RL service.  
    Supports both *kb_store* (shared) and *chat_store* (per‐chat) collections.
    """
    def __init__(self, translator: LocalMarianTranslator, rag_chain, global_prompt: str):
        if not global_prompt:
            raise ValueError("Global prompt cannot be None")

        # ----------------------------------------------------------------
        # 1) Static / injected parts
        # ----------------------------------------------------------------
        self.translator = translator
        self.rag_chain = rag_chain
        self.global_prompt = global_prompt
        self.app = self.rag_chain

        # LLM and embedding models
        self.llm = ChatOllama(model="gemma3:12b", temperature=0.1, stream=True)
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

        # ----------------------------------------------------------------
        # 2) Vector stores: try to load & validate both KB and per‐chat
        # ----------------------------------------------------------------
        self.kb_store = self._get_validated_vector_store("kb")
        self.chat_store = self._get_validated_vector_store("chat")

       
        # ----------------------------------------------------------------
        # 3) Dynamic query processor (holds its own `vector_store` + TF‐IDF corpus stats)
        # ----------------------------------------------------------------
        self.dynamic_processor = DynamicQueryProcessor(self.kb_store, self.embedding_model)

        # ----------------------------------------------------------------
        # 4) Embedding dimension & caching
        # ----------------------------------------------------------------
        test_vec = self.embedding_model.embed_query("test")
        self.embedding_dim = len(test_vec)
        self.embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.embedding_cache_max_size = 1000
        self._cache_hits = 0
        self._cache_requests = 0

        # ----------------------------------------------------------------
        # 5) Batch inference managers (for streaming queries vs. analysis)
        # ----------------------------------------------------------------
        self.batch_manager = BatchInferenceManager(
            batch_interval=0.1, max_batch_size=6, model="gemma3:12b", quantization="Q4_0"
        )
        self.analysis_batch_manager = BatchInferenceManager(
            batch_interval=0.1, max_batch_size=6, model="gemma3:4b", quantization="Q4_0"
        )

        # ----------------------------------------------------------------
        # 6) RL policy network + optimizer
        # ----------------------------------------------------------------
        self.top_k = 5
        # Input‐dim = (query_embedding + top_k chunk_embeddings) → (self.embedding_dim * (self.top_k + 1))
        self.policy_net = PolicyNetwork(
            input_dim=self.embedding_dim * (self.top_k + 1),
            hidden_dim=128,
            output_dim=self.top_k
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.reward_history: List[Tuple[np.ndarray, int, float]] = []

        # Also build a dynamic RL feature‐extractor
        self.rl_feature_extractor = EnhancedRLFeatureExtractor(self.dynamic_processor)
        self.experience_buffer = []  # Store up to 100 past experiences

        # ----------------------------------------------------------------
        # 7) Conversation histories & cleanup
        # ----------------------------------------------------------------
        self.conversation_histories: Dict[str, List[AIMessage | HumanMessage]] = {}
        self.conversation_last_access: Dict[str, float] = {}
        self.cleanup_interval = 3600        # once per hour
        self.max_conversation_age = 24 * 3600  # 24h
        self.last_cleanup = time.time()
        self.read_lock = asyncio.Lock()

        # For external search toggles
        self.tavily_search = False
        self.tavily_enabled = False

        # ----------------------------------------------------------------
        # 8) Workflow graph (unchanged from v1)
        # ----------------------------------------------------------------
        self.workflow = StateGraph(state_schema=MessagesState)
        self._setup_workflow()


    # -------------------------
    #  Utility / helper methods
    # -------------------------
    @property
    def vector_store(self):
        """
        If someone calls `self.vector_store` directly, re‐grab the global KB‐store.  
        Note: most code now goes through `_get_store(...)` instead.
        """
        vs = get_vector_store()
        if not vs:
            raise ValueError("Vector store not initialized")
        return vs

    def _get_validated_vector_store(self, store_type: str):
        """
        Load either the shared KB store or per‐chat store, then attempt a quick "count()" or 
        "get(1)" to verify we can talk to Chroma without immediately failing.  
        If Chroma isn’t running or the collection doesn’t exist yet, we return None (but log).
        """
        try:
            if store_type == "kb":
                store = get_vector_store()
            elif store_type == "chat":
                store = get_personal_vector_store()
            else:
                raise ValueError(f"Invalid store_type: {store_type}")

            if not store:
                logger.warning(f"Vector store ({store_type}) not available")
                return None

           
            try:
                if hasattr(store, "_collection"):
                    count = store._collection.count()
                    logger.debug(f"Validated '{store_type}' store with {count} docs")
                elif hasattr(store, "get"):
                    result = store.get(limit=1)
                    logger.debug(f"Validated '{store_type}' store accessibility")
                else:
                    logger.warning(f"Unknown interface for store_type='{store_type}'")
            except Exception as e:
                logger.warning(f"Store validation failed for {store_type}: {e}")
               
            return store

        except Exception as e:
            logger.error(f"Error loading {store_type} vector store: {e}", exc_info=True)
            return None

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cached_embedding(self, content: str) -> Optional[np.ndarray]:
        self._cache_requests += 1
        h = self._hash_content(content)
        if h in self.embedding_cache:
            self._cache_hits += 1
            return self.embedding_cache[h]
        return None

    def _cache_embedding(self, content: str, embedding: np.ndarray) -> None:
        h = self._hash_content(content)
        if len(self.embedding_cache) >= self.embedding_cache_max_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[h] = embedding
        self.embedding_cache.move_to_end(h)

    async def _get_cached_query_embedding(self, query: str) -> np.ndarray:
        """
        Return a cached embedding if available; otherwise embed + cache.
        """
        cached = self._get_cached_embedding(query)
        if cached is not None:
            return cached

        emb = np.array(self.embedding_model.embed_query(query))
        emb = self._ensure_embedding_compatibility(emb)
        self._cache_embedding(query, emb)
        return emb

    def _ensure_embedding_compatibility(self, embedding, expected_dim: Optional[int] = None) -> np.ndarray:
        """
        If Ollama suddenly returns a different embedding size, pad or truncate as needed.
        """
        if expected_dim is None:
            expected_dim = self.embedding_dim

        emb_np = np.asarray(embedding)
        if emb_np.shape[0] != expected_dim:
            logger.warning(f"Embedding dim mismatch: got {emb_np.shape[0]}, expected {expected_dim}")
            if emb_np.shape[0] > expected_dim:
                return emb_np[:expected_dim]
            else:
                padded = np.zeros(expected_dim, dtype=float)
                padded[: emb_np.shape[0]] = emb_np
                return padded
        return emb_np

    def _cleanup_old_conversations(self):
        """Evict any conversation IDs that have not been accessed for > 24h."""
        now = time.time()
        to_delete = []
        for conv_id, last_access in self.conversation_last_access.items():
            if now - last_access > self.max_conversation_age:
                to_delete.append(conv_id)
        for conv_id in to_delete:
            self.conversation_histories.pop(conv_id, None)
            self.conversation_last_access.pop(conv_id, None)
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old conversations")


    def _setup_workflow(self):
        """(unchanged) Builds a simple 1‐node STATE graph that calls the LLM."""
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}
        self.workflow.add_node("model", call_model)
        self.workflow.add_edge(START, "model")


    def _get_store(self, filter_document_id: Optional[str] = None):
        """
        Returns either the per‐chat store (if filter_document_id is set) or the kb_store.  
        If that store is None, we try a fallback. Otherwise we log an error.
        """
        try:
            if filter_document_id and self.chat_store is not None:
                chosen = self.chat_store
                name = "chat"
            elif not filter_document_id and self.kb_store is not None:
                chosen = self.kb_store
                name = "kb"
            else:
                chosen = None
                name = "kb" if not filter_document_id else "chat"

            if chosen is None:
               
                fallback = self.chat_store or self.kb_store
                if fallback:
                    logger.warning(f"Primary '{name}' store is missing → using fallback.")
                    chosen = fallback
                else:
                    raise RuntimeError(f"No vector store available (tried '{name}').")

         
            try:
                cnt = 0
                if hasattr(chosen, "_collection") and hasattr(chosen._collection, "count"):
                    cnt = chosen._collection.count()
                elif hasattr(chosen, "get"):
                    temp = chosen.get(limit=1)
                    cnt = len(temp.get("ids", [])) if temp else 0
                logger.debug(f"Selected '{name}' store has {cnt} documents.")
            except Exception as e:
                logger.warning(f"Cannot verify '{name}' store content: {e}")

            return chosen

        except Exception:
            raise RuntimeError("Vector store selection failed.")


    def _deduplicate_docs(self, docs: List) -> List:
        """Drop exact‐duplicate documents based on first 200 chars of `page_content`."""
        seen = set()
        unique = []
        for d in docs:
            h = hashlib.md5(d.page_content[:200].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(d)
        return unique


    # ----------------------------------
    #  get_relevant_chunks (main retrieval)
    # ----------------------------------
    async def get_relevant_chunks(
        self,
        query: str,
        *,
        filter_document_id: Optional[str] = None
    ) -> Tuple[List, List, str]:
        """
        1) Pick store (kb_store vs. chat_store).
        2) Run dynamic adaptive preprocessing: `processed = await dynamic_processor.adaptive_preprocess_query(query)`.
        3) Gather ~15 “candidates” via `_get_rl_candidates_dynamic(...)`.
        4) Run `_rl_select_chunks_dynamic(...)` to pick final top_k.
        5) Run `_final_ranking_dynamic(...)` to reorder, then return
           (candidate_docs, final_docs, source_name).
        """
        try:
            store = self._get_store(filter_document_id)
            if store is None:
                logger.warning("No vector store available for chunk retrieval")
                return [], [], "Unknown Document"

            # (A)  Dynamic preprocessing
            try:
                processed_q = await self.dynamic_processor.adaptive_preprocess_query(query)
                logger.debug(f"Dynamic preprocessing: '{query}' → '{processed_q}'")
            except Exception as e:
                logger.warning(f"Dynamic preprocessing failed, fallback to original: {e}")
                processed_q = query

            # (B)  Candidate retrieval
            try:
                candidates = await self._get_rl_candidates_dynamic(query, processed_q, store)
            except Exception as e:
                logger.error(f"Candidate retrieval failed: {e}", exc_info=True)
                # fallback to simple similarity_search
                try:
                    candidates = store.similarity_search(query, k=self.top_k)
                    logger.warning("Fallback: simple similarity_search for candidates")
                except Exception as e2:
                    logger.error(f"Fallback similarity_search also failed: {e2}")
                    return [], [], "Unknown Document"

            if not candidates:
                logger.warning("No candidate documents found.")
                return [], [], "Unknown Document"

            # (C)  RL selection (top_k out of candidates)
            try:
                selected = await self._rl_select_chunks_dynamic(query, candidates)
            except Exception as e:
                logger.warning(f"RL selection failed, fallback to head of candidates: {e}")
                selected = candidates[: self.top_k]

            # (D)  Final ranking
            try:
                final = self._final_ranking_dynamic(query, selected)
            except Exception as e:
                logger.warning(f"Final ranking failed, using RL‐selected docs: {e}")
                final = selected

            # (E)  Extract a “source_name” from the first doc’s metadata (filename or “Unknown Document”)
            source_name = "Unknown Document"
            if final and hasattr(final[0], "metadata"):
                source_name = final[0].metadata.get("filename", source_name)

            return candidates, final, source_name

        except Exception as e:
            logger.error(f"Error in get_relevant_chunks: {e}", exc_info=True)
            # Try a last‐ditch fallback: a plain similarity_search
            try:
                store = self._get_store()
                if store:
                    docs = store.similarity_search(query, k=3)
                    src = docs[0].metadata.get("filename", "Unknown Document") if docs else "Unknown Document"
                    return docs, docs, src
            except:
                pass
            return [], [], "Unknown Document"


    # --------------------------------------------
    #  _get_rl_candidates_dynamic: gather ~15 docs
    # --------------------------------------------
    async def _get_rl_candidates_dynamic(
        self,
        original_query: str,
        processed_query: str,
        store,
    ) -> List:
        """
        Three “strategies”:
          1) similarity_search(original_query, k=8)
          2) similarity_search(processed_query, k=8) if processed != original
          3) similarity_search(variation_i, k=6) for up to 2 query variations
        Deduplicate, then pad to at least top_k, then return up to 15 unique docs.
        """
        all_cands = []

        # (1) Original query
        try:
            docs1 = store.similarity_search(original_query, k=8)
            all_cands.extend(docs1)
            logger.debug(f"Strategy 1: got {len(docs1)} docs from original_query")
        except Exception as e:
            logger.debug(f"Strategy 1 original_search failed: {e}")

        # (2) Processed query
        if processed_query != original_query:
            try:
                docs2 = store.similarity_search(processed_query, k=8)
                all_cands.extend(docs2)
                logger.debug(f"Strategy 2: got {len(docs2)} docs from processed_query")
            except Exception as e:
                logger.debug(f"Strategy 2 processed_search failed: {e}")

        # (3) variations
        try:
            variations = await self._generate_dynamic_query_variations(processed_query)
            for i, var in enumerate(variations[:2]):  # at most 2 variations
                try:
                    docs3 = store.similarity_search(var, k=6)
                    all_cands.extend(docs3)
                    logger.debug(f"Strategy 3.{i+1}: got {len(docs3)} docs from variation '{var[:30]}...'")
                except Exception as ve:
                    logger.debug(f"Strategy 3.{i+1} variation_search failed: {ve}")
        except Exception as e:
            logger.debug(f"Could not generate variations: {e}")

        # Deduplicate
        unique = self._deduplicate_docs(all_cands)

        # If fewer than top_k, try to pad with original_query again
        if len(unique) < self.top_k:
            try:
                extra = store.similarity_search(original_query, k=self.top_k * 2)
                unique.extend(extra)
                unique = self._deduplicate_docs(unique)
                logger.debug(f"Added {len(extra)} extra docs to ensure >= top_k candidates")
            except Exception as e:
                logger.debug(f"Could not retrieve additional docs for padding: {e}")

        # Limit to 15
        return unique[:15]


    # ------------------------------------------------
    #  _generate_dynamic_query_variations (helper)
    # ------------------------------------------------
    async def _generate_dynamic_query_variations(self, query: str) -> List[str]:
        """
        Look at `self.dynamic_processor.analyzer.corpus_stats` for domain terms / technical
        patterns / important phrases. Return up to ~3 variations such as:
          - f"{query} <domain_term>"
          - f"{query} <technical_pattern>"
          - f"{query} <important_phrase>"
        """
        try:
            stats = None
            try:
                stats = self.dynamic_processor.analyzer.corpus_stats
            except Exception:
                stats = None

            if not stats:
                return [query]

            variations: List[str] = []
            query_lower = query.lower().split()

            # (a) domain terms
            try:
                domain_terms = stats.get("domain_terms", set())
                added = 0
                for term in list(domain_terms)[:20]:
                    if any(word in term for word in query_lower if len(word) > 3):
                        variations.append(f"{query} {term}")
                        added += 1
                        if added >= 2:
                            break
            except Exception:
                pass

            # (b) technical patterns
            try:
                tech = stats.get("technical_patterns", {})
                for cat, terms in tech.items():
                    if not terms:
                        continue
                    for t in terms[:3]:
                        if t and t.lower() not in query.lower():
                            variations.append(f"{query} {t}")
                            break
                    if len(variations) >= 3:
                        break
            except Exception:
                pass

            # (c) important phrases
            try:
                phrases = stats.get("important_phrases", [])
                for p in phrases[:10]:
                    if any(w in p.lower() for w in query_lower if len(w) > 3):
                        variations.append(f"{query} {p}")
                        break
            except Exception:
                pass

            # Deduplicate & limit length
            clean_vars = []
            for v in variations:
                if isinstance(v, str) and v.strip():
                    clean_vars.append(v[:300])
            return clean_vars if clean_vars else [query]

        except Exception as e:
            logger.error(f"Error in _generate_dynamic_query_variations: {e}", exc_info=True)
            return [query]


    # --------------------------------------------------------
    #  _rl_select_chunks_dynamic: pick up to self.top_k docs
    # --------------------------------------------------------
    async def _rl_select_chunks_dynamic(self, query: str, candidates: List) -> List:
        """
        If candidates ≤ top_k, just return them. Otherwise:
        1) Get query_embedding (cached).
        2) Build `state = [ query_emb ; doc1_emb ; ... ; doc_k_emb ]` (base state only).
        3) Run `action_probs = self.policy_net(state_tensor)`.
        4) Choose up to top_k "diverse" indices from the top_k highest‐prob documents.
        5) Return [candidates[i] for i in selected_indices].
        """
        if len(candidates) <= self.top_k:
            return candidates

        try:
            query_emb = await self._get_cached_query_embedding(query)

            # (1) get the "base" embeddings for the first top_k candidates
            doc_embs = []
            for doc in candidates[: self.top_k]:
                cached_d = self._get_cached_embedding(doc.page_content)
                if cached_d is not None:
                    doc_embs.append(cached_d)
                else:
                    e = np.array(self.embedding_model.embed_query(doc.page_content))
                    e = self._ensure_embedding_compatibility(e)
                    self._cache_embedding(doc.page_content, e)
                    doc_embs.append(e)

            # pad if < top_k
            while len(doc_embs) < self.top_k:
                doc_embs.append(np.zeros(self.embedding_dim))

            # Use ONLY the base state (query + doc embeddings) for the policy network
            # This ensures the input dimension matches what PolicyNetwork expects
            base_state = np.concatenate([query_emb] + doc_embs).astype(np.float32)

            # (2) run policy net with base state only
            st = torch.tensor(base_state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = self.policy_net(st)

            # (3) pick "diverse" top_k indices
            chosen_indices = self._select_diverse_chunks_dynamic(action_probs, candidates, query)

            # Store experience for later reward assignment
            # For enhanced features, compute them separately but don't use in policy network
            if chosen_indices:
                primary = chosen_indices[0]
                
                # Compute enhanced features for experience buffer (but not for policy network)
                try:
                    enhanced_feats = self.rl_feature_extractor.extract_features(
                        query_emb, candidates[: self.top_k], self.dynamic_processor
                    )
                    full_state = np.concatenate([base_state, enhanced_feats]).astype(np.float32)
                except Exception as e:
                    logger.warning(f"Could not compute enhanced features for experience: {e}")
                    full_state = base_state
                
                self.experience_buffer.append({
                    "state": full_state,  # Store enhanced state for analysis
                    "action": primary,
                    "query": query,
                    "selected_docs": [candidates[i] for i in chosen_indices],
                    "timestamp": time.time()
                })
                if len(self.experience_buffer) > 100:
                    self.experience_buffer = self.experience_buffer[-50:]

            return [candidates[i] for i in chosen_indices]

        except Exception as e:
            logger.error(f"Error in _rl_select_chunks_dynamic: {e}", exc_info=True)
            return candidates[: self.top_k]


    # ----------------------------------------------------------
    #  _select_diverse_chunks_dynamic: pick up to top_k indices
    # ----------------------------------------------------------
    def _select_diverse_chunks_dynamic(self, action_probs: torch.Tensor, candidates: List, query: str) -> List[int]:
        """
        1) Find the top `k=top_k` indices by prob (action_probs).  
        2) Always pick the highest‐prob (index 0).  
        3) For each of the next highest, only pick if it is “diverse enough” 
           compared to what’s already selected (via `_is_diverse_selection_dynamic`).  
        4) Stop once we have `top_k` indices.
        """
        topk_vals, topk_idxs = torch.topk(action_probs, min(self.top_k, len(candidates)))
        selected = []

        # always include the best (if there is one)
        if len(topk_idxs) > 0:
            selected.append(topk_idxs[0].item())

        # try to fill up to top_k with diversity checks
        for i in range(1, min(self.top_k, len(topk_idxs))):
            idx = topk_idxs[i].item()
            if self._is_diverse_selection_dynamic(idx, selected, candidates, query):
                selected.append(idx)
            if len(selected) >= self.top_k:
                break

        return selected


    # -----------------------------------------------------------------------
    #  _is_diverse_selection_dynamic: checks the “cosine‐like” dynamic score
    # -----------------------------------------------------------------------
    def _is_diverse_selection_dynamic(self, candidate_idx: int, selected_indices: List[int], docs: List, query: str) -> bool:
        """
        Return False if the candidate’s content is “too similar” to any already‐selected doc, 
        using a weighted overlap that emphasizes domain terms and query words.
        """
        if candidate_idx >= len(docs):
            return False

        content_cand = docs[candidate_idx].page_content.lower()
        stats = self.dynamic_processor.analyzer.corpus_stats or {}
        domain_terms = stats.get("domain_terms", set())
        query_words = set(query.lower().split())

        for selected_idx in selected_indices:
            if selected_idx >= len(docs):
                continue
            content_sel = docs[selected_idx].page_content.lower()

            # word overlap
            w1 = set(content_cand.split())
            w2 = set(content_sel.split())
            if not w1 or not w2:
                continue

            intersection = w1.intersection(w2)
            union = w1.union(w2)

            weighted_inter = 0.0
            for w in intersection:
                weight = 1.0
                if w in domain_terms:
                    weight = 2.0
                if w in query_words:
                    weight = max(weight, 1.5)
                weighted_inter += weight

            weighted_union = float(len(union))
            # penalize missing domain terms
            for w in union:
                if w in domain_terms and w not in intersection:
                    weighted_union += 0.5

            sim_score = weighted_inter / max(weighted_union, 1.0)
            if sim_score > 0.75:
                return False

        return True


    # ---------------------------------------------------
    #  _final_ranking_dynamic: reorder selected docs by score
    # ---------------------------------------------------
    def _final_ranking_dynamic(self, query: str, selected_docs: List) -> List:
        """
        If ≤ 2 docs, return as‐is. Otherwise, compute a “dynamic score”:
          final_score = 0.6 * rl_score + 0.4 * dynamic_score
        where rl_score is implicitly 1.0 for all selected.  
        dynamic_score = combination of:
          - fraction of domain_terms present
          - fraction of technical_patterns present
          - query relevance w.r.t. dynamic_stopwords
          - important_phrases presence
        """
        if len(selected_docs) <= 2:
            return selected_docs

        stats = self.dynamic_processor.analyzer.corpus_stats or {}
        domain_terms = stats.get("domain_terms", set())
        tech_patterns = stats.get("technical_patterns", {})
        dynamic_stops = stats.get("dynamic_stopwords", set())
        imp_phrases = stats.get("important_phrases", [])

        scored = []
        for doc in selected_docs:
            content = doc.page_content.lower()
            # (a) domain term score
            dom_match = sum(1 for t in domain_terms if t in content)
            dom_score = min(dom_match / max(len(domain_terms) * 0.1, 1), 1.0) if domain_terms else 0.0

            # (b) technical patterns score
            tech_match = 0
            for _, terms in tech_patterns.items():
                tech_match += sum(1 for t in terms if t.lower() in content)
            tech_score = min(tech_match / 10.0, 1.0) if tech_match > 0 else 0.0

            # (c) query relevance w.r.t. dynamic_stopwords
            qwords = [w for w in query.lower().split() if w not in dynamic_stops]
            qr_match = sum(1 for w in qwords if w in content)
            qr_score = qr_match / len(qwords) if qwords else 0.0

            # (d) important phrases
            ip_match = sum(1 for p in imp_phrases if p.lower() in content)
            ip_score = min(ip_match / max(len(imp_phrases) * 0.1, 1), 1.0) if imp_phrases else 0.0

            dynamic_score = dom_score * 0.3 + tech_score * 0.2 + qr_score * 0.4 + ip_score * 0.1
            final_score = 0.6 * 1.0 + 0.4 * dynamic_score
            scored.append((doc, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored]


    # -------------------------------------------------
    #  calculate_enhanced_reward_dynamic: composite reward
    # -------------------------------------------------
    def calculate_enhanced_reward_dynamic(self, query: str, selected_docs: List, user_feedback: Optional[float] = None) -> float:
        """
        base_reward = len(selected_docs[0].page_content)/1000  
        user_feedback (if provided, 0..1) → 70% weight  
        dynamic_reward = _calculate_comprehensive_dynamic_score(query, doc0, stats)  
        final = 0.7*base_reward + 0.3*dynamic_reward  (then re‐blend with user_feedback if present)
        """
        base_reward = (len(selected_docs[0].page_content) / 1000.0) if selected_docs else 0.0
        if user_feedback is not None:
            base_reward = 0.3 * base_reward + 0.7 * user_feedback

        if selected_docs:
            stats = self.dynamic_processor.analyzer.corpus_stats or {}
            dynamic_r = self._calculate_comprehensive_dynamic_score(query, selected_docs[0], stats)
            base_reward = 0.7 * base_reward + 0.3 * dynamic_r

        return base_reward


    def _calculate_comprehensive_dynamic_score(self, query: str, doc, stats: Dict) -> float:
        """
        A standalone “dynamic score” function:
          0.3 * (domain_term_fraction) 
          + 0.2 * (technical_pattern_fraction)
          + 0.4 * (query relev. w.r.t. dynamic_stopwords)
          + 0.1 * (important_phrases fraction)
        """
        content = doc.page_content.lower()
        domain_terms = stats.get("domain_terms", set())
        tech_patterns = stats.get("technical_patterns", {})
        dynamic_stops = stats.get("dynamic_stopwords", set())
        imp_phrases = stats.get("important_phrases", [])

        # (1) domain term fraction
        dom_match = sum(1 for t in domain_terms if t in content)
        dom_score = min(dom_match / max(len(domain_terms) * 0.1, 1), 1.0) if domain_terms else 0.0

        # (2) technical patterns
        tech_count = 0
        for _, terms in tech_patterns.items():
            tech_count += sum(1 for t in terms if t.lower() in content)
        tech_score = min(tech_count / 10.0, 1.0) if tech_count else 0.0

        # (3) query relev w.r.t dynamic_stopwords
        qwords = [w for w in query.lower().split() if w not in dynamic_stops]
        qr_count = sum(1 for w in qwords if w in content) if qwords else 0
        qr_score = qr_count / len(qwords) if qwords else 0.0

        # (4) important phrases
        ip_count = sum(1 for p in imp_phrases if p.lower() in content)
        ip_score = min((ip_count / (len(imp_phrases) * 0.1)), 1.0) if imp_phrases else 0.0

        return dom_score * 0.3 + tech_score * 0.2 + qr_score * 0.4 + ip_score * 0.1


    # -------------------------
    #  Update RL / policy
    # -------------------------
    def update_policy(self, state: np.ndarray, action: int, reward: float):
        """
        Single‐step policy update (used if you only want a per‐query update).  
        Negative log‐prob * reward, then backprop.
        """
        self.optimizer.zero_grad()
        st_ten = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_net(st_ten)
        logp = torch.log(probs[action])
        loss = -logp * reward
        loss.backward()
        self.optimizer.step()


    def update_policy_enhanced_dynamic(self, query: str, selected_docs: List, user_feedback: Optional[float] = None):
        """
        Scan `self.experience_buffer` for the most recent experience for this query (within 5min).
        Compute a “dynamic” reward = calculate_enhanced_reward_dynamic(...)
        Append to `self.reward_history`, and if we have ≥10, call `_batch_update_policy()`.
        """
        now = time.time()
        recent = None
        for exp in reversed(self.experience_buffer):
            if exp["query"] == query and now - exp["timestamp"] < 300:
                recent = exp
                break

        if not recent:
            return

        reward = self.calculate_enhanced_reward_dynamic(query, selected_docs, user_feedback)
        self.reward_history.append((recent["state"], recent["action"], reward))
        if len(self.reward_history) >= 10:
            self._batch_update_policy()


    def _batch_update_policy(self):
        """Aggregate all stored (state, action, reward) tuples and do one backward pass."""
        if not self.reward_history:
            return
        self.optimizer.zero_grad()
        for (st, act, rew) in self.reward_history:
            st_ten = torch.tensor(st, dtype=torch.float32)
            probs = self.policy_net(st_ten)
            logp = torch.log(probs[act])
            loss = -logp * rew
            loss.backward()
        self.optimizer.step()
        logger.info("Updated RL policy with batch of rewards")
        self.reward_history.clear()


    # -----------------------------------------------------
    #  format_response (unchanged from old v1, just copy‐paste)
    # -----------------------------------------------------
    def _format_response(
        self,
        content: str,
        *,
        is_koreanis_korean: bool = True,
        section_labels: Optional[List[str]] = None,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        line_width: int = 90,
        source_name: str = "Unknown Document",
        document_type: str = "unknown"
    ) -> str:
        content = content.strip()
        if not content:
            return "정보를 찾거나 요약할 수 없습니다."

        # Document‐type defaults
        doc_configs = {
            "troubleshooting": {
                "section_labels": ["문제", "원인", "해결 방안", "참고"],
                "keyword_map": {
                    "원인": ["원인", "이유", "문제의 원인", "problem", "cause"],
                    "해결 방안": ["해결", "방안", "해결 방법", "solution", "fix", "resolve"],
                    "참고": ["참고", "추가 정보", "note", "reference"]
                },
                "terms": ["ERROR", "LOG", "CONFIGURATION", "DEBUG"],
                "code_patterns": [r'\b(ping|ssh|grep|awk|telnet)\b',
                                  r'(\b[A-Za-z]:\\[^ \n]*?|[^ \n]*?\.log\b)']
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

        config = doc_configs.get(document_type.lower(), doc_configs["unknown"])
        section_labels = section_labels or config["section_labels"]
        keyword_map    = keyword_map    or config["keyword_map"]
        terms          = config["terms"]
        code_patterns  = config["code_patterns"]

        # Direct answer if very short
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', content) if s.strip()]
        if sentences and len(sentences) <= 2 and len(content) < 100:
            ans = sentences[0]
            if not ans.endswith("."):
                ans += "."
            for term in terms:
                ans = re.sub(rf'\b{term}\b', f'**{term}**', ans, flags=re.IGNORECASE, count=1)
            for patt in code_patterns:
                ans = re.sub(patt, r'`\1`', ans, flags=re.IGNORECASE)
            return ans

        # Otherwise, split into paragraphs
        content = re.sub(r'^\s*#+\s*', '', content, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]
        if not paragraphs:
            return "정보를 찾거나 요약할 수 없습니다."

        md = [f"## 📋 {section_labels[0]}:", textwrap.fill(paragraphs[0], line_width), ""]
        buckets = {lbl: [] for lbl in section_labels[1:]}
        current = section_labels[1] if len(section_labels) > 1 else None

        # Pre‐compile cue regex
        cue_regex = {
            sec: re.compile(rf"^(?:#+\s*)?({'|'.join(map(re.escape, cues))})\s*[:\-]?\s*", re.IGNORECASE)
            for sec, cues in keyword_map.items()
        }

        for para in paragraphs[1:]:
            plain = para.lstrip("# ").strip()
            lowered = plain.lower()

            hit = next((sec for sec, cues in keyword_map.items()
                        if any(lowered.startswith(c.lower()) for c in cues)), None)
            if hit:
                current = hit
                plain = cue_regex[hit].sub("", plain).strip()

            # Normalize bullets
            if plain.startswith("* "):
                plain = "- " + plain[2:]

            # Re‐wrap text except lists
            if not plain.startswith(("-", "1.", "2.", "3.")):
                plain = textwrap.fill(plain, line_width)

            if current in buckets:
                buckets[current].append(plain)
            else:
                md.append(plain)

        # Emit buckets under headings
        for i, sec in enumerate(section_labels[1:], start=1):
            if i == 1:
                heading = f"## 🔍 {sec}:"
            elif i == 2:
                heading = f"## 🛠️ {sec}:"
            else:
                heading = f"## 📌 {sec}:"
            md.append(heading)
            md.append("")
            body = "\n".join(buckets[sec]).strip()
            md.append(body if body else "- 정보가 제공되지 않았습니다.")
            md.append("")

        # Source reference
        md.append(f"**출처**: {source_name}")
        md.append("")

        out = "\n".join(md).strip()

        # Bold terms & format code/LaTeX
        for term in terms:
            out = re.sub(rf'\b{term}\b', f'**{term}**', out, flags=re.IGNORECASE, count=1)
        for patt in code_patterns:
            out = re.sub(patt, r'`\1`', out, flags=re.IGNORECASE)
        out = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', out, flags=re.DOTALL)
        out = re.sub(r'\$(.*?)\$', r'$\1$', out)

        return out


    # ---------------------------------------------------------
    #  process_streaming_query: same high‐level as old v1, but 
    #  now calls our enhanced get_relevant_chunks(...) above
    # ---------------------------------------------------------
    async def process_streaming_query(
        self,
        query: str,
        conversation_id: str = None,
        *,
        plain_text: bool = False,
        filter_document_id: Optional[str] = None
    ):
        if not query.strip():
            raise ValueError("Query text is empty.")

        # Periodically clean old convos
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_conversations()
            self.last_cleanup = time.time()

        conversation_id = conversation_id or str(uuid.uuid4())
        hist = self.conversation_histories.setdefault(conversation_id, [])
        self.conversation_last_access[conversation_id] = time.time()

        hist_txt = self._format_history_for_prompt(hist)
        retrieval_query = f"{hist_txt} {query}" if hist_txt else query

        async with self.read_lock:
            # (1) get candidates vs. final via get_relevant_chunks(...)
            docs, selected_chunks, source_name = await self.get_relevant_chunks(
                retrieval_query, filter_document_id=filter_document_id
            )

            # (2) Group the “docs” by their source for building `context`
            grouped_docs = self._group_chunks_by_source(docs, query)
            context = "\n\n".join(
                [f"Document: {src}\n{cont}" for src, cont in grouped_docs]
            )
            document_type = docs[0].metadata.get("document_type", "unknown") if docs else "unknown"
            source_name = source_name if docs else "Unknown Document"

        # If we have no context at all, return a “no data” AIMessage
        if not context.strip():
            no_context_resp = AIMessage(content="벡터 데이터베이스가 비어 있거나 관련 정보가 없습니다.")
            hist.extend([HumanMessage(content=query), no_context_resp])
            token_handler = AsyncTokenStreamHandler()
            return AsyncPreGeneratedLLM(no_context_resp, token_handler, chunk_size=1), [], conversation_id

        # Build the system + human message for the RAG prompt
        korean_instruction = """You are a smart, RAG‐powered document analysis assistant that can read and answer questions about any company document type listed in the VALID_DOCUMENT_TYPES environment variable (e.g. troubleshooting, contract, memo, wbs, rnr, proposal, presentation). Always reply in clear, professional English.

            When you generate a response, follow these guidelines:

            1. **Start with a one‐sentence summary** of your answer.
            2. **Use Markdown**:
            - Use `# Heading` or `## Subheading` for structure.
            - Use bullet points or numbered lists for steps or examples.
            - Wrap commands, file paths, or code in backticks: ``like_this``.

            3. **Be conversational but concise**—write as if you were ChatGPT:
            - Explain jargon in plain terms.
            - Offer next steps or tips if relevant.

            4. **If the user's question is outside of document analysis**, say, “Sure, let me help with that,” and just answer naturally without forcing the template.

            Environment note:
            - Document types = `os.getenv("VALID_DOCUMENT_TYPES")`
            - Use those to decide whether to trigger detailed doc‐analysis style or freeform chat."""
        current_messages = [
            SystemMessage(content=korean_instruction),
            HumanMessage(content=f"문맥 정보: {context}\n\n질문: {query}")
        ]

        # Submit to the batch LLM for streaming
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
                result = response_future
            else:
                raise

        # Potentially translate to Korean if it is not Korean already
        #is_hangul = any(ord(char) > 127 for char in result.content[:100])
        #if not is_hangul:
         #   result.content = await asyncio.to_thread(self.translator.translate_text, result.content)

        is_hangul=True


        # Format (unless plain_text=True)
        if not plain_text:
            result.content = self._format_response(
                result.content,
                is_korean=is_hangul,
                source_name=source_name,
                document_type=document_type
            )

        # Compute RL reward & update
        try:
            query_emb = np.array(self.embedding_model.embed_query(retrieval_query))
            query_emb = self._ensure_embedding_compatibility(query_emb)
            state = self._get_enhanced_rl_state_dynamic(query_emb, docs)
            action = docs.index(selected_chunks[0]) if len(selected_chunks) == 1 else 0
            reward = self.calculate_enhanced_reward_dynamic(retrieval_query, selected_chunks)
            self.reward_history.append((state, action, reward))
            if len(self.reward_history) >= 10:
                self._batch_update_policy()
        except Exception as e:
            logger.warning(f"Could not update RL policy for this query: {e}")

        # Finally, append to conversation history
        hist.extend([HumanMessage(content=query), AIMessage(content=result.content)])

        token_handler = AsyncTokenStreamHandler()
        return AsyncPreGeneratedLLM(result, token_handler, chunk_size=1), current_messages, conversation_id

    def get_cache_stats(self) -> Dict[str, float]:
        hit_rate = self._cache_hits / (self._cache_requests or 1)
        return {
            "size": len(self.embedding_cache),
            "max_size": self.embedding_cache_max_size,
            "hit_rate": hit_rate,
            "total_requests": self._cache_requests,
            "cache_hits": self._cache_hits
        }

    def _get_snippet_with_keyword(self, content: str, query: str, max_length: int = 500):
        if not query or not content:
            return (content[:max_length] + "...") if len(content) > max_length else content

        qterms = query.lower().split()
        clow = content.lower()
        positions = [clow.find(t) for t in qterms if clow.find(t) != -1]
        if not positions:
            return (content[:max_length] + "...") if len(content) > max_length else content

        pos = min(positions)
        start = max(0, pos - 200)
        end = min(len(content), pos + 300)
        if start > 0:
            sb = content.rfind(" ", 0, start)
            if sb != -1:
                start = sb + 1
        if end < len(content):
            sa = content.find(" ", end)
            if sa != -1:
                end = sa

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet

    async def perform_similarity_search(self, query: str):
        """
        Very similar to v1:  
        1) Lock, embed or re‐use cached embedding.  
        2) similarity_search_with_score(query, k=self.top_k).  
        3) RL‐select 1 result (or top 3 fallback).  
        4) Return JSON with id/snippet/relevance.
        """
        try:
            async with self.read_lock:
                cached_q = self._get_cached_embedding(query)
                if cached_q is not None:
                    qemb = cached_q
                else:
                    qemb = np.array(self.embedding_model.embed_query(query))
                    qemb = self._ensure_embedding_compatibility(qemb)
                    self._cache_embedding(query, qemb)

                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
                docs = [d for d, _ in docs_with_scores]

                state = self._get_state(qemb, docs)
                ap = self.policy_net(torch.tensor(state, dtype=torch.float32))
                action = torch.multinomial(ap, 1).item()
                if torch.max(ap) > 0.5:
                    chosen = [docs[action]]
                else:
                    chosen = docs[:3]

                results = []
                for doc in chosen:
                    idx = docs.index(doc) if doc in docs else 0
                    score = float(docs_with_scores[idx][1]) if idx < len(docs_with_scores) else 0.5
                    results.append({
                        "id": doc.metadata.get("id", "unknown"),
                        "snippet": self._get_snippet_with_keyword(doc.page_content, query),
                        "relevance": score
                    })
                results.sort(key=lambda x: x["relevance"], reverse=True)

            # RL reward = len(results)/3
            reward = len(results) / 3.0
            self.reward_history.append((state, action, reward))
            if len(self.reward_history) >= 10:
                self._batch_update_policy()

            return {
                "query": query,
                "results": results,
                "total_results": len(results)
            }

        except Exception as e:
            logger.error(f"Error in perform_similarity_search: {e}", exc_info=True)
            raise


    async def search_by_vector(self, query: Optional[str], status_code: str):
        """
        Virtually identical to v1, just wrapped in async + read_lock.  
        I.e. try three “strategies” of filtered/unfiltered searches by status_code,
        group results by source, make a snippet, then run `_generate_analysis(...)` 
        for a short summary at the end.
        """
        try:
            logger.info(f"search_by_vector(query={query!r}, status_code={status_code!r})")
            docs = []

            if query:
                # Strategy 1: filter by error_code_nm
                try:
                    fdocs = self.vector_store.similarity_search(
                        f"Status Code {status_code} {query}",
                        filter={"error_code_nm": status_code},
                        k=5
                    )
                    if fdocs:
                        docs = fdocs
                        logger.info(f"Found {len(fdocs)} via error_code_nm filter")
                    else:
                        logger.debug("No documents found with error_code_nm filter (Strategy 1)")
                except Exception as e:
                    logger.warning(f"Strategy 1 filtered search failed: {e}")

                # Strategy 2: filter by document_id or logical_nm
                if not docs:
                    try:
                        md_filt = {
                            "$or": [
                                {"document_id": {"$contains": status_code}},
                                {"logical_nm": {"$contains": status_code}}
                            ]
                        }
                        fdocs2 = self.vector_store.similarity_search(
                            f"NetBackup Status Code {status_code} {query}",
                            filter=md_filt,
                            k=5
                        )
                        if fdocs2:
                            docs = fdocs2
                            logger.info(f"Found {len(fdocs2)} via document_id/logical_nm filter")
                        else:
                            logger.debug("No docs found in Strategy 2")
                    except Exception as e:
                        logger.warning(f"Strategy 2 filtered search failed: {e}")

                # Strategy 3: unfiltered search with manual filtering
                if not docs:
                    try:
                        back = self.vector_store.similarity_search(
                            f"NetBackup Status Code {status_code} {query}",
                            k=10
                        )
                        filtered = [
                            d for d in back
                            if any(
                                phrase in d.page_content
                                for phrase in [
                                    f"Status Code {status_code}",
                                    f"Status Code: {status_code}",
                                    f"Code {status_code}",
                                    f"ErrorCode {status_code}"
                                ]
                            )
                        ]
                        if filtered:
                            docs = filtered
                            logger.info(f"Found {len(filtered)} after filtering unfiltered search")
                        elif back:
                            docs = back[:5]
                            logger.info(f"Using top {len(docs)} from unfiltered search")
                        else:
                            logger.debug("No docs found in Strategy 3")
                    except Exception as e:
                        logger.warning(f"Strategy 3 search failed: {e}")

            else:
                # No query: retrieve all for status_code with filter
                try:
                    all_docs = self.vector_store.similarity_search(
                        f"Status Code {status_code}",
                        filter={"error_code_nm": status_code},
                        k=100
                    )
                    if all_docs:
                        docs = all_docs
                        logger.info(f"Found {len(all_docs)} for status_code {status_code} (Strategy A)")
                    else:
                        logger.debug("No docs in Strategy A (filtered by error_code_nm)")

                        md_filt = {
                            "$or": [
                                {"document_id": {"$contains": status_code}},
                                {"logical_nm": {"$contains": status_code}}
                            ]
                        }
                        fdocs3 = self.vector_store.similarity_search(
                            f"Status Code {status_code}",
                            filter=md_filt,
                            k=100
                        )
                        if fdocs3:
                            docs = fdocs3
                            logger.info(f"Found {len(fdocs3)} via docs_id/logical_nm (Strategy B)")
                        else:
                            logger.info("No docs found for status_code at all")
                except Exception as e:
                    logger.warning(f"No docs fetched for status_code {status_code}: {e}")

            logger.info(f"Total documents found: {len(docs)}")

            # Group by source (logical_nm or document_id)
            grouped: Dict[str, List] = {}
            for d in docs:
                md = d.metadata if hasattr(d, "metadata") else {}
                key = md.get("logical_nm", md.get("document_id", "unknown"))
                grouped.setdefault(key, []).append(d)

            results = []
            for key, grp in grouped.items():
                first = grp[0]
                md = first.metadata if hasattr(first, "metadata") else {}
                if "logical_nm" in md:
                    src = md.get("logical_nm", f"File {key}")
                    doc_url = md.get("url", "")
                else:
                    src = md.get("document_id", "Troubleshooting Report Text")
                    doc_url = ""

                combined = " ".join(d.page_content for d in grp)
                snippet = self._get_snippet_with_keyword(combined, f"{status_code} {query or ''}")
                if not snippet:
                    snippet = combined[:800] + "..." if len(combined) > 800 else combined

                ext = ""
                file_type = md.get("file_type", "")
                if not file_type and "." in src:
                    e = src.split(".")[-1].lower()
                    if e in ["xlsx", "xls", "pdf", "docx", "doc", "txt", "log", "html", "kb"]:
                        ext = f"{e.upper()} 파일"

                doc_id = md.get("id", "") or f"doc-{hashlib.md5(combined[:1000].encode()).hexdigest()[:8]}"
                created = md.get("created", "")
                title = md.get("title", src)

                results.append({
                    "filename": src,
                    "snippet": snippet,
                    "metadata": {
                        "source": src,
                        "title": title,
                        "file_type": file_type or ext,
                        "url": doc_url,
                        "path": md.get("path", ""),
                        "id": doc_id,
                        "created": created,
                        "status_code": md.get("error_code_nm", status_code)
                    }
                })
                logger.debug(f"Grouped result for source: {src}")

            logger.info(f"Processed {len(results)} grouped doc groups")

            # Generate summary (only use first 5 docs’ content to avoid extremely long LLM prompts)
            if docs:
                sample_ctx = "\n".join(d.page_content for d in docs[:5])
                summary_resp = await self._generate_analysis(sample_ctx, query or "", status_code)
            else:
                if not query:
                    summary_resp = f"상태 코드 {status_code}에 대한 정보를 찾을 수 없습니다."
                else:
                    summary_resp = f"상태 코드 {status_code}에 대한 '{query}' 관련 정보를 찾을 수 없습니다."

            return {
                "status_code": status_code,
                "query": query,
                "summary": summary_resp,
                "results": results
            }

        except Exception as e:
            logger.error(f"Unexpected error in search_by_vector: {e}", exc_info=True)
            return {
                "status_code": status_code,
                "query": query,
                "summary": f"검색 중 오류 발생: {e}",
                "results": []
            }


    # ---------------------------------------------------------
    #  _generate_analysis: same as old v1, just ensure we await
    # ---------------------------------------------------------
    async def _generate_analysis(self, content: str, query: str, status_code: str) -> str:
        try:
            if query:
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
            msgs = [
                SystemMessage(content="You are a NetBackup expert analyzing technical documents."),
                HumanMessage(content=prompt)
            ]
            response_future = await self.analysis_batch_manager.submit_request(
                query=query or f"Summary for status code {status_code}",
                context=content,
                messages=msgs,
                conversation_id=conversation_id
            )
            try:
                result = await response_future
            except TypeError as e:
                if "can't be used in 'await'" in str(e):
                    result = response_future
                else:
                    raise
            eng = result.content
            return await asyncio.to_thread(self.translator.translate_text, eng)

        except Exception as e:
            logger.error(f"Error generating analysis: {e}", exc_info=True)
            try:
                logger.warning("Falling back to direct Ollama (mistral) for analysis")
                r = ollama.chat(
                    model="mistral:latest",
                    messages=[{"role": "user", "content": prompt}]
                )
                eng = r["message"]["content"]
                return await asyncio.to_thread(self.translator.translate_text, eng)
            except Exception as fe:
                logger.error(f"Fallback analysis failed: {fe}", exc_info=True)
                return "AI 분석 중 오류가 발생했습니다. 다시 시도해 주시기 바랍니다."


    # ---------------------------------------------------------
    #  refresh_stores: reloads kb_store & chat_store & updates dynamic_processor
    # ---------------------------------------------------------
    def refresh_stores(self) -> bool:
        """
        Call when you know you’ve just run `/query/resetChromaCollection`.  
        This method will re‐call `_get_validated_vector_store("kb")` and `_get_validated_vector_store("chat")`.  
        If the KB‐store changed, we also update `dynamic_processor.vector_store` so it stops pointing to the old UUID.
        """
        try:
            old_kb = self.kb_store
            old_chat = self.chat_store

            self.kb_store = self._get_validated_vector_store("kb")
            self.chat_store = self._get_validated_vector_store("chat")

            if self.kb_store is not None and self.kb_store != old_kb:
                try:
                    self.dynamic_processor.vector_store = self.kb_store
                    # force the TF‐IDF corpus to re‐analyze
                    self.dynamic_processor.last_corpus_update = 0
                    logger.info("DynamicProcessor's KB store updated successfully")
                except Exception as e:
                    logger.warning(f"Could not update dynamic_processor.vector_store: {e}")

            ok_kb = self.kb_store is not None
            ok_chat = self.chat_store is not None
            logger.info(f"refresh_stores completed – KB: {'✓' if ok_kb else '✗'}, Chat: {'✓' if ok_chat else '✗'}")
            return ok_kb or ok_chat

        except Exception as e:
            logger.error(f"Error refreshing vector stores: {e}", exc_info=True)
            return False

    def _get_enhanced_rl_state_dynamic(self, query_emb: np.ndarray, docs: List) -> np.ndarray:
        """
        Build enhanced RL state combining query embedding, document embeddings, and dynamic features.
        This is called during RL policy updates in process_streaming_query.
        """
        try:
            # Get base embeddings for first top_k docs
            doc_embs = []
            for doc in docs[:self.top_k]:
                cached_d = self._get_cached_embedding(doc.page_content)
                if cached_d is not None:
                    doc_embs.append(cached_d)
                else:
                    e = np.array(self.embedding_model.embed_query(doc.page_content))
                    e = self._ensure_embedding_compatibility(e)
                    self._cache_embedding(doc.page_content, e)
                    doc_embs.append(e)

            # Pad if fewer than top_k
            while len(doc_embs) < self.top_k:
                doc_embs.append(np.zeros(self.embedding_dim))

            # Base state (query + doc embeddings)
            base_state = np.concatenate([query_emb] + doc_embs).astype(np.float32)

            # Add enhanced features if available
            try:
                enhanced_feats = self.rl_feature_extractor.extract_features(
                    query_emb, docs[:self.top_k], self.dynamic_processor
                )
                full_state = np.concatenate([base_state, enhanced_feats]).astype(np.float32)
                return full_state
            except Exception as e:
                logger.warning(f"Could not compute enhanced features, using base state: {e}")
                return base_state

        except Exception as e:
            logger.error(f"Error building enhanced RL state: {e}")
            # Fallback to minimal state
            return query_emb
    
    def _format_history_for_prompt(self, messages: List)->str:
        history_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content
                assistant_msg = messages[i + 1].content
                history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
        return "\n\n".join(history_pairs)        
    
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


    async def _generate_analysis(self, content: str, query: str, status_code: str):
        """
        Generate a technical analysis or summary based on document content.
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