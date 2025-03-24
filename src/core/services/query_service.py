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

# Import LangGraph memory modules
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages

from src.core.inference.batch_inference import BatchInferenceManager


logger = logging.getLogger(__name__)

class AsyncPreGeneratedLLM:
    """A class that mimics the streaming LLM interface but returns a pre-generated response."""
    
    def __init__(self, result, token_handler, chunk_size):
        self.result = result
        self.token_handler = token_handler
        self.chunk_size=chunk_size
    
    async def astream(self, messages):
        """Stream the pre-generated result."""
        content = self.result.content
        
        # Put content into token handler's queue
        for chunk in self._split_content_into_chunks(content):
            await self.token_handler.queue.put(chunk)
        
        # Signal end of streaming
        await self.token_handler.queue.put(None)
        
        # Generate tokens from the queue
        async for token in self.token_handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, content, chunk_size=4):
        """Split content into small chunks to simulate streaming."""
        if chunk_size <= 1:
        # Character by character for smoother streaming
            return [char for char in content]
        else:
        # Word by word if chunk_size > 1
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
        """Stream the pre-generated result."""
        content = self.result.content
        
        # Put content into token handler's queue
        for chunk in self._split_content_into_chunks(content):
            await self.token_handler.queue.put(chunk)
        
        # Signal end of streaming
        await self.token_handler.queue.put(None)
        
        # Generate tokens from the queue
        async for token in self.token_handler.stream():
            yield AIMessage(content=token)
    
    def _split_content_into_chunks(self, content, chunk_size=4):
        """Split content into small chunks to simulate streaming."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
            
        return chunks


#     def get_memory(self, conversation_id, memory_type="buffer", window_size=5, llm=None):
#         """Get or create a memory object for the given conversation ID"""
#         if conversation_id not in self.memories:
#             if memory_type == "summary":
#                 if llm is None:
#                     raise ValueError("LLM must be provided for summary memory")
#                 self.memories[conversation_id] = ConversationSummaryMemory(
#                     llm=llm,
#                     max_token_limit=2000,
#                     return_messages=True
#                 )
#             elif memory_type == "buffer_window":
#                 self.memories[conversation_id] = ConversationBufferWindowMemory(
#                     k=window_size,
#                     return_messages=True
#                 )
#             else:  # Default to buffer memory
#                 self.memories[conversation_id] = ConversationBufferMemory(
#                     return_messages=True
#                 )
        
#         return self.memories[conversation_id]
    
#     def clear_memory(self, conversation_id):
#         """Clear the memory for a conversation"""
#         if conversation_id in self.memories:
#             del self.memories[conversation_id]
#             return True
#         return False

# # stream pre-generated responses




class QueryService:
    def __init__(self, translator, rag_chain, global_prompt):
        if not global_prompt:
            raise ValueError("Global prompt cannot be None")
        #self.vector_store = vector_store
        self.translator = translator
        self.rag_chain = rag_chain
        self.global_prompt = global_prompt

        #self.workflow=None
        #self.memory=None

        self.app=self.rag_chain
        #deepseek-r1:1.5b
        #llama3:latest
        #gemma3:4b
        self.llm = ChatOllama(model="gemma3:12b", temperature=0.1, stream=True)

        self.tavily_search = False
        self.tavily_enabled = False

        # try:
        #     self.tavily_search = False
        #     self.tavily_enabled = True
        # except Exception as e:
        #     logger.warning(f"Failed to initialize TavilySearch: {e}. Web search will be disabled.")
        #     self.tavily_enabled = False

        self.batch_manager=BatchInferenceManager(
            batch_interval=0.1,
            max_batch_size=5
        )

        self.analysis_batch_manager = BatchInferenceManager(
            batch_interval=0.1,
            max_batch_size=5,
            model="mistral:latest"  # Specify the mistral model for analysis
        )

        self.conversation_histories={}


    @property
    def vector_store(self):
            """Dynamically fetch the latest vector store."""
            vs = get_vector_store()
            if not vs:
                raise ValueError("Vector store not initialized")
            return vs


    def _setup_workflow(self):
        """Set up the LangGraph workflow for conversation management"""
        
        def call_model(state: MessagesState):
            """Call the model with the current state"""
            # We could implement context retrieval here, but for now we'll keep it separate
            response = self.llm.invoke(state["messages"])
            return {"messages": response}
        
        # Add the model node
        self.workflow.add_node("model", call_model)
        self.workflow.add_edge(START, "model")
    
    def _get_recent_messages(self, conversation_id, max_messages=8):
        """Get recent messages for a conversation from local history."""
        if conversation_id not in self.conversation_histories:
            self.conversation_histories[conversation_id] = []
        history = self.conversation_histories[conversation_id]
        return history[-max_messages:] if len(history) > max_messages else history
    
    def _format_history_for_prompt(self, messages):
        """Format conversation history for the prompt"""
        history_pairs = []
        
        for i in range(0, len(messages), 2):
            if i+1 < len(messages):
                user_msg = messages[i].content
                assistant_msg = messages[i+1].content
                history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
            
        return "\n\n".join(history_pairs)        


    async def process_basic_query(self, query: str):
        """Handle basic query processing"""
        try:
            if not query.strip():
                raise ValueError("Query text is empty.")
            
            # Use direct chat with relevant context
            # Get context from vector store
            try:
                docs = self.vector_store.similarity_search(query, k=5)
                context = "\n".join(doc.page_content[:600] for doc in docs)
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                context = ""
                
            # Create prompt with context
            messages = [
                SystemMessage(content="당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 명확하게 답변하세요."),
                HumanMessage(content=f"문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:")
            ]
            
            response = self.llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Error in process_basic_query: {e}")
            raise

    async def process_streaming_query(self, query: str,conversation_id: str = None):
        try:
            if not query.strip():
                raise ValueError("Query text is empty.")

            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Make sure we have a conversation history
            if conversation_id not in self.conversation_histories:
                self.conversation_histories[conversation_id] = []
            
            # Get conversation history 
            history = self._get_recent_messages(conversation_id)
         
            
            retrieval_query = query
            
            if history:
                # Extract recent user messages for better context
                user_msgs = [msg for msg in history if isinstance(msg, HumanMessage)][-2:]
                history_text = " ".join([msg.content for msg in user_msgs])
                retrieval_query = f"{history_text} {query}"
            
            retrieval_query = self._enhance_query(retrieval_query)
            
            # Get context from vector store with enhanced query
            docs = self.vector_store.similarity_search(retrieval_query, k=5)

            grouped_docs = self._group_chunks_by_source(docs, query)

            context = "\n\n".join([f"Document: {source}\n{content}" for source, content in grouped_docs])

            if not context.strip():
                logger.info(f"No documents found for query: {query}")
                # Create a pre-generated response for empty context
                no_context_response = AIMessage(content="현재 데이터베이스에 관련 정보가 없습니다. NetBackup 문서를 추가해 주세요.")
                self.conversation_histories[conversation_id].append(HumanMessage(content=query))
                self.conversation_histories[conversation_id].append(no_context_response)
                
                max_messages = 8
                if len(self.conversation_histories[conversation_id]) > max_messages:
                    self.conversation_histories[conversation_id] = self.conversation_histories[conversation_id][-max_messages:]

                token_handler = AsyncTokenStreamHandler()
                streaming_llm = AsyncPreGeneratedLLM(no_context_response, token_handler, chunk_size=1)
                return streaming_llm, [], conversation_id


            #  # Check if we should use web search
            # use_web_search = False
            # if self.tavily_enabled:
            #     try:
            #         decision = await self.tavily_search.decide_search_method(query, docs)
            #         use_web_search = decision["use_web_search"]
            #     except Exception as e:
            #         logger.error(f"Error in deciding search method: {e}")
            #         use_web_search = False
            # else:
            #     use_web_search = False
                
            # # If web search is needed, perform it
            # if use_web_search:
            #     try:
            #         logger.info(f"Using web search for query: {query}")
            #         web_results = await self.tavily_search.search_web(query)
                    
            #         # Format context from web results
            #         web_context = self.tavily_search.format_web_search_results(web_results)
            #         context = f"웹 검색 결과: {web_context}"
            #     except Exception as e:
            #         logger.error(f"Error in web search: {e}")
            #         # Fallback to document results
            #         context = "\n".join(doc.page_content[:600] for doc in docs)
            # else:
            #     # Use document context
            #     context = "\n".join(doc.page_content[:600] for doc in docs)

            # memory_content = memory.load_memory_variables({})
            history_for_prompt = self._format_history_for_prompt(history)
            
        
            # if "history" in memory_content and memory_content["history"]:
            #     history_pairs = []
            #     chat_history=memory_content["history"]
                
            #     for i in range(0, len(chat_history), 2):
            #         if i+1 < len(chat_history):
            #             user_msg = chat_history[i].content
            #             assistant_msg = chat_history[i+1].content
            #             history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
                
            #     history_for_prompt = "\n\n".join(history_pairs)    

            # Initialize message history if needed
            # if not hasattr(self, 'message_history'):
            #     self.message_history = []
            
           # Prepare the system instruction and query with context
            korean_instruction = """당신은 NetBackup 시스템 전문가입니다. 
            반드시 한국어로 명확하게 답변하세요. 기술 용어만 영어로 유지하세요.
            대화의 맥락을 유지하고 이전 대화를 참조하여 일관성 있는 답변을 제공하세요.
            사용자가 이전 질문이나 답변을 언급할 때는 그 맥락을 이해하고 적절히 응답하세요."""
            
            #if use_web_search:
             #   query_with_context = f"대화 기록:\n{history_for_prompt}\n\n문맥 정보 (웹 검색 결과): {context}\n\n질문: {query}\n\n한국어로 답변해 주세요. 답변 시작에 '웹 검색 결과:' 라고 표시하고, 답변 끝에 관련 URL을 포함하세요:"
            #else:
            query_with_context = f"대화 기록:\n{history_for_prompt}\n\n문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:"
            
            # Create message format
            current_messages = [
                SystemMessage(content=korean_instruction),
                HumanMessage(content=query_with_context)
            ]

            self.conversation_histories[conversation_id].append(HumanMessage(content=query))
            
            # Submit to batch manager instead of directly calling the LLM
            #logger.info(f"Submitting request to batch manager for conversation {conversation_id}")
            response_future = await self.batch_manager.submit_request(
                query=query,
                context=context,
                messages=current_messages,
                conversation_id=conversation_id
            )
            
            # Wait for the result
            result = await response_future
            # config = {"configurable": {"thread_id": conversation_id}}
            self.conversation_histories[conversation_id].append(AIMessage(content=result.content))

            max_messages = 8
            if len(self.conversation_histories[conversation_id]) > max_messages:
                self.conversation_histories[conversation_id] = self.conversation_histories[conversation_id][-max_messages:]

            # Create a token handler for streaming the pre-generated response
            token_handler = AsyncTokenStreamHandler()
            streaming_llm = AsyncPreGeneratedLLM(result, token_handler, chunk_size=1)
            
            return streaming_llm, current_messages, conversation_id
        
        except Exception as e:
            logger.error(f"Error in process_streaming_query: {e}")
            raise
    
    def _enhance_query(self, query):
        """Enhance the query to improve retrieval of relevant chunks"""
        # Detect command requests
        if any(term in query.lower() for term in ["how to", "command", "steps", "procedure", "명령어", "단계", "절차"]):
            return f"command procedure steps {query}"
        
        # Detect database related queries
        if any(term in query.lower() for term in ["database", "db", "데이터베이스"]):
            return f"database recovery restore {query}"
            
        # Detect error or status code queries
        if any(term in query.lower() for term in ["error", "status code", "status", "code", "에러", "상태", "코드"]):
            return f"error status code troubleshooting {query}"
            
        return query

    def _group_chunks_by_source(self, docs, query):
        """Group chunks by source document and sort by relevance to query"""
        # Extract key terms from query
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        
        # Group docs by source
        source_groups = {}
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Score each source based on term matches
        source_scores = {}
        for source, source_docs in source_groups.items():
            # Count term matches across all chunks from this source
            match_count = 0
            for doc in source_docs:
                content = doc.page_content.lower()
                match_count += sum(1 for term in query_terms if term in content)
            
            # Calculate score based on matches and document count
            source_scores[source] = match_count / len(query_terms) if query_terms else 0
        
        # Sort sources by score
        sorted_sources = sorted(source_groups.keys(), key=lambda s: source_scores.get(s, 0), reverse=True)
        
        # Take the top 3 sources and concatenate their chunks
        result = []
        for source in sorted_sources[:3]:
            # Get chunks from this source
            chunks = [doc.page_content for doc in source_groups[source]]
            
            # Join chunks with minimal formatting
            source_text = "\n".join(chunks)
            
            # Add to result
            result.append((source, source_text))
        
        return result

    
    # Add this to your QueryService's process_streaming_query method
    def preprocess_query(query):
        """Enhance queries with key terms to improve retrieval"""
        # Extract command-specific keywords
        if "how to" in query.lower() or "steps" in query.lower() or "command" in query.lower():
            return f"command steps procedure {query}"
        # Extract database-related keywords
        if "database" in query.lower() or "db" in query.lower():
            return f"database recovery NetBackup command {query}"
        return query

 
    def clear_conversation(self, conversation_id):
        """Clear a conversation's memory"""
        try:
            # Clear history
            if conversation_id in self.conversation_histories:
                del self.conversation_histories[conversation_id]
                
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False  
    
    async def perform_similarity_search(self, query: str):
        """Perform basic similarity search"""
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
        """Generate snippet with keyword context"""
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
        """Perform vector similarity search with status code filtering"""
        try:
            logger.debug(f"Searching for query: {query} with status code: {status_code}")
            matching_filter = {"status_code": {"$eq": status_code}}
            
            # Use the dynamic vector_store property
            docs = self.vector_store.similarity_search(
                query,
                k=5,
                filter=matching_filter
            )

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

            summary = await self._generate_analysis(
                "\n".join(doc.page_content for doc in docs),
                query,
                status_code
            )

            return {
                "status_code": status_code,
                "results": results,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error in search_by_vector: {e}")
            raise

    async def _generate_analysis(self, content: str, query: str, status_code: str):

        """Generate analysis from content"""
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

            # Create a conversation ID for this analysis request
            conversation_id = str(uuid.uuid4())
        
             # Create messages for the LLM
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
            
            #return self.translator.translate_text(english_summary)
            summary= await asyncio.to_thread(self.translator.translate_text, english_summary)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            # If batch processing fails, fall back to direct Ollama call
            try:
                logger.warning(f"Falling back to direct Ollama call for analysis with mistral model")
                response = ollama.chat(
                    model='mistral:latest',
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
                english_summary = response['message']['content']
                return await asyncio.to_thread(self.translator.translate_text, english_summary)
            except Exception as fallback_error:
                logger.error(f"Error in fallback analysis: {fallback_error}")
                return "AI 분석 중 오류가 발생했습니다. 다시 시도해 주시기 바랍니다."

               