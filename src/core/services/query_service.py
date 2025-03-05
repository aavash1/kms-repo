# src/core/services/query_service.py
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

from src.core.inference.batch_inference import BatchInferenceManager
import uuid

logger = logging.getLogger(__name__)

# stream pre-generated responses
class AsyncPreGeneratedLLM:
    """A class that mimics the streaming LLM interface but returns a pre-generated response."""
    
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

class QueryService:
    def __init__(self, vector_store, translator, rag_chain, global_prompt):
        if not global_prompt:
            raise ValueError("Global prompt cannot be None")
        self.vector_store = vector_store
        self.translator = translator
        self.rag_chain = rag_chain
        self.global_prompt = global_prompt

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

        if not vector_store:
            raise ValueError("Vector store cannot be None")
        if not rag_chain:
            raise ValueError("RAG chain cannot be None")

    async def process_basic_query(self, query: str):
        """Handle basic query processing"""
        try:
            if not query.strip():
                raise ValueError("Query text is empty.")
            
            response = self.rag_chain.invoke(query)
            return {"answer": response}
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
            
            # Initialize or retrieve conversation history
            if conversation_id not in self.conversation_histories:
                self.conversation_histories[conversation_id] = []
            
            # Get the conversation history
            conversation_history = self.conversation_histories.get(conversation_id, [])

            # Get context from vector store
            docs = self.vector_store.similarity_search(query)


             # Check if we should use web search
            use_web_search = False
            if self.tavily_enabled:
                try:
                    decision = await self.tavily_search.decide_search_method(query, docs)
                    use_web_search = decision["use_web_search"]
                except Exception as e:
                    logger.error(f"Error in deciding search method: {e}")
                    use_web_search = False
            else:
                use_web_search = False
                
            # If web search is needed, perform it
            if use_web_search:
                try:
                    logger.info(f"Using web search for query: {query}")
                    web_results = await self.tavily_search.search_web(query)
                    
                    # Format context from web results
                    web_context = self.tavily_search.format_web_search_results(web_results)
                    context = f"웹 검색 결과: {web_context}"
                except Exception as e:
                    logger.error(f"Error in web search: {e}")
                    # Fallback to document results
                    context = "\n".join(doc.page_content[:600] for doc in docs)
            else:
                # Use document context
                context = "\n".join(doc.page_content[:600] for doc in docs)

            history_text = ""
            if conversation_history:
                history_pairs = []
                for i in range(0, len(conversation_history), 2):
                    if i+1 < len(conversation_history):
                        user_msg = conversation_history[i].content
                        assistant_msg = conversation_history[i+1].content
                        history_pairs.append(f"사용자: {user_msg}\n시스템: {assistant_msg}")
                
                history_text = "\n\n".join(history_pairs)    

            # Initialize message history if needed
            if not hasattr(self, 'message_history'):
                self.message_history = []
            
           # Prepare the system instruction and query with context
            korean_instruction = "당신은 NetBackup 시스템 전문가입니다. 반드시 한국어로 명확하게 답변하세요. 기술 용어만 영어로 유지하세요."
            
            if use_web_search:
                query_with_context = f"대화 기록:\n{history_text}\n\n문맥 정보 (웹 검색 결과): {context}\n\n질문: {query}\n\n한국어로 답변해 주세요. 답변 시작에 '웹 검색 결과:' 라고 표시하고, 답변 끝에 관련 URL을 포함하세요:"
            else:
                query_with_context = f"대화 기록:\n{history_text}\n\n문맥 정보: {context}\n\n질문: {query}\n\n한국어로 답변해 주세요:"
            
            # Create message format
            messages = [
                SystemMessage(content=korean_instruction),
                HumanMessage(content=query_with_context)
            ]
             # Add user query to conversation history
            self.conversation_histories[conversation_id].append(HumanMessage(content=query))
            
            # Submit to batch manager instead of directly calling the LLM
            logger.info(f"Submitting request to batch manager for conversation {conversation_id}")
            response_future = await self.batch_manager.submit_request(
                query=query,
                context=context,
                messages=messages,
                conversation_id=conversation_id
            )
            
            # Wait for the result
            result = await response_future
            
            # Create a token handler for streaming the pre-generated response
            token_handler = AsyncTokenStreamHandler()
            streaming_llm = AsyncPreGeneratedLLM(result, token_handler)
            
            return streaming_llm, messages, conversation_id
        
        except Exception as e:
            logger.error(f"Error in process_streaming_query: {e}")
            raise

    
    
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
            if not self.vector_store:
                raise ValueError("Vector store not initialized.")
            
            logger.debug(f"Searching for query: {query} with status code: {status_code}")

            matching_filter={"status_code": {"$eq":status_code}}
            
            docs = self.vector_store.similarity_search(
                query,
                k=5,
                filter=matching_filter
            )

            # Process results
            results = []
            for doc in docs:
                snippet = self._get_snippet_with_keyword(doc.page_content, query)
                if snippet:
                    results.append({
                        "filename": doc.metadata.get("filename", "unknown"),
                        "snippet": snippet,
                        "score": doc.metadata.get("score", 0.0)
                    })

            # If no relevant content found
            if not results:
                return {
                    "status_code": status_code,
                    "results": [],
                    "summary": "죄송합니다. 요청하신 검색어와 관련된 문서 내용을 찾을 수 없습니다."
                }

            
            # Generate summary from documents
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