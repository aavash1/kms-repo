"""
Batch Inference Manager for KMSChatbot.

This module provides efficient batching of LLM inference requests to optimize
GPU utilization and increase concurrent user capacity.
"""

import asyncio
import time
import logging
import threading
from typing import List, Any, Dict, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.tracers.langchain import LangChainTracer
import os
from langchain_core.runnables.config import RunnableConfig


logger = logging.getLogger(__name__)

class BatchInferenceManager:
    """
    A manager that batches inference requests together and processes them
    efficiently on the GPU. It offloads the blocking batch inference using 
    asyncio.to_thread to keep the main event loop responsive.
    """
    #def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 5, model: str = "deepseek-r1:14b"):
    #llama3:latest
    def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 5, model: str = "llama3:latest"):
        """
        Initialize the batch inference manager.
        
        Args:
            batch_interval: Time window (in seconds) to wait before processing a batch.
            max_batch_size: Maximum number of requests to collect before processing immediately.
        """
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.requests = []  # List of tuples: (query, context, messages, conversation_id, future)
        self.lock = asyncio.Lock()  # Protects access to the requests list
        self.ollama_model = model  # Default model
        self.ollama_params = {
            "num_gpu": 1,
            "num_thread": 4,
            "temperature": 0.1,
        }

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # LLM instance (created on first use)
        self._llm_instance = None
        self._llm_lock = threading.Lock()
        
        # Start the background task that periodically processes the batch
        self._processor_task = None
        #self._processor_task = asyncio.create_task(self._batch_processor())
        logger.info(f"Batch Inference Manager initialized with interval={batch_interval}s, max_size={max_batch_size}")
    
    def _ensure_processor_task(self):
        """Ensure the background processor task is running."""
        if self._processor_task is None or self._processor_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._processor_task = loop.create_task(self._batch_processor())
                logger.info("Batch processor background task started")
            except RuntimeError:
                # This will only be called during the first request
                logger.warning("No running event loop found, will start processor during first request")
    
    async def submit_request(self, 
                           query: str, 
                           context: str, 
                           messages: List[Any],
                           conversation_id: str) -> asyncio.Future:
        """
        Submit an inference request to be batched with others.
        
        Args:
            query: The user query
            context: Context information (from documents or web search)
            messages: List of message objects for the LLM
            conversation_id: Unique ID for the conversation
            
        Returns:
            Future that will resolve to the LLM response
        """
        
        self._ensure_processor_task()
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        request_data = (query, context, messages, conversation_id, future)
        
        async with self.lock:
            self.requests.append(request_data)
            logger.debug(f"Request queued. Current batch size: {len(self.requests)}/{self.max_batch_size}")
            
            # If the max batch size is reached, process immediately
            if len(self.requests) >= self.max_batch_size:
                logger.info(f"Max batch size reached ({self.max_batch_size}). Processing batch immediately.")
                await self._process_batch()
        
        return future
    
    async def _batch_processor(self):
        """
        Background task that periodically processes accumulated requests.
        """
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                async with self.lock:
                    if self.requests:
                        await self._process_batch()
            except asyncio.CancelledError:
                logger.info("Batch processor task cancelled")
                break 
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    async def _process_batch(self):
        """
        Process the current batch of requests.
        """
        # Retrieve current batch and reset the requests list
        current_batch = self.requests
        self.requests = []
        
        if not current_batch:
            return
        
        batch_size = len(current_batch)
        logger.info(f"Processing batch of {batch_size} requests")
        
        # Extract data from the batch
        queries, contexts, messages_list, conversation_ids, futures = zip(*current_batch)
        
        # Offload the blocking batch inference to a separate thread
        try:
            start_time = time.time()
            results = await asyncio.to_thread(
                self._blocking_batch_inference, 
                queries, 
                contexts, 
                messages_list,
                conversation_ids
            )
            elapsed = time.time() - start_time
            logger.info(f"Batch inference completed in {elapsed:.2f}s for {batch_size} requests")
            
            # Set the result for each corresponding future
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            # Set exception for all futures in the batch
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    def _get_llm_instance(self):
        """Get or create the LLM instance (thread-safe)."""
        with self._llm_lock:
            if self._llm_instance is None:
                logger.info(f"Creating LLM instance with model {self.ollama_model}")
                self._llm_instance = ChatOllama(
                    model=self.ollama_model,
                    temperature=self.ollama_params["temperature"],
                    num_gpu=self.ollama_params["num_gpu"],
                    num_thread=self.ollama_params["num_thread"],
                )
            return self._llm_instance
    
    def _blocking_batch_inference(self, 
                           queries: List[str], 
                           contexts: List[str],
                           messages_list: List[List[Any]],
                           conversation_ids: List[str]) -> List[Any]:
        """
        Perform batch inference for multiple queries (blocking call).
        
        This runs in a separate thread to avoid blocking the event loop.
        
        Args:
            queries: List of user queries
            contexts: List of context information 
            messages_list: List of message lists for each query
            conversation_ids: List of conversation IDs
            
        Returns:
            List of inference results
        """
        llm = self._get_llm_instance()
        results = []
        
        # Process each request in the batch
        for i, (query, context, messages, conversation_id) in enumerate(
            zip(queries, contexts, messages_list, conversation_ids)
        ):
            try:
                # Log before calling the LLM
                logger.info(f"Processing request {i} (conv: {conversation_id}) with Ollama model")
                
                # Call the LLM with a timeout mechanism
                try:
                    # Use a direct ollama.chat call instead of langchain for better control
                    import ollama
                    
                    # Format messages for ollama.chat
                    ollama_messages = []
                    for msg in messages:
                        if isinstance(msg, SystemMessage):
                            ollama_messages.append({"role": "system", "content": msg.content})
                        elif isinstance(msg, HumanMessage):
                            ollama_messages.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            ollama_messages.append({"role": "assistant", "content": msg.content})
                    
                    logger.info(f"Calling Ollama API for request {i}")
                    response = ollama.chat(
                        model=self.ollama_model,
                        messages=ollama_messages,
                        options={
                            "temperature": self.ollama_params["temperature"],
                            "num_thread": self.ollama_params["num_thread"],
                        }
                    )
                    
                    # Convert response to AIMessage
                    result = AIMessage(content=response["message"]["content"])
                    logger.info(f"Successfully processed request {i}, got {len(result.content)} characters")
                except Exception as e:
                    logger.error(f"Error calling Ollama API: {e}")
                    # Fallback to langchain if direct API call fails
                    logger.info(f"Falling back to langchain for request {i}")
                    result = llm.invoke(messages)
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing request {i} (conv: {conversation_id}): {e}")
                # Create an error message
                error_response = AIMessage(content=f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}")
                results.append(error_response)
        
        return results                  