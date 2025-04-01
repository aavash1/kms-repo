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
import re
from langchain_core.runnables.config import RunnableConfig
import torch
import psutil
if torch.cuda.is_available():
    import pynvml
    pynvml.nvmlInit()


logger = logging.getLogger(__name__)

class BatchInferenceManager:
    """
    A manager that batches inference requests together and processes them
    efficiently on the GPU. It offloads the blocking batch inference using 
    asyncio.to_thread to keep the main event loop responsive.
    """
    def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 8, model: str = "gemma3:12b", max_concurrent: int = 10, quantization: str = "Q8_0"):
    #llama3:latest
    #def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 8, model: str = "llama3:latest"):
        """
        Initialize the batch inference manager.
        
        Args:
            batch_interval: Time window (in seconds) to wait before processing a batch.
            max_batch_size: Maximum number of requests to collect before processing immediately.
            model: The name of the Ollama model to use for inference.
            max_concurrent: Maximum number of concurrent requests allowed (throttling).
            quantization: Quantization level for the model (e.g., "Q4_0", "Q8_0").
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.requests = []  # List of tuples: (query, context, messages, conversation_id, future)
        self.lock = asyncio.Lock()  # Protects access to the requests list
        self.ollama_model = model  # Default model
        self.ollama_params = {
            "num_gpu": 1,
            "num_thread": 4,
            "temperature": 0.1,
            "quantization": quantization,
        }

        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Add a semaphore to limit concurrent requests
        self.max_concurrent = max_concurrent
        self.request_semaphore = asyncio.Semaphore(max_concurrent)
        
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
        
        async with self.request_semaphore:
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
        try:
            return await future
        finally:
                pass
    
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
            
            # Apply formatting and set the result for each corresponding future
            for result, future in zip(results, futures):
                if not future.done():
                    # Apply formatting improvements to the content
                    formatted_content = self._improve_formatting(result.content)
                    result.content = formatted_content
                    future.set_result(result)
            
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared")
                # Log memory stats
                device = torch.cuda.current_device()
                total_mem = torch.cuda.get_device_properties(device).total_memory
                reserved_mem = torch.cuda.memory_reserved(device)
                allocated_mem = torch.cuda.memory_allocated(device)
                free_mem = total_mem - reserved_mem
                logger.info(f"CUDA memory stats: total={total_mem/(1024**3):.2f}GB, "
                            f"reserved={reserved_mem/(1024**3):.2f}GB, "
                            f"allocated={allocated_mem/(1024**3):.2f}GB, "
                            f"free={free_mem/(1024**3):.2f}GB")
        
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            # Set exception for all futures in the batch
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    # In src/core/inference/batch_inference.py - Replace the _improve_formatting method

    async def shutdown(self):
        """Gracefully shut down the inference manager."""
        logger.info("Shutting down BatchInferenceManager...")
        
        # Cancel the processor task
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up any remaining requests
        async with self.lock:
            for _, _, _, _, future in self.requests:
                if not future.done():
                    future.cancel()
            self.requests = []
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("BatchInferenceManager shutdown complete")
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on available GPU memory."""
        if not torch.cuda.is_available():
            return
        
        device = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem = total_mem - torch.cuda.memory_reserved(device)
        free_gb = free_mem / (1024**3)
        
        # Adjust batch size based on available memory
        if free_gb < 4:  # Less than 4GB free
            new_size = max(1, self.max_batch_size // 2)
            if new_size != self.max_batch_size:
                logger.warning(f"Low GPU memory ({free_gb:.2f}GB free). Reducing batch size from {self.max_batch_size} to {new_size}")
                self.max_batch_size = new_size
        elif free_gb > 12 and self.max_batch_size < 10:  # Plenty of memory available
            new_size = min(10, self.max_batch_size + 1)
            logger.info(f"Sufficient GPU memory ({free_gb:.2f}GB free). Increasing batch size from {self.max_batch_size} to {new_size}")
            self.max_batch_size = new_size
    
    def _improve_formatting(self, content):
        """Format content with Streamlit-friendly markdown, but with more subtle styling."""
        # Clean any thinking artifacts
        content = self.clean_thinking_content(content)
        
        # Step 1: Format section headers properly for markdown, but more subtly
        # Convert plain headers to markdown headers (less aggressively)
        content = re.sub(r'([0-9]+\.\s*[\w\s가-힣]+:)', r'\n### \1', content)
        content = re.sub(r'(\*\*[\w\s가-힣]+\*\*:|\b[\w\s가-힣]{3,20}:)', r'\n### \1', content)
        
        # Replace common section titles with more subtle formatted versions (no emojis)
        section_titles = {
            "문제:": "### 문제:",
            "문제 분석:": "### 문제 분석:",
            "원인:": "### 원인:",
            "해결 방안:": "### 해결 방안:",
            "해결책:": "### 해결책:",
            "해결 방법:": "### 해결 방법:",
            "트러블슈팅:": "### 트러블슈팅:",
            "참고:": "### 참고:"
        }
        
        for old, new in section_titles.items():
            content = content.replace(old, new)
        
        # Step 2: Improve list formatting
        # Format numbered lists
        content = re.sub(r'([^\n])\s*([0-9]+\.\s)', r'\1\n\n\2', content)
        
        # Step 3: Format code blocks and commands
        # Find patterns that look like commands and wrap them in code blocks
        command_patterns = [
            (r'(/etc/hosts)', r'`\1`'),
            (r'(ping\s+[\w\.]+)', r'`\1`'),
            (r'(<install_path>.*?\.LOG)', r'`\1`'),
            (r'(nslookup)', r'`\1`')
        ]
        
        for pattern, replacement in command_patterns:
            content = re.sub(pattern, replacement, content)
        
        # Step 4: Ensure proper paragraph breaks
        # Add proper spacing for paragraphs
        paragraphs = content.split('\n')
        formatted_paragraphs = []
        
        for p in paragraphs:
            p = p.strip()
            if p:
                formatted_paragraphs.append(p)
        
        content = '\n\n'.join(formatted_paragraphs)
        
        # Step 5: Selective bolding of key terms (be more conservative)
        # Only bold the most important terms and only once per paragraph
        for term in ["NetBackup", "SQL Server"]:
            # Don't double-bold or bold within code blocks
            # And limit to first occurrence in each paragraph
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if term in para and "**" + term + "**" not in para and "`" + term + "`" not in para:
                    # Only bold the first occurrence
                    paragraphs[i] = para.replace(term, "**" + term + "**", 1)
            content = '\n\n'.join(paragraphs)
        
        # Step 6: Remove excessive whitespace
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        # Step 7: Add subtle dividers between major sections
        content = re.sub(r'\n### ', r'\n\n### ', content)
        
        return content
    
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
                    # Add other parameters from ollama_params
                    **({"quantization": self.ollama_params["quantization"]} if "quantization" in self.ollama_params else {})
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
                            **({"quantization": self.ollama_params["quantization"]} if "quantization" in self.ollama_params else {})
                        }
                    )
                    
                    # Convert response to AIMessage
                    #result = AIMessage(content=response["message"]["content"])
                    cleaned_content = self.clean_thinking_content(response["message"]["content"])
                    result = AIMessage(content=cleaned_content)
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

    def clean_thinking_content(self, content):
        """
        Remove <think>...</think> blocks from the content.
        This helps clean up responses from models like deepseek-r1:14b that include thinking process.
        """
        import re
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Remove other common thinking markers
        cleaned = re.sub(r'###.*?###', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'Thinking:.*?Answer:', '', cleaned, flags=re.DOTALL)
        # Clean up any excessive whitespace that might have been created
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        return cleaned                      