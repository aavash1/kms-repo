# src/core/inference/batch_inference.py

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import ollama
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a single inference request in a batch."""
    request_id: str
    query: str
    context: str
    messages: List[Dict[str, str]]  # Pre-formatted for Ollama
    conversation_id: str
    timestamp: float
    future: asyncio.Future

class BatchInferenceManager:
    """Manages batched inference requests for improved throughput and logs timings."""
    
    def __init__(
        self, 
        batch_interval: float = 0.1, 
        max_batch_size: int = 6,
        model: str = "gemma3:12b",
        quantization: str = "Q4_0",
        max_wait_time: float = 2.0
    ):
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.model = model
        self.quantization = quantization
        self.max_wait_time = max_wait_time
        
        # Pending requests keyed by request_id
        self.pending_requests: Dict[str, BatchRequest] = {}
        self.request_lock = asyncio.Lock()
        
        # Task for continuously checking and dispatching batches
        self.batch_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.average_batch_size = 0.0
        
        logger.info(f"BatchInferenceManager initialized with model={model}, max_batch_size={max_batch_size}")

    async def start(self):
        """Start the background batch‐processing loop."""
        if self.is_running:
            return
        self.is_running = True
        self.batch_task = asyncio.create_task(self._batch_processing_loop())
        logger.info("BatchInferenceManager started")
    
    async def stop(self):
        """Stop the batch‐processing loop and cancel all pending requests."""
        if not self.is_running:
            return
        self.is_running = False

        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # For any requests still pending, set an exception
        async with self.request_lock:
            for request in self.pending_requests.values():
                if not request.future.done():
                    request.future.set_exception(Exception("BatchInferenceManager stopped"))
            self.pending_requests.clear()
        
        logger.info("BatchInferenceManager stopped")
    
    async def submit_request(
        self,
        query: str,
        context: str,
        messages: List[Any],
        conversation_id: str
    ) -> asyncio.Future:
        """
        Submit a new inference request. Returns a Future that will be completed
        once the batched inference result is ready.
        """
        if not self.is_running:
            await self.start()
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()

        # Pre-format LangChain messages into Ollama format (list of {"role","content"})
        ollama_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = "system" if msg.__class__.__name__ == "SystemMessage" else "user"
            ollama_messages.append({"role": role, "content": msg.content})

        request = BatchRequest(
            request_id=request_id,
            query=query,
            context=context,
            messages=ollama_messages,
            conversation_id=conversation_id,
            timestamp=time.time(),
            future=future
        )
        
        # Enqueue under lock
        async with self.request_lock:
            self.pending_requests[request_id] = request
            self.total_requests += 1
        
        logger.debug(f"Submitted request {request_id} for conversation {conversation_id}")
        return future
    
    async def _batch_processing_loop(self):
        """Continuously checks for ready batches and processes them."""
        logger.info("Starting batch processing loop")
        
        while self.is_running:
            try:
                # If enough pending requests have piled up, process immediately
                async with self.request_lock:
                    count = len(self.pending_requests)
                if count >= self.max_batch_size:
                    await self._process_pending_requests()
                    continue

                # Otherwise wait for a short interval, then try again
                await asyncio.sleep(self.batch_interval)
                await self._process_pending_requests()
            except asyncio.CancelledError:
                logger.info("Batch processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
    
    async def _process_pending_requests(self):
        """Collect up to max_batch_size ready requests, expire any that waited too long."""
        async with self.request_lock:
            if not self.pending_requests:
                return
            
            current_time = time.time()
            ready_requests: List[BatchRequest] = []
            expired_requests: List[BatchRequest] = []
            
            for request in list(self.pending_requests.values()):
                if len(ready_requests) >= self.max_batch_size:
                    break
                
                if current_time - request.timestamp > self.max_wait_time:
                    expired_requests.append(request)
                else:
                    ready_requests.append(request)
            
            # Remove these requests from pending_requests
            for request in ready_requests + expired_requests:
                self.pending_requests.pop(request.request_id, None)
        
        # Handle expired requests first
        for request in expired_requests:
            if not request.future.done():
                request.future.set_exception(TimeoutError("Request expired"))
        
        # Now process the batch of ready requests
        if ready_requests:
            await self._process_batch(ready_requests)
    
    async def _process_batch(self, requests: List[BatchRequest]):
        """
        Process a batch of requests in parallel by invoking Ollama API concurrently.
        Logs the total time taken for the entire batch.
        """
        if not requests:
            return
        
        batch_start = time.time()
        logger.debug(f"Starting batch of {len(requests)} requests")

        # Create one coroutine per request
        coroutines = [self._process_single_request(req) for req in requests]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        batch_elapsed = time.time() - batch_start
        logger.info(f"Finished batch of {len(requests)} requests in {batch_elapsed:.3f} seconds")
        
        # Assign each result or exception back to its Future
        for req, res in zip(requests, results):
            if isinstance(res, Exception):
                logger.error(f"Error processing request {req.request_id}: {res}")
                if not req.future.done():
                    req.future.set_exception(res)
            else:
                if not req.future.done():
                    req.future.set_result(res)
        
        # Update metrics
        self.total_batches += 1
        self.average_batch_size = (
            (self.average_batch_size * (self.total_batches - 1) + len(requests))
            / self.total_batches
        )
    
    async def _process_single_request(self, request: BatchRequest) -> AIMessage:
        """
        Call Ollama.chat on this single pre-formatted request.
        Logs the time taken for this individual call.
        Returns an AIMessage, or a fallback AIMessage if an error occurs.
        """
        single_start = time.time()
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=request.messages,
                options={"temperature": 0.1}
            )
            elapsed = time.time() - single_start
            logger.debug(
                f"Request {request.request_id} (conversation {request.conversation_id}) "
                f"returned in {elapsed:.3f} seconds"
            )

            content = response.get("message", {}).get("content", "")
            return AIMessage(content=content)
        except Exception as e:
            elapsed = time.time() - single_start
            logger.error(
                f"Error in Ollama inference for request {request.request_id} "
                f"after {elapsed:.3f} seconds: {e}", 
                exc_info=True
            )
            # Return a safe fallback rather than leaving the Future hanging
            return AIMessage(content="Sorry, an error occurred while generating the response.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return some basic metrics about usage and pending queue size."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": self.average_batch_size,
            "pending_requests": len(self.pending_requests),
            "is_running": self.is_running,
            "model": self.model,
            "max_batch_size": self.max_batch_size
        }
    
    async def __aenter__(self):
        """Enter async context: start batch processing."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context: stop batch processing."""
        await self.stop()
