# src/core/inference/batch_inference.py
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import ollama
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a single inference request in a batch"""
    request_id: str
    query: str
    context: str
    messages: List[Any]
    conversation_id: str
    timestamp: float
    future: asyncio.Future

class BatchInferenceManager:
    """Manages batched inference requests for improved throughput"""
    
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
        
        # Request management
        self.pending_requests: Dict[str, BatchRequest] = {}
        self.request_lock = asyncio.Lock()
        
        # Batch processing
        self.batch_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance metrics
        self.total_requests = 0
        self.total_batches = 0
        self.average_batch_size = 0.0
        
        logger.info(f"BatchInferenceManager initialized with model={model}, batch_size={max_batch_size}")
    
    async def start(self):
        """Start the batch processing loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_task = asyncio.create_task(self._batch_processing_loop())
        logger.info("BatchInferenceManager started")
    
    async def stop(self):
        """Stop the batch processing loop"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Complete any pending requests
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
        """Submit a request for batched inference"""
        if not self.is_running:
            await self.start()
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        request = BatchRequest(
            request_id=request_id,
            query=query,
            context=context,
            messages=messages,
            conversation_id=conversation_id,
            timestamp=time.time(),
            future=future
        )
        
        async with self.request_lock:
            self.pending_requests[request_id] = request
            self.total_requests += 1
        
        logger.debug(f"Submitted request {request_id} for conversation {conversation_id}")
        return future
    
    async def _batch_processing_loop(self):
        """Main batch processing loop"""
        logger.info("Starting batch processing loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._process_pending_requests()
            except asyncio.CancelledError:
                logger.info("Batch processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Wait before retrying
    
    async def _process_pending_requests(self):
        """Process pending requests in batches"""
        async with self.request_lock:
            if not self.pending_requests:
                return
            
            # Get requests ready for processing
            current_time = time.time()
            ready_requests = []
            expired_requests = []
            
            for request in list(self.pending_requests.values()):
                if len(ready_requests) >= self.max_batch_size:
                    break
                
                # Check if request has expired
                if current_time - request.timestamp > self.max_wait_time:
                    expired_requests.append(request)
                else:
                    ready_requests.append(request)
            
            # Remove processed requests from pending
            for request in ready_requests + expired_requests:
                self.pending_requests.pop(request.request_id, None)
        
        # Handle expired requests
        for request in expired_requests:
            if not request.future.done():
                request.future.set_exception(TimeoutError("Request expired"))
        
        # Process ready requests
        if ready_requests:
            await self._process_batch(ready_requests)
    
    async def _process_batch(self, requests: List[BatchRequest]):
        """Process a batch of requests"""
        if not requests:
            return
        
        logger.debug(f"Processing batch of {len(requests)} requests")
        
        try:
            # For now, process requests individually
            # In a real implementation, you might batch the actual model calls
            for request in requests:
                try:
                    result = await self._process_single_request(request)
                    if not request.future.done():
                        request.future.set_result(result)
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    if not request.future.done():
                        request.future.set_exception(e)
            
            # Update metrics
            self.total_batches += 1
            self.average_batch_size = (
                (self.average_batch_size * (self.total_batches - 1) + len(requests)) 
                / self.total_batches
            )
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            # Set exception for all requests in batch
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_single_request(self, request: BatchRequest) -> AIMessage:
        """Process a single inference request"""
        try:
            # Convert LangChain messages to Ollama format
            ollama_messages = []
            for msg in request.messages:
                if hasattr(msg, 'content'):
                    role = 'system' if msg.__class__.__name__ == 'SystemMessage' else 'user'
                    ollama_messages.append({
                        'role': role,
                        'content': msg.content
                    })
            
            # Call Ollama API
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=ollama_messages,
                options={'temperature': 0.1}
            )
            
            content = response.get('message', {}).get('content', '')
            return AIMessage(content=content)
            
        except Exception as e:
            logger.error(f"Error in Ollama inference: {e}")
            # Fallback response
            return AIMessage(content="죄송합니다. 응답 생성 중 오류가 발생했습니다.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'average_batch_size': self.average_batch_size,
            'pending_requests': len(self.pending_requests),
            'is_running': self.is_running,
            'model': self.model,
            'max_batch_size': self.max_batch_size
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()