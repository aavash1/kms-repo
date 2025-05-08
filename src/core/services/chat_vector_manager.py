# src/core/services/chat_vector_manager.py
import os
import logging
from datetime import datetime, timedelta
from fastapi import BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.core.services.file_utils import clean_expired_chat_vectors

logger = logging.getLogger(__name__)

class ChatVectorManager:
    """
    Manages the lifecycle of chat vectors, including scheduled cleanup.
    """
    def __init__(self, vector_storage_dir=None, retention_days=7):
        """
        Initialize the ChatVectorManager.
        
        Args:
            vector_storage_dir: Directory where chat vectors are stored
            retention_days: Number of days to retain chat vectors before cleanup
        """
        self.vector_storage_dir = vector_storage_dir or os.path.join(os.getcwd(), "chat_vectors")
        self.retention_days = retention_days
        self.scheduler = AsyncIOScheduler()
        self._setup_scheduler()
    
    def _setup_scheduler(self):
        """Set up weekly cleanup job."""
        # Schedule cleanup every Sunday at 2 AM
        self.scheduler.add_job(
            self._cleanup_expired_vectors,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="cleanup_chat_vectors",
            replace_existing=True
        )
        
        # Also add a manual trigger method for testing
        self.scheduler.add_job(
            self._cleanup_expired_vectors,
            'interval',
            days=7,
            id="backup_cleanup_chat_vectors",
            replace_existing=True
        )
    
    async def _cleanup_expired_vectors(self):
        """Clean up vectors older than the retention period."""
        try:
            logger.info(f"Starting scheduled cleanup of chat vectors older than {self.retention_days} days")
            clean_expired_chat_vectors(days=self.retention_days)
            logger.info("Completed scheduled cleanup of chat vectors")
        except Exception as e:
            logger.error(f"Error during chat vector cleanup: {e}", exc_info=True)
    
    def start(self):
        """Start the scheduler if it's not already running."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Chat vector cleanup scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler if it's running."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Chat vector cleanup scheduler shut down")
    
    def force_cleanup(self, days=None):
        """
        Manually trigger a cleanup operation.
        
        Args:
            days: Optional override for retention period
        """
        retention = days or self.retention_days
        clean_expired_chat_vectors(days=retention)
        logger.info(f"Manual cleanup of chat vectors older than {retention} days completed")


# Create a singleton instance
chat_vector_manager = ChatVectorManager()

def get_chat_vector_manager():
    """Get the singleton instance of ChatVectorManager."""
    return chat_vector_manager

