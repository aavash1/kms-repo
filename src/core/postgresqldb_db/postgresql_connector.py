# src/core/postgresqldb_db/postgresql_connector.py
import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from typing import List, Dict, Any, Optional, Tuple
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
import threading
import time

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PostgreSQLConnector:
    """
    PostgreSQL database connector with connection pooling and error handling.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure one connection pool per application."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PostgreSQLConnector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize PostgreSQL connection pool if not already initialized."""
        if hasattr(self, '_initialized'):
            return
            
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "netbackup")
        self.username = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        
        # Connection pool settings
        self.min_connections = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "2"))
        self.max_connections = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20"))
        
        # Connection timeout settings
        self.connection_timeout = int(os.getenv("POSTGRES_CONNECTION_TIMEOUT", "30"))
        self.query_timeout = int(os.getenv("POSTGRES_QUERY_TIMEOUT", "300"))
        
        self.pool = None
        self._pool_lock = threading.Lock()
        self._initialized = True
        
        # Initialize connection pool
        self._create_connection_pool()
    
    def _create_connection_pool(self):
        """Create connection pool with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.pool = ThreadedConnectionPool(
                    minconn=self.min_connections,
                    maxconn=self.max_connections,
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    connect_timeout=self.connection_timeout
                )
                
                # Test the pool with a simple query
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                
                logger.info(f"PostgreSQL connection pool created successfully: {self.host}:{self.port}/{self.database}")
                return
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to create connection pool: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise ConnectionError(f"Failed to create PostgreSQL connection pool after {max_retries} attempts: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool using context manager.
        Automatically returns connection to pool when done.
        """
        if not self.pool:
            raise ConnectionError("Connection pool not initialized")
        
        conn = None
        try:
            conn = self.pool.getconn()
            if conn is None:
                raise ConnectionError("Unable to get connection from pool")
            
            # Set autocommit to False for transaction control
            conn.autocommit = False
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise e
        finally:
            if conn:
                try:
                    conn.commit()
                except:
                    pass
                self.pool.putconn(conn)
    
    def execute_query(self, query: str, params: Tuple = None, fetch: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch: Whether to fetch results (False for INSERT/UPDATE/DELETE)
            
        Returns:
            List of dictionaries representing rows, or None for non-fetch queries
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Set query timeout
                    cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                    
                    # Execute the main query
                    cursor.execute(query, params)
                    
                    if fetch:
                        results = cursor.fetchall()
                        logger.debug(f"Query executed successfully, returned {len(results)} rows")
                        return results
                    else:
                        affected_rows = cursor.rowcount
                        logger.debug(f"Query executed successfully, affected {affected_rows} rows")
                        return None
                        
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise e
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Total number of affected rows
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                    cursor.executemany(query, params_list)
                    affected_rows = cursor.rowcount
                    logger.debug(f"Batch query executed successfully, affected {affected_rows} rows")
                    return affected_rows
                    
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error executing batch query: {e}")
            logger.error(f"Query: {query}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error executing batch query: {e}")
            logger.error(f"Query: {query}")
            raise e
    
    def execute_transaction(self, queries_and_params: List[Tuple[str, Tuple]]) -> bool:
        """
        Execute multiple queries in a single transaction.
        
        Args:
            queries_and_params: List of (query, params) tuples
            
        Returns:
            True if transaction successful, raises exception otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                    
                    for query, params in queries_and_params:
                        cursor.execute(query, params)
                    
                    # Commit is handled by context manager
                    logger.debug(f"Transaction with {len(queries_and_params)} queries executed successfully")
                    return True
                    
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error in transaction: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in transaction: {e}")
            raise e
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection pool information.
        
        Returns:
            Dictionary with connection pool stats
        """
        if not self.pool:
            return {"status": "not_initialized"}
        
        try:
            # Note: ThreadedConnectionPool doesn't provide detailed stats
            # This is a basic implementation
            return {
                "status": "active",
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "connection_timeout": self.connection_timeout,
                "query_timeout": self.query_timeout
            }
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {"status": "error", "error": str(e)}
    
    def close_all_connections(self):
        """
        Close all connections in the pool.
        Should be called when shutting down the application.
        """
        if self.pool:
            try:
                self.pool.closeall()
                logger.info("All PostgreSQL connections closed")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close_all_connections()