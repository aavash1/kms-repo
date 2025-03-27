# src/core/mariadb_db/mariadb_connector.py
import os
import mariadb
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv
import logging
import threading

logger = logging.getLogger(__name__)

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

class ConnectionPool:
    """Singleton connection pool for MariaDB connections."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.pool = []
                cls._instance.max_size = 10  # Configurable max pool size
                cls._instance.active = 0
                cls._instance.db_config = {
                    'host': DB_HOST,
                    'port': DB_PORT,
                    'user': DB_USER,
                    'password': DB_PASSWORD,
                    'database': DB_NAME
                }
        return cls._instance

    def get_connection(self):
        """Retrieve a connection from the pool or create a new one."""
        with self._lock:
            while self.pool:
                conn = self.pool.pop()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    self.active += 1
                    return conn
                except mariadb.Error:
                    # Stale connection, discard and try next
                    try:
                        conn.close()
                    except:
                        pass

            # No valid pooled connection, create new one
            if self.active >= self.max_size:
                logger.warning("Max connections reached, waiting for available connection")
                raise RuntimeError("Connection pool exhausted")
            try:
                conn = mariadb.connect(**self.db_config)
                self.active += 1
                logger.debug("Created new database connection")
                return conn
            except mariadb.Error as e:
                logger.error(f"Failed to create DB connection: {e}")
                raise

    def return_connection(self, conn):
        """Return a connection to the pool or close it if pool is full."""
        with self._lock:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    if len(self.pool) < self.max_size:
                        self.pool.append(conn)
                        logger.debug("Returned connection to pool")
                    else:
                        conn.close()
                        logger.debug("Pool full, closed connection")
                except mariadb.Error:
                    # Bad connection, close it
                    try:
                        conn.close()
                    except:
                        pass
                finally:
                    self.active -= 1

# Global connection pool
_pool = ConnectionPool()

class MariaDBConnector:
    def __init__(self, host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME):
        self.db_config = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database}
        self.conn = None
        self.cursor = None
        self._is_pooled = (host == DB_HOST and port == DB_PORT and user == DB_USER and
                          password == DB_PASSWORD and database == DB_NAME)

    def connect(self):
        """Establish a connection using the pool if default config, otherwise direct."""
        if self.conn and self.is_connection_active():
            return
        try:
            if self._is_pooled:
                self.conn = _pool.get_connection()
            else:
                self.conn = mariadb.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            logger.debug("Connected to MariaDB successfully!")
        except mariadb.Error as e:
            logger.error(f"MariaDB error: {str(e)}")
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def is_connection_active(self) -> bool:
        """Check if the current connection is active."""
        if not self.conn or not self.cursor:
            return False
        try:
            self.cursor.execute("SELECT 1")
            return True
        except mariadb.Error:
            return False

    def execute_query(self, query, params=None):
        """Execute a query, reconnecting if necessary."""
        if not self.conn or not self.is_connection_active():
            logger.warning("Database connection lost or not initialized. Reconnecting...")
            self.connect()
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except mariadb.Error as e:
            logger.error(f"Query execution error: {e}")
            if 'lost connection' in str(e).lower():
                self.connect()
                self.cursor.execute(query, params or ())
                return self.cursor.fetchall()
            raise RuntimeError(f"Database query failed: {e}")

    def fetch_dataframe(self, query, params=None):
        """Fetch query results as a pandas DataFrame."""
        rows = self.execute_query(query, params)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_unprocessed_troubleshooting_reports(self) -> pd.DataFrame:
        """Fetch troubleshooting reports with files and metadata."""
        query = """
            SELECT DISTINCT
                r.error_code_id,
                e.error_code_nm,
                e.explanation_en,
                e.message_en,
                e.recom_action_en,
                r.client_nm AS client_name,
                r.content,
                r.os_version_id AS os_version,
                r.resolve_id,
                atf.logical_nm,
                atf.physical_nm,
                atf.url
            FROM resolve r
            LEFT JOIN resolve_to_file rtf ON r.resolve_id = rtf.resolve_id
            LEFT JOIN attachment_files atf ON rtf.file_id = atf.file_id
            LEFT JOIN error_code e ON r.error_code_id = e.error_code_id
            WHERE atf.delete_yn = 'N'
        """
        logger.info(f"Executing query: {query}")
        df = self.fetch_dataframe(query)
        logger.info(f"Found {len(df)} troubleshooting reports")
        return df

    def get_files_by_error_code(self, error_code_id: str, logical_names: Optional[List[str]] = None) -> pd.DataFrame:
        query = """
            SELECT DISTINCT r.content, af.file_id, af.logical_nm, af.url
            FROM resolve r
            LEFT JOIN resolve_to_file rtf ON r.resolve_id = rtf.resolve_id
            LEFT JOIN attachment_files af ON rtf.file_id = af.file_id
            LEFT JOIN error_code e ON r.error_code_id = e.error_code_id
            WHERE e.error_code_nm = ?
            AND af.delete_yn = 'N'
        """
        params = [error_code_id]
        if logical_names:
            placeholders = ','.join(['?' for _ in logical_names])
            query += f" AND af.logical_nm IN ({placeholders})"
            params.extend(logical_names)
        query += " GROUP BY af.file_id, af.logical_nm, af.url, r.content"
        logger.info(f"Executing query: {query} with params: {params}")
        file_df = self.fetch_dataframe(query, tuple(params))
        logger.info(f"Found {len(file_df)} unique files and content for error_code_id: {error_code_id}")
        return file_df

    def get_files_by_logical_names(self, logical_names: List[str]) -> pd.DataFrame:
        if not logical_names:
            return pd.DataFrame()
        query = """
            SELECT file_id, logical_nm, url
            FROM attachment_files
            WHERE logical_nm COLLATE utf8mb4_general_ci IN ({})
            AND delete_yn = 'N'
        """.format(','.join(['?' for _ in logical_names]))
        logger.info(f"Executing query: {query} with params: {logical_names}")
        return self.fetch_dataframe(query, logical_names)

    def close(self):
        """Close or return the connection to the pool."""
        if self.cursor:
            try:
                self.cursor.close()
            except mariadb.Error as e:
                logger.error(f"Error closing cursor: {str(e)}")
            self.cursor = None
        if self.conn:
            if self._is_pooled:
                _pool.return_connection(self.conn)
                logger.debug("Returned connection to pool")
            else:
                try:
                    self.conn.close()
                    logger.debug("Closed direct connection")
                except mariadb.Error as e:
                    logger.error(f"Error closing connection: {str(e)}")
            self.conn = None

    def __del__(self):
        """Ensure connection is closed or returned on object destruction."""
        self.close()