import os
import mariadb
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

class MariaDBConnector:
    def __init__(self, host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME):
        self.db_config = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database}
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            logger.info(f"Attempting to connect to {self.db_config['host']}:{self.db_config['port']}")
            self.conn = mariadb.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            logger.debug("Connected to MariaDB successfully!")
        except mariadb.Error as e:
            logger.error(f"MariaDB error: {str(e)}")
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def is_connection_active(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        if self.conn is None:
            return False
        try:
            # Perform a lightweight query to test the connection
            self.cursor.execute("SELECT 1")
            return True
        except (mariadb.Error, AttributeError):
            return False

    def execute_query(self, query, params=None):
        if not self.conn or not self.is_connection_active():
            logger.warning("Database connection is closed or not initialized. Reconnecting...")
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
        rows = self.execute_query(query, params)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_files_by_error_code(self, error_code_id: str, logical_names: Optional[List[str]] = None) -> pd.DataFrame:
        query = """
        SELECT DISTINCT r.content, af.file_id, af.logical_nm, af.url
        FROM resolve r
        LEFT JOIN resolve_to_file rtf ON r.resolve_id = rtf.resolve_id
        LEFT JOIN attachment_files af ON rtf.file_id = af.file_id
        WHERE r.error_code_id = ? AND af.delete_yn = 'N'
        """
        params = [error_code_id]
        if logical_names:
            placeholders = ','.join(['?' for _ in logical_names])
            query += f" AND af.logical_nm IN ({placeholders})"
            params.extend(logical_names)
        query += " GROUP BY af.file_id, af.logical_nm, af.url, r.content"
        logger.info(f"Executing query: {query}")
        logger.info(f"Parameters: {params}")
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
        logger.info(f"Executing query: {query}")
        logger.info(f"Parameters: {logical_names}")
        return self.fetch_dataframe(query, logical_names)

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed.")