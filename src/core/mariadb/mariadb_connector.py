# src/core/mariadb/mariadb_connector.py
import os
import mariadb
import sys
import pandas as pd

import logging

logger = logging.getLogger(__name__)


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "kms_db")

class MariaDBConnector:
    def __init__(self, host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME):
        self.db_config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the MariaDB database."""
        try:
            self.conn = mariadb.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            logger.debug("Connected to MariaDB successfully!")
        except mariadb.Error as e:
            logger.error(f"Error connecting to MariaDB Platform: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def execute_query(self, query, params=None):
        """Execute a parameterized query and return the result."""
        if not self.cursor:
            self.connect()
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except mariadb.Error as e:
            logger.error(f"Query execution error: {e}")
            raise RuntimeError(f"Database query failed: {e}")
    
    def fetch_dataframe(self, query, params=None):
        """Execute a parameterized query and return the result as a Pandas DataFrame."""
        rows = self.execute_query(query, params)
        if not rows:
            return pd.DataFrame()
            
        # Convert list of dictionaries to DataFrame
        return pd.DataFrame(rows)

    def close(self):
        """Close the cursor and database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed.")

def get_file_metadata(status_code):
        """
        Fetch file metadata (file_id, file_path, file_type) from MariaDB using status_code.
        """
        db_connector = MariaDBConnector(
            host="your-mariadb-host",
            port=3306,
            user="your-username",
            password="your-password",
            database="your-database"
        )
        
        db_connector.connect()
        query = f"SELECT file_id, file_path, file_type FROM file_metadata WHERE status_code = '{status_code}'"
        metadata_df = db_connector.fetch_dataframe(query)
        db_connector.close()

        return metadata_df if not metadata_df.empty else None

# def get_file_metadata(status_code):
#     """
#     Fetch file metadata from MariaDB using status_code.
#     Uses parameterized query to prevent SQL injection.
    
#     Args:
#         status_code: Status/error code to look up
        
#     Returns:
#         DataFrame containing file metadata or None if no data
#     """
#     try:
#         db_connector = MariaDBConnector()
        
#         # Use parameterized query to prevent SQL injection
#         query = """
#         SELECT f.file_id, f.file_path, f.file_name, f.file_type, r.error_code_id
#         FROM files f
#         JOIN resolve_data r ON f.resolve_id = r.id
#         WHERE r.error_code_id = ?
#         """
        
#         metadata_df = db_connector.fetch_dataframe(query, (status_code,))
#         return metadata_df if not metadata_df.empty else None
        
#     except Exception as e:
#         logger.error(f"Error fetching file metadata: {e}")
#         return None
#     finally:
#         if 'db_connector' in locals():
#             db_connector.close()

def get_files_by_resolve_id(resolve_id):
    """
    Fetch files associated with a resolve ID.
    
    Args:
        resolve_id: The ID of the resolve record
        
    Returns:
        DataFrame containing file information
    """
    try:
        db_connector = MariaDBConnector()
        
        query = """
        SELECT file_id, file_path, file_name, file_type, is_image
        FROM files
        WHERE resolve_id = ?
        """
        
        files_df = db_connector.fetch_dataframe(query, (resolve_id,))
        return files_df
        
    except Exception as e:
        logger.error(f"Error fetching files for resolve ID {resolve_id}: {e}")
        return pd.DataFrame()
    finally:
        if 'db_connector' in locals():
            db_connector.close()
