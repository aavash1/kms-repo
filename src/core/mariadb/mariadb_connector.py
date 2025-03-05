# src/core/mariadb/mariadb_connector.py

import mariadb
import sys
import pandas as pd

class MariaDBConnector:
    def __init__(self, host, port, user, password, database):
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
            self.cursor = self.conn.cursor()
            print("Connected to MariaDB successfully!")
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)
    
    def fetch_dataframe(self, query):
        """Execute the query and return the result as a Pandas DataFrame."""
        if not self.cursor:
            raise Exception("Not connected to database. Call connect() first.")
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        return df

    def close(self):
        """Close the cursor and database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

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