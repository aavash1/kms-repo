# src/core/services/static_data_cache.py
import logging
from typing import Dict, Optional
from src.core.mariadb_db.mariadb_connector import MariaDBConnector

logger = logging.getLogger(__name__)

class StaticDataCache:
    def __init__(self):
        self.error_code_data: Dict[str, Dict] = {}
        self.db_connector = MariaDBConnector()
        self._load_static_data()

    def _load_static_data(self):
        """
        Load static error code data from the MariaDB database into memory.
        """
        try:
            self.db_connector.connect()
            query = """
            SELECT error_code_id, error_code_nm, explanation_en, message_en, recom_action_en
            FROM error_code
            """
            logger.info("Fetching static error code data from MariaDB...")
            rows = self.db_connector.execute_query(query)

            for row in rows:
                error_code_id = str(row["error_code_id"])
                error_code_nm = str(row["error_code_nm"])
                self.error_code_data[error_code_id] = {  # Use error_code_id as key
                    "error_code_id": error_code_id,
                    "error_code_nm": error_code_nm,
                    "explanation_en": row["explanation_en"] or "",
                    "message_en": row["message_en"] or "",
                    "recom_action_en": row["recom_action_en"] or ""
                }

            logger.info(f"Loaded {len(self.error_code_data)} error code entries into static cache.")

        except Exception as e:
            logger.error(f"Failed to load static error code data from MariaDB: {e}")
            self.error_code_data = {
                "0": {"error_code_id": "0", "error_code_nm": "0", "explanation_en": "Explanation for error code 0", "message_en": "Message for error code 0", "recom_action_en": "Recommended action for error code 0"},
                "2": {"error_code_id": "2", "error_code_nm": "2", "explanation_en": "Explanation for error code 1", "message_en": "Message for error code 1", "recom_action_en": "Recommended action for error code 1"}
            }
            logger.info(f"Fallback: Loaded {len(self.error_code_data)} dummy error code entries into static cache.")

        finally:
            self.db_connector.close()

    def get_error_code_info(self, error_code_nm: str) -> Optional[Dict]:
        """
        Retrieve static error code information by error_code_nm or error_code_id.

        Args:
            error_code_nm (str): The error code/status code.

        Returns:
            dict: Static data for the error code, or None if not found.
        """
        # Try by error_code_nm first, then by error_code_id
        for data in self.error_code_data.values():
            if data["error_code_nm"] == error_code_nm:
                return data
        return None

    def refresh_static_data(self):
        """
        Refresh the static error code data from the database.
        """
        self.error_code_data.clear()
        self._load_static_data()

# Singleton instance
static_data_cache = StaticDataCache()