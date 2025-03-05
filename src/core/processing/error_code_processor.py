# src/core/processing/error_code_processing.py
import sys
import pandas as pd
import time
from tqdm import tqdm
import logging
from datetime import datetime

from src.core.mariadb.mariadb_connector import MariaDBConnector
from src.core.processing.translator import Translator
from src.core.processing.local_translator import LocalMarianTranslator

class ErrorCodeProcessor:
    def __init__(self, table_name="error_code"):
        self.table_name = table_name
        self.db_connector = MariaDBConnector(
            host='192.168.100.36',
            port=3306,
            user='netbackup',
            password='Dsti123!',
            database='netbackup'
        )
        self.translator = LocalMarianTranslator()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for tracking progress and errors"""
        log_filename = f'translation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    def run(self, mode='batch', batch_size=64, db_batch_size=2000, where_clause=None):
        """
        Main method to process the data with flexible execution modes
        
        Args:
            mode (str): 'batch' for batch processing, 'single' for row-by-row
            batch_size (int): Size of translation batches
            db_batch_size (int): Size of database fetch batches
            where_clause (str): Optional WHERE clause for SQL query
        """
        try:
            self.db_connector.connect()
            
            # Construct query
            query = f"SELECT * FROM {self.table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            if mode == 'batch':
                self.process_in_batches(batch_size, db_batch_size, query)
            else:
                self.process_single_rows(query)
                
        except Exception as e:
            logging.error(f"Error in run method: {e}")
        finally:
            self.db_connector.close()

    def process_single_rows(self, query):
        """Process rows one by one (original method)"""
        df = self.db_connector.fetch_dataframe(query)
        total_rows = len(df)
        
        for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows"):
            try:
                df.at[index, 'explanation'] = self.translator.translate_text(row['explanation_en'])
                df.at[index, 'message'] = self.translator.translate_text(row['message_en'])
                df.at[index, 'recom_action'] = self.translator.translate_text(row['recom_action_en'])
                
                self.update_database(df.iloc[[index]])
                logging.info(f"Processed row {index + 1} of {total_rows}")
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")

    def process_in_batches(self, batch_size, db_batch_size, query):
        """Process data in efficient batches"""
        try:
            # Get total count using fetch_dataframe
            count_query = f"SELECT COUNT(*) as count FROM ({query}) as subquery"
            count_df = self.db_connector.fetch_dataframe(count_query)
            total_count = count_df.iloc[0]['count']
            logging.info(f"Total rows to process: {total_count}")

            # Process in database batches
            for offset in range(0, total_count, db_batch_size):
                batch_query = f"""
                    SELECT error_code_id, explanation_en, message_en, recom_action_en 
                    FROM ({query}) as subquery
                    LIMIT {db_batch_size} OFFSET {offset}
                """
                df_batch = self.db_connector.fetch_dataframe(batch_query)
                
                if df_batch.empty:
                    break

                # Process translation in smaller batches with progress bar
                with tqdm(total=len(df_batch), desc=f"Processing batch {offset//db_batch_size + 1}") as pbar:
                    for i in range(0, len(df_batch), batch_size):
                        sub_batch = df_batch.iloc[i:i + batch_size].copy()
                        try:
                            translations = self.translate_batch(sub_batch)
                            sub_batch['explanation'] = translations['explanation']
                            sub_batch['message'] = translations['message']
                            sub_batch['recom_action'] = translations['recom_action']
                            
                            self.update_database_batch(sub_batch)
                            
                            # Update progress
                            pbar.update(len(sub_batch))
                            current_progress = offset + i + len(sub_batch)
                            logging.info(f"Processed {current_progress}/{total_count} rows")
                            
                        except Exception as e:
                            logging.error(f"Error processing batch at offset {offset + i}: {e}")
                            continue

        except Exception as e:
            logging.error(f"Major error in batch processing: {e}")

    def translate_batch(self, df_batch):
        """Translate a batch of rows efficiently"""
        translations = {
            'explanation': [],
            'message': [],
            'recom_action': []
        }
        
        # Prepare lists for batch translation
        expl_texts = df_batch['explanation_en'].fillna('').tolist()
        msg_texts = df_batch['message_en'].fillna('').tolist()
        recom_texts = df_batch['recom_action_en'].fillna('').tolist()
        
        try:
            # Translate in batches
            translations['explanation'] = self.translator.translate_batch(expl_texts)
            translations['message'] = self.translator.translate_batch(msg_texts)
            translations['recom_action'] = self.translator.translate_batch(recom_texts)
        except Exception as e:
            logging.error(f"Translation error: {e}")
            translations['explanation'] = expl_texts
            translations['message'] = msg_texts
            translations['recom_action'] = recom_texts
        
        return translations

    def update_database(self, df):
        """Update single row in database (original method)"""
        cursor = self.db_connector.cursor
        update_query = f"""
            UPDATE {self.table_name}
            SET explanation = ?, message = ?, recom_action = ?
            WHERE error_code_id = ?
        """
        
        for index, row in df.iterrows():
            try:
                cursor.execute(
                    update_query,
                    (row['explanation'], row['message'], row['recom_action'], row['error_code_id'])
                )
            except Exception as e:
                logging.error(f"Error updating row with error_code_id {row['error_code_id']}: {e}")
        
        self.db_connector.conn.commit()

    def update_database_batch(self, df_batch):
        """Update database with a batch of translations"""
        cursor = self.db_connector.cursor
        
        update_query = f"""
            UPDATE {self.table_name}
            SET explanation = %s,
                message = %s,
                recom_action = %s
            WHERE error_code_id = %s
        """
        
        update_data = [
            (row['explanation'], row['message'], row['recom_action'], row['error_code_id'])
            for _, row in df_batch.iterrows()
        ]
        
        try:
            cursor.executemany(update_query, update_data)
            self.db_connector.conn.commit()
        except Exception as e:
            logging.error(f"Database update error: {e}")
            self.db_connector.conn.rollback()