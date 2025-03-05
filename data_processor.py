# data_processor.py

from src.core.processing.error_code_processor import ErrorCodeProcessor
import logging

def process_error_codes(mode='batch', batch_size=64, db_batch_size=2000, where_clause=None):
    """
    Process error codes with specified parameters
    
    Args:
        mode (str): 'batch' for batch processing, 'single' for row-by-row
        batch_size (int): Size of translation batches (optimized for Tesla V100 32GB)
        db_batch_size (int): Size of database fetch batches
        where_clause (str): Optional WHERE clause for SQL query
    """
    try:
        processor = ErrorCodeProcessor(table_name="error_code")
        processor.run(
            mode=mode,
            batch_size=batch_size,
            db_batch_size=db_batch_size,
            where_clause=where_clause
        )
    except Exception as e:
        logging.error(f"Error in process_error_codes: {e}")
        print(f"Error in process_error_codes: {e}")

if __name__ == "__main__":
    # For testing with a small subset first
    process_error_codes(
        mode='batch',           # Use 'batch' for efficient processing, 'single' for row-by-row
        batch_size=64,         # Optimized for Tesla V100 32GB (can be increased up to 128)
        db_batch_size=2000,    # Optimized for database performance
        where_clause=None  # Start with a subset for testing
    )
    
    # After successful testing, you can process all records by removing the where_clause:
    # process_error_codes(
    #     mode='batch',
    #     batch_size=64,
    #     db_batch_size=2000,
    #     where_clause=None
    # )