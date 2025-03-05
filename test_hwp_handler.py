import os
import sys
import unittest
from pathlib import Path

# Ensure project root directory is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.core.file_handlers.hwp_handler import HWPHandler

class TestHWPHandler(unittest.TestCase):
    """Test suite for HWPHandler class."""
    
    def setUp(self):
        """Set up test environment."""
        self.handler = HWPHandler()
        self.test_file = Path(PROJECT_ROOT) / "sample_data" / "test.hwp"

    def tearDown(self):
        """Clean up after test."""
        self.handler.__del__()

    def test_hwp_extraction(self):
        """Test extracting content from test.hwp file."""
        # Extract and print the content
        extracted_text = self.handler.extract_text(str(self.test_file))
        status_codes = self.handler.get_status_codes()
        print("Status code",status_codes)
        print("\n=== Extracted Content ===\n")
        print(extracted_text)
        print("\n========================\n")
        
        # Verify extraction worked
        self.assertIsInstance(extracted_text, str)
        self.assertGreater(len(extracted_text), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)