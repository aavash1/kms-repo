# test_htmlcontent_handler.py

from src.core.file_handlers.htmlcontent_handler import HTMLContentHandler
import sys
from pathlib import Path
import os
import tempfile
import logging
import pandas as pd
import json
import re
from urllib.parse import urlparse
import time

# Import MariaDBConnector
try:
    from src.core.mariadb_db.mariadb_connector import MariaDBConnector
    MARIADB_AVAILABLE = True
except ImportError:
    MARIADB_AVAILABLE = False
    print("MariaDB connector not available")

# Import GraniteVisionExtractor
try:
    from src.core.ocr.granite_vision_extractor import GraniteVisionExtractor
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("GraniteVisionExtractor not available")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def extract_text_with_vision(image_content):
    """
    Extract text from an image using vision model.
    Only runs if vision model is available.
    """
    if not VISION_AVAILABLE:
        return ""
        
    # Initialize vision extractor
    vision = GraniteVisionExtractor()
    
    # Extract text
    try:
        return vision.extract_text_from_bytes(image_content)
    except Exception as e:
        logger.error(f"Error extracting text with vision model: {e}")
        return ""

if __name__ == "__main__":
    print("=== Processing Images for Text Extraction ===")
    
    # Check if MariaDB connector is available
    if not MARIADB_AVAILABLE:
        print("ERROR: MariaDB connector is not available")
        sys.exit(1)
    
    try:
        # Initialize database connector
        print("Connecting to database...")
        db = MariaDBConnector()
        db.connect()
        
        # Get all unprocessed troubleshooting reports
        print("Fetching troubleshooting reports...")
        reports_df = db.get_unprocessed_troubleshooting_reports()
        
        if reports_df.empty:
            print("No troubleshooting reports found with error_code_id=7")
            db.close()
            sys.exit(0)
        
        # Group by content to handle duplicate content entries
        print("Grouping reports by unique content...")
        
        # Create a hash of each content to identify duplicates
        reports_df['content_hash'] = reports_df['content'].apply(lambda x: hash(x))
        
        # Get unique content entries
        unique_contents = reports_df.drop_duplicates(subset=['content_hash'])
        print(f"Found {len(unique_contents)} unique content entries out of {len(reports_df)} total rows")
        
        # Initialize HTML content handler
        handler = HTMLContentHandler(languages=['ko', 'en'])
        
        # Process each unique content entry
        for i, (_, row) in enumerate(unique_contents.iterrows()):
            print(f"\n{'=' * 80}")
            print(f"CONTENT {i+1}/{len(unique_contents)}")
            print(f"{'=' * 80}")
            
            # Get HTML content
            html_content = row.get('content', '')
            
            # Extract text from HTML
            html_text = handler._extract_text_from_html(html_content)
            
            # Print the HTML content text
            print("\n--- HTML CONTENT ---")
            print(html_text)
            
            # Set base URL if available (for image downloads)
            base_url = None
            if 'url' in row and row['url']:
                try:
                    parsed_url = urlparse(row['url'])
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    handler.base_url = base_url
                except Exception:
                    pass
            
            # Extract images from HTML
            images = handler._extract_images_from_html(html_content)
            
            if not images:
                print("\nNo images found in this content")
                continue
                
            print(f"\n--- IMAGES ({len(images)}) ---")
            
            # Process each image
            for j, image_info in enumerate(images, 1):
                img_src = image_info['src']
                img_id = os.path.basename(img_src)
                print(f"\n--- Image {j}: {img_id} ---")
                
                # Download image
                image_content = handler._download_image(img_src)
                if not image_content:
                    print("Failed to download image")
                    continue
                
                # Extract text using vision model
                if VISION_AVAILABLE:
                    extracted_text = extract_text_with_vision(image_content)
                    if extracted_text:
                        print(extracted_text)
                    else:
                        print("No text extracted from this image")
                else:
                    print("Vision model not available")
            
            # Create a version of the HTML content with extracted text in place of image references
            # This is useful for reporting and documentation
            modified_content = html_text
            
            # For each image, replace its reference in the HTML text with the extracted text
            for j, image_info in enumerate(images, 1):
                img_src = image_info['src']
                img_id = os.path.basename(img_src)
                
                # Download image
                image_content = handler._download_image(img_src)
                if not image_content:
                    continue
                    
                # Extract text
                if VISION_AVAILABLE:
                    extracted_text = extract_text_with_vision(image_content)
                    if extracted_text and img_id in modified_content:
                        # Prepare the replacement text
                        replacement = f"{img_id}:\n{extracted_text}\n"
                        # Replace the image reference with the text
                        modified_content = modified_content.replace(img_id, replacement)
            
            # Print the modified content with extracted text
            print("\n--- CONTENT WITH EXTRACTED IMAGE TEXT ---")
            print(modified_content)
        
        # Close database connection
        db.close()
        print("\n=== Processing completed ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()