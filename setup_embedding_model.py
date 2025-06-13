# setup_embedding_model.py
#not to be pushed [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]
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

def setup_embedding_model():
    """One-time setup to download embedding model to standardized path"""
    load_dotenv()
    
    target_path = os.getenv('EMBEDDING_MODEL_PATH', r"C:\AI_Models\local_cache\models--mixedbread-ai--mxbai-embed-large")
    
    # Correct model names to try
    model_candidates = [
        'mixedbread-ai/mxbai-embed-large-v1',  # Most likely correct name
        'sentence-transformers/all-MiniLM-L6-v2',  # Fallback option
        'BAAI/bge-large-en-v1.5',  # Another good embedding model
    ]
    
    try:
        # Create directory structure
        os.makedirs(target_path, exist_ok=True)
        cache_dir = os.path.dirname(target_path)
        
        # Try each model candidate
        for model_name in model_candidates:
            try:
                print(f"Trying to download: {model_name}")
                
                # Add your token for private repos
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir,
                    device='cuda',
                    #token=''  # Your token
                )
                
                print(f"✓ Successfully downloaded: {model_name}")
                print(f"✓ Model cached at: {target_path}")
                print("You can now run your application offline!")
                
                return model_name  # Return successful model name
                
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
                continue
        
        print("✗ All model candidates failed")
        return None
        
    except Exception as e:
        print(f"✗ Setup failed: {e}")
        return None

if __name__ == "__main__":
    successful_model = setup_embedding_model()
    if successful_model:
        print(f"\nUpdate your code to use: {successful_model}")