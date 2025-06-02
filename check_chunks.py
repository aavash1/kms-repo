# debug_chromadb.py - Find actual chunk IDs and fix collection access

import chromadb
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chromadb():
    """Debug ChromaDB to find actual data and chunk IDs"""
    try:
        # Initialize ChromaDB client with correct path
        client = chromadb.PersistentClient(path="./chroma_db1")  # Match your path
        
        print("=== ChromaDB Collections Debug ===")
        
        # Use new API for ChromaDB 0.6.0+
        try:
            collection = client.get_collection("netbackup_docs")
            print(f"Found collection: netbackup_docs")
            print(f"Collection ID: {collection.id}")
            
            count = collection.count()
            print(f"Document count: {count}")
            
            if count > 0:
                # Get first 10 documents to see structure
                sample = collection.get(limit=10, include=['metadatas', 'documents'])
                
                print(f"Sample IDs: {sample.get('ids', [])[:5]}")
                
                # Check if this is our target collection
                for idx, (doc_id, metadata) in enumerate(zip(
                    sample.get('ids', []), 
                    sample.get('metadatas', [])
                )):
                    if metadata and metadata.get('document_id') == 'TEST1':
                        print(f"\n*** FOUND TARGET DATA ***")
                        print(f"Chunk ID pattern: {doc_id}")
                        print(f"Metadata: {metadata}")
                        print(f"Document preview: {sample['documents'][idx][:200]}...")
                        break
                
                # Show all metadata keys to understand structure
                if sample.get('metadatas'):
                    all_keys = set()
                    for meta in sample['metadatas']:
                        if meta:
                            all_keys.update(meta.keys())
                    print(f"\nAll metadata keys found: {sorted(all_keys)}")
                    
            return [collection]
                
        except Exception as e:
            print(f"Error accessing netbackup_docs collection: {e}")
            return []
        
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
        return None

def query_by_metadata():
    """Query using metadata filters instead of IDs"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db1")
        collection = client.get_collection("netbackup_docs")
        
        print("\n=== Querying by Metadata ===")
        
        # Query by document_id
        results = collection.get(
            where={"document_id": "TEST1"},
            include=["documents", "metadatas"]
        )
        
        print(f"Found {len(results['ids'])} chunks with document_id='TEST1'")
        
        if results['ids']:
            for i, (chunk_id, metadata, document) in enumerate(zip(
                results['ids'][:3],  # First 3 chunks
                results['metadatas'][:3], 
                results['documents'][:3]
            )):
                print(f"\nChunk {i+1}:")
                print(f"  ID: {chunk_id}")
                print(f"  Chunk index: {metadata.get('chunk_index')}")
                print(f"  Filename: {metadata.get('logical_nm', 'N/A')}")
                print(f"  Content: {document[:150]}...")
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying by metadata: {e}")
        return None

def test_similarity_search():
    """Test similarity search functionality"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db1")
        collection = client.get_collection("netbackup_docs")
        
        print("\n=== Testing Similarity Search ===")
        
        # First, check what embedding model was used during ingestion
        sample = collection.get(limit=1, include=['embeddings'])
        if sample.get('embeddings') and len(sample['embeddings']) > 0:
            embedding_dim = len(sample['embeddings'][0])
            print(f"Collection embedding dimension: {embedding_dim}")
            
            # Determine which embedding model to use based on dimension
            if embedding_dim == 1024:
                print("Using nomic-embed-text (1024d) to match collection")
                # Try with manual embedding using the same model as ingestion
                try:
                    import ollama
                    query = "How to map from 2D to 3D"
                    
                    # Get embedding using the same model as ingestion
                    embedding_response = ollama.embeddings(
                        model="mxbai-embed-large",
                        prompt=query
                    )
                    query_embedding = embedding_response["embedding"]
                    
                    # Query with manual embedding
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    print(f"Similarity search results for '{query}':")
                    print(f"Found {len(results['ids'][0])} results")
                    
                    for i, (chunk_id, metadata, document, distance) in enumerate(zip(
                        results['ids'][0],
                        results['metadatas'][0],
                        results['documents'][0],
                        results['distances'][0]
                    )):
                        print(f"\nResult {i+1}:")
                        print(f"  ID: {chunk_id}")
                        print(f"  Distance: {distance:.4f}")
                        print(f"  Document ID: {metadata.get('document_id', 'N/A')}")
                        print(f"  Content: {document[:200]}...")
                    
                    return results
                    
                except ImportError:
                    print("ollama package not available for manual embedding")
                    return None
                    
            elif embedding_dim == 384:
                print("Collection uses 384d embeddings - trying mxbai-embed-large")
                # Test with text query (ChromaDB will handle embedding)
                query = "How to map from 2D to 3D"
                results = collection.query(
                    query_texts=[query],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                print(f"Found {len(results['ids'][0])} results")
                return results
            
        else:
            print("No embeddings found in collection")
            return None
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return None

if __name__ == "__main__":
    print("Starting ChromaDB debug...")
    
    # Step 1: Debug collections
    collections = debug_chromadb()
    
    # Step 2: Query by metadata
    metadata_results = query_by_metadata()
    
    # Step 3: Test similarity search
    search_results = test_similarity_search()
    
    print("\n=== Summary ===")
    print(f"Collections found: {len(collections) if collections else 0}")
    print(f"Chunks with TEST1: {len(metadata_results['ids']) if metadata_results else 0}")
    print(f"Similarity search works: {'Yes' if search_results else 'No'}")