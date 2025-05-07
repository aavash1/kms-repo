import chromadb
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_chunks(document_id: str, filename: str):
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="C:/Users/admin/Desktop/KMSChatbot/chroma_db1")
        collection = client.get_collection("netbackup_docs")
        logger.info(f"Connected to collection: {collection.name}")

        # Query chunks for the file
        chunk_ids = [f"{filename}_chunk_{i}" for i in range(100)]  # Adjust range if expecting more chunks
        results = collection.get(ids=chunk_ids, include=["documents", "metadatas", "embeddings"])
        logger.info(f"Found {len(results['ids'])} chunks for {filename}")

        # Print chunk details
        for idx, (chunk_id, document, metadata) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
            print(f"\nChunk {idx} (ID: {chunk_id}):")
            print(f"Document: {document[:200]}..." if document else "No document")
            print("Metadata:")
            pprint(metadata)
            print(f"Embedding length: {len(results['embeddings'][idx]) if results['embeddings'] else 'None'}")

        return results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return None

if __name__ == "__main__":
    document_id = "TEST123"
    filename = "넷백업 2차 WBS-AI_part.xlsx"
    check_chunks(document_id, filename)