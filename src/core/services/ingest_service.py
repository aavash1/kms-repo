import os
import tempfile
import json
import ollama
import hashlib
import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import List, Optional, Dict, Any
from src.core.services.file_utils import process_file, flatten_embedding, clean_extracted_text, get_chromadb_collection, process_file_content
from src.core.services.file_download import download_file_from_url
from src.core.mariadb_db.mariadb_connector import MariaDBConnector
import logging
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self, db_connector: MariaDBConnector = None):
        self.sample_data_dir = os.path.join(os.getcwd(), "sample_data")
        os.makedirs(self.sample_data_dir, exist_ok=True)
        self.db_connector = db_connector or MariaDBConnector()

    def _sanitize_filename(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return f"{hashlib.md5(filename.encode('utf-8')).hexdigest()[:10]}{ext}"

    async def process_uploaded_files_optimized(self, files: List[UploadFile], status_code: str, metadata: Optional[str] = None, chunk_size=1000, chunk_overlap=200):
        results = []
        status_dir = os.path.join(self.sample_data_dir, status_code)
        os.makedirs(status_dir, exist_ok=True)
        metadata_dict = json.loads(metadata) if metadata else {}
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            logger.error("ChromaDB collection is not initialized!")
            return {"status": "error", "message": "ChromaDB collection is not initialized."}
        for file in files:
            try:
                safe_filename = self._sanitize_filename(file.filename)
                file_path = os.path.join(status_dir, safe_filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                extraction_result = process_file(file_path, chunk_size, chunk_overlap)
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({"filename": file.filename, "status": "error", "message": "No text was extracted from the file."})
                    continue
                base_metadata = {"filename": file.filename, "status_code": status_code, "source_id": file_path, **metadata_dict}
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file.filename}_chunk_{i}"
                    chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                    if embedding:
                        chromadb_collection.add(ids=[chunk_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                        chunk_ids.append(chunk_id)
                results.append({"filename": file.filename, "status": "success", "message": f"File stored successfully with {len(chunk_ids)} chunks.", "chunk_count": len(chunk_ids)})
            except Exception as e:
                logger.error(f"Error processing file '{file.filename}': {str(e)}")
                results.append({"filename": file.filename, "status": "error", "message": str(e)})
        successful = len([r for r in results if r["status"] == "success"])
        return {"status": "completed", "total_files": len(files), "successful": successful, "failed": len(files) - successful, "results": results}

    async def process_text_content(self, text_content: str, status_code: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200):
        try:
            if not text_content.strip():
                return {"content_id": f"content_{status_code}", "status": "error", "message": "No text content provided."}
            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                return {"content_id": f"content_{status_code}", "status": "error", "message": "ChromaDB collection is not initialized."}
            cleaned_text = clean_extracted_text(text_content)
            from src.core.utils.text_chunking import chunk_text
            chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
            if not chunks:
                return {"content_id": f"content_{status_code}", "status": "error", "message": "Failed to chunk text content."}
            base_metadata = {"content_type": "text", "status_code": status_code, **(metadata or {})}
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                content_id = f"content_{status_code}_{i}"
                chunk_metadata = {**base_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                if embedding:
                    chromadb_collection.add(ids=[content_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                    chunk_ids.append(content_id)
            return {"content_id": f"content_{status_code}", "status": "success", "message": f"Text content processed successfully with {len(chunk_ids)} chunks.", "chunk_count": len(chunk_ids)}
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return {"content_id": f"content_{status_code}", "status": "error", "message": str(e)}

    async def process_embedded_images(self, content: str, error_code_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        img_urls = re.findall(r'<img[^>]+src=["\'](.*?)["\']', content)
        if not img_urls:
            return results
        for url in img_urls:
            df = self.db_connector.fetch_dataframe("SELECT file_id, logical_nm, url FROM attachment_files WHERE url = ? AND delete_yn = 'N'", (url,))
            if df.empty:
                results.append({"url": url, "status": "error", "message": "Image not found in attachment_files"})
                continue
            for _, row in df.iterrows():
                file_id, logical_nm, file_url = row["file_id"], row["logical_nm"], row["url"]
                file_content = await download_file_from_url(file_url)
                if not file_content:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download image"})
                    continue
                image_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": file_url, "error_code_id": error_code_id, "source": "embedded_image", **metadata}
                image_result = await self.process_files_by_logical_names([logical_nm], error_code_id, image_metadata)
                results.extend(image_result)
        return results

    async def process_troubleshooting_report(self, logical_names: List[str], error_code_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        client_name, os_version = metadata.get("client_name"), metadata.get("os_version")
        logger.info(f"Fetching metadata for error_code_id: {error_code_id}")
        file_df = self.db_connector.get_files_by_error_code(error_code_id, logical_names)
        if file_df.empty:
            logger.warning(f"No file metadata found for error_code_id: {error_code_id} with logical_names: {logical_names}")
            return {"status": "warning", "message": f"No files or content found for error_code_id: {error_code_id}", "results": []}
        unique_contents = list(set(file_df["content"].dropna().unique()))
        file_df = file_df.drop(columns=["content"])
        results = []
        chromadb_collection = get_chromadb_collection()
        if not chromadb_collection:
            raise RuntimeError("ChromaDB collection is not initialized")
        processed_ids = set(chromadb_collection.get()["ids"])
        content_metadata = {"error_code_id": error_code_id, "client_name": client_name, "os_version": os_version}
        for content in unique_contents:
            content_id = f"content_{error_code_id}_{hash(content)}"
            if content_id in processed_ids:
                logger.info(f"Skipping duplicate content: {content_id}")
                continue
            image_results = await self.process_embedded_images(content, error_code_id, content_metadata)
            results.extend(image_results)
            cleaned_content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
            content_result = await self.process_text_content(cleaned_content, error_code_id, content_metadata)
            results.append(content_result)
        for _, row in file_df.iterrows():
            file_id, logical_nm, url = row.get("file_id"), row.get("logical_nm"), row.get("url")
            if not url:
                results.append({"logical_nm": logical_nm, "status": "error", "message": "No URL found for file"})
                continue
            chunk_id_base = f"{logical_nm}_chunk_0"
            if chunk_id_base in processed_ids:
                logger.info(f"Skipping duplicate file: {logical_nm}")
                continue
            logger.info(f"Downloading file: {logical_nm} from {url}")
            file_content = await download_file_from_url(url)
            if not file_content:
                results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file"})
                continue
            file_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": url, "error_code_id": error_code_id, "client_name": client_name, "os_version": os_version}
            file_result = await self.process_files_by_logical_names([logical_nm], error_code_id, file_metadata)
            results.extend(file_result)
        successful = sum(1 for r in results if r["status"] == "success")
        return {"status": "completed", "total_processed": len(results), "successful": successful, "failed": len(results) - successful, "error_code_id": error_code_id, "results": results}

    async def process_files_by_logical_names(self, logical_names: List[str], error_code: str, metadata: Dict[str, Any] = None, chunk_size=1000, chunk_overlap=200) -> List[Dict[str, Any]]:
        try:
            results = []
            file_df = self.db_connector.get_files_by_logical_names(logical_names)
            if file_df.empty:
                logger.warning(f"No files found for logical names: {logical_names}")
                return results
            chromadb_collection = get_chromadb_collection()
            if not chromadb_collection:
                raise RuntimeError("ChromaDB collection is not initialized")
            for _, row in file_df.iterrows():
                file_id, logical_nm, url = row.get("file_id"), row.get("logical_nm"), row.get("url")
                if not url:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No URL found for file"})
                    continue
                logger.info(f"Downloading file: {logical_nm} from {url}")
                file_content = await download_file_from_url(url)
                if not file_content:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "Failed to download file"})
                    continue
                extraction_result = process_file_content(file_content, logical_nm, chunk_size, chunk_overlap)
                chunks = extraction_result.get("chunks", [])
                if not chunks:
                    results.append({"logical_nm": logical_nm, "status": "error", "message": "No text extracted from the file"})
                    continue
                file_metadata = {"file_id": str(file_id), "logical_nm": logical_nm, "url": url, "error_code_id": error_code, **(metadata or {})}
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{logical_nm}_chunk_{i}"
                    chunk_metadata = {**file_metadata, "chunk_index": i, "chunk_count": len(chunks)}
                    embed_response = ollama.embed(model="mxbai-embed-large", input=chunk)
                    embedding = flatten_embedding(embed_response.get("embedding") or embed_response.get("embeddings"))
                    if embedding:
                        chromadb_collection.add(ids=[chunk_id], embeddings=[embedding], documents=[chunk], metadatas=[chunk_metadata])
                        chunk_ids.append(chunk_id)
                results.append({"logical_nm": logical_nm, "status": "success", "message": f"Processed {len(chunk_ids)} chunks", "chunk_count": len(chunk_ids)})
            return results
        except Exception as e:
            logger.error(f"Error processing files by logical names: {e}")
            raise