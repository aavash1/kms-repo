# src/core/utils/template_handler.py

import os
import sys
import re
import subprocess
from pathlib import Path

import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import ollama
import chromadb

import warnings
import logging

import asyncio
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager

from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class TemplateHandler:
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / 'templates'
        
    def get_template_path(self, template_name: str) -> str:
        """Get full path for template file"""
        return str(self.template_dir / template_name)
    
    def create_custom_model(self, template_name: str, model_name: str):
        """Create custom Ollama model using template"""
        template_path = self.get_template_path(template_name)
        os.system(f"ollama create {model_name} -f {template_path}")

# main.py modifications

# Add these imports at the top
from src.core.utils.template_handler import TemplateHandler
import logging

# Add these configurations after existing imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_response.log'),
        logging.StreamHandler()
    ]
)

def validate_korean_response(response: str) -> bool:
    """Validate if response is primarily in Korean"""
    korean_char_count = len([c for c in response if ord('가') <= ord(c) <= ord('힣')])
    total_char_count = len(response.strip())
    return korean_char_count / total_char_count > 0.6

# Modify startup_event function
def startup_event():
    global pdf_handler, doc_handler, chromadb_collection, rag_chain, vector_store, global_prompt
    try:
        print("Starting up: initializing handlers and Chroma collection...")
        
        # Initialize template handler
        template_handler = TemplateHandler()
        
        # Create custom model if it doesn't exist
        model_name = "netbackup-ko-deepseek:14b"
        template_handler.create_custom_model("netbackup-ko.template", model_name)
        
        # Initialize LLM with custom model
        llm = ChatOllama(
            model=model_name,
            streaming=True,
            system="모든 답변은 반드시 한국어로 작성되어야 합니다."
        )
        
        # Rest of your existing initialization code...
        
        # Modify RAG chain to include validation
        rag_chain = (
            {"context": retriever, "query": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("Startup complete: All components including Korean language support initialized successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

# Update query endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is empty.")
    try:
        response = rag_chain.invoke(request.query)
        
        # Validate Korean response
        if not validate_korean_response(response):
            logging.warning("Response validation failed: Not primarily in Korean")
        
        return {"answer": response}
    except Exception as e:
        logging.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))