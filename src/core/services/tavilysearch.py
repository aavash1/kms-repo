"""
TavilySearch module for KMSChatbot to handle web search integration.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

class TavilySearch:
    """
    Handles web search using Tavily API and implements routing logic
    between document-based answers and web search.
    """
    
    def __init__(self, llm=None):
        """
        Initialize TavilySearch with optional LLM.
        
        Args:
            llm: LLM instance to use for decision making and formatting results
        """
        # Set Tavily API key from environment
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(api_key=self.tavily_api_key, max_results=5)
        
        # Use provided LLM or create default
        self.llm = llm or ChatOllama(model="deepseek-r1:14b", temperature=0.1)
    
    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using Tavily API.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        try:
            results = await asyncio.to_thread(self.search_tool.invoke, {"query": query})
            return results
        except Exception as e:
            print(f"Error in Tavily search: {e}")
            return []
    
    async def decide_search_method(self, query: str, document_results: List[Any], 
                               threshold: float = 0.7) -> Dict[str, Any]:
        """
        Decide whether to use document results or perform web search.
        
        Args:
            query: User query
            document_results: Results from document search
            threshold: Confidence threshold for document results
            
        Returns:
            Dict containing decision and reasoning
        """
        # If no document results, use web search
        if not document_results:
            return {
                "use_web_search": True,
                "reason": "No relevant documents found in the knowledge base."
            }
        
        # Prompt for the LLM to decide if document results are sufficient
        decision_prompt = f"""
        You are evaluating if the retrieved documents sufficiently answer a user query.
        
        USER QUERY: {query}
        
        RETRIEVED DOCUMENTS:
        {document_results}
        
        Do these documents provide a complete and accurate answer to the user query?
        If they seem incomplete, outdated, or don't directly address the query, say "no".
        If they provide a good answer to the query, say "yes".
        
        First explain your reasoning, and then conclude with either "DECISION: YES" or "DECISION: NO".
        """
        
        decision_message = HumanMessage(content=decision_prompt)
        decision_result = await self.llm.ainvoke([decision_message])
        
        # Extract decision
        decision_text = decision_result.content.strip().lower()
        use_web_search = "decision: no" in decision_text
        
        return {
            "use_web_search": use_web_search,
            "reason": decision_text.split("decision:")[0].strip(),
            "full_reasoning": decision_text
        }
    
    def format_web_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format web search results into a coherent text.
        
        Args:
            results: List of search results from Tavily
            
        Returns:
            Formatted text of search results
        """
        if not results:
            return "No relevant information found on the web."
        
        formatted_results = "\n\n".join([
            f"SOURCE: {result.get('url', 'Unknown source')}\n"
            f"TITLE: {result.get('title', 'No title')}\n"
            f"CONTENT: {result.get('content', 'No content')}"
            for result in results
        ])
        
        return formatted_results
    
    async def generate_answer_from_web(self, query: str, web_results: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on web search results.
        
        Args:
            query: User query
            web_results: Results from web search
            
        Returns:
            Generated answer
        """
        formatted_results = self.format_web_search_results(web_results)
        
        # Create a system message that instructs responses in Korean
        system_message = SystemMessage(content="""
        당신은 NetBackup 시스템 전문가입니다. 사용자의 질문에 대한 웹 검색 결과를 바탕으로 답변을 제공합니다.
        반드시 한국어로 답변하되, 기술 용어와 명령어는 영어로 유지하세요.
        출처 URL을 반드시 포함시키고, 정확하고 쉽게 이해할 수 있는 답변을 제공하세요.
        """)
        
        answer_prompt = f"""
        다음 웹 검색 결과를 기반으로 사용자 질문에 답변해주세요:
        
        사용자 질문: {query}
        
        웹 검색 결과:
        {formatted_results}
        
        이 검색 결과를 바탕으로 사용자 질문에 대한 포괄적인 답변을 제공해주세요.
        검색 결과에 관련 정보가 없다면, 그 점을 인정하고 기존 지식을 바탕으로 최선의 답변을 제공하세요.
        
        답변 시작 부분에 "웹 검색 기반: "이라고 표시하여 이 답변이 웹 소스에서 온 것임을 나타내세요.
        답변 끝에 관련 출처 URL을 포함하세요.
        """
        
        answer_message = HumanMessage(content=answer_prompt)
        messages = [system_message, answer_message]
        
        answer_result = await self.llm.ainvoke(messages)
        
        return answer_result.content.strip()
    
    async def adaptive_search(self, query: str, document_results: List[Any], 
                        document_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method for adaptive search routing between documents and web.
        
        Args:
            query: User query
            document_results: Results from document search
            document_answer: Optional pre-generated answer from documents
            
        Returns:
            Dict with answer and metadata
        """
        # Decide whether to use document results or web search
        decision = await self.decide_search_method(query, document_results)
        
        if decision["use_web_search"]:
            # Perform web search
            web_results = await self.search_web(query)
            
            # Generate answer from web results
            answer = await self.generate_answer_from_web(query, web_results)
            
            return {
                "answer": answer,
                "source": "web",
                "web_results": web_results,
                "reasoning": decision["reason"]
            }
        else:
            # Use document answer if provided, otherwise generate one
            if not document_answer:
                # This would be handled by your existing document answering logic
                document_answer = "This is a placeholder for document-based answer generation."
            
            return {
                "answer": document_answer,
                "source": "documents",
                "reasoning": decision["reason"]
            }