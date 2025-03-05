# src/core/services/summary_service.py
from src.core.processing.local_translator import LocalMarianTranslator
import ollama

class StatusCodeSummaryService:
    def __init__(self, vector_store, translator, db_connection):
        self.vector_store = vector_store
        self.translator = translator
        self.db_connection = db_connection

    async def get_or_generate_summary(self, status_code: str, query: str = None):
        """
        Get summary based on status code and optional search query.
        Uses existing error_code_ai_summary table.
        """
        try:
            # Case 1: Search with query - generate temporary summary
            if query:
                docs = self.vector_store.similarity_search(
                    query,
                    k=10,
                    filter={"status_code": status_code}
                )
                
                if not docs:
                    return {
                        "results": [],
                        "insight": "해당 상태 코드에 대한 관련 문서를 찾을 수 없습니다."
                    }
                
                # Generate temporary summary from matching documents
                return await self._generate_temp_summary(docs, query, status_code)

            # Case 2: No query - get default summary from database
            else:
                db_summary = await self._get_db_summary(status_code)
                if db_summary:
                    return {
                        "results": [],
                        "insight": db_summary
                    }
                
                return {
                    "results": [],
                    "insight": "해당 상태 코드에 대한 기본 요약이 없습니다."
                }

        except Exception as e:
            print(f"Error in summary generation: {e}")
            return {
                "results": [],
                "insight": "요약 생성 중 오류가 발생했습니다."
            }

    async def _get_db_summary(self, status_code: str):
        """Get existing summary from error_code_ai_summary table."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT summary FROM error_code_ai_summary WHERE error_code = %s",
                (status_code,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Database error: {e}")
            return None

    async def _generate_temp_summary(self, docs, query: str, status_code: str):
        """Generate temporary summary based on search results."""
        try:
            # Extract relevant snippets with keyword context
            results = [
                {
                    "id": doc.metadata.get("filename", "unknown"),
                    "snippet": self._get_snippet_with_keyword(doc.page_content, query)
                }
                for doc in docs
            ]
            
            combined_content = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""Analyze these specific documents related to status code {status_code}.

            Search query: {query}
            
            Documents:
            {combined_content}
            
            Provide a focused analysis of these specific cases."""
            
            response = ollama.chat(
                model='mistral:latest',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            english_summary = response['message']['content']
            korean_summary = self.translator.translate_text(english_summary)
            
            return {
                "results": results,
                "insight": korean_summary
            }
            
        except Exception as e:
            print(f"Error generating temporary summary: {e}")
            return {
                "results": results if 'results' in locals() else [],
                "insight": "임시 요약 생성 중 오류가 발생했습니다."
            }

    def _get_snippet_with_keyword(self, content: str, query: str, max_length: int = 500):
        """Generate snippet with keyword context."""
        if not query or not content:
            return content[:max_length] + "..."
        
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        positions = []
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                positions.append(pos)
        
        if not positions:
            return content[:max_length] + "..."
        
        pos = min(positions)
        start = max(0, pos - 200)
        end = min(len(content), pos + 300)
        
        if start > 0:
            space_before = content.rfind(" ", 0, start)
            if space_before != -1:
                start = space_before + 1
        
        if end < len(content):
            space_after = content.find(" ", end)
            if space_after != -1:
                end = space_after
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet