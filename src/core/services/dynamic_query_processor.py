# src/core/services/dynamic_query_processor.py
import re
import time
import logging
import asyncio
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import threading
import os

logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for resource_path, resource_name in required_data:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource_name}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource_name}: {e}")

# Call this function at module level
ensure_nltk_data()


class DynamicCorpusAnalyzer:
    """Analyzes document corpus to extract dynamic vocabularies and patterns"""
    
    def __init__(self, min_doc_freq: int = 2, max_doc_freq: float = 0.8):
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.corpus_stats = None
        self.last_analysis = 0
        self.analysis_interval = 3600  # Reanalyze every hour
        self.lock = threading.Lock()
        
    def analyze_corpus(self, documents: List[str]) -> Dict:
        """Analyze corpus to extract dynamic vocabularies with robust error handling"""
        if not documents:
            return self._get_empty_stats()
            
        with self.lock:
            try:
                # **FIX 1: Validate documents before processing**
                valid_documents = [doc for doc in documents if doc and isinstance(doc, str) and len(doc.strip()) > 10]
                if len(valid_documents) < 2:
                    logger.warning(f"Insufficient valid documents for analysis: {len(valid_documents)}")
                    return self._get_empty_stats()
                
                # Extract domain-specific terms using TF-IDF
                domain_terms = self._extract_domain_terms(valid_documents)
                
                # Extract technical patterns
                technical_patterns = self._extract_technical_patterns(valid_documents)
                
                # Extract frequent n-grams
                important_phrases = self._extract_important_phrases(valid_documents)
                
                # Analyze document structure patterns
                structure_patterns = self._analyze_structure_patterns(valid_documents)
                
                # Build dynamic stop words based on corpus
                dynamic_stopwords = self._build_dynamic_stopwords(valid_documents)
                
                stats = {
                    'domain_terms': domain_terms,
                    'technical_patterns': technical_patterns,
                    'important_phrases': important_phrases,
                    'structure_patterns': structure_patterns,
                    'dynamic_stopwords': dynamic_stopwords,
                    'corpus_size': len(valid_documents),
                    'analysis_time': time.time()
                }
                
                self.corpus_stats = stats
                self.last_analysis = time.time()
                logger.info(f"Successfully analyzed corpus with {len(valid_documents)} documents")
                return stats
                
            except Exception as e:
                logger.error(f"Error analyzing corpus: {e}", exc_info=True)
                return self._get_empty_stats()
    
    def _extract_domain_terms(self, documents: List[str]) -> Set[str]:
        """Extract domain-specific terms using TF-IDF with enhanced error handling"""
        try:
            # **FIX 2: Enhanced document validation and cleaning**
            cleaned_docs = []
            for doc in documents:
                cleaned = self._clean_for_analysis(doc)
                if cleaned and len(cleaned.split()) >= 3:  # Minimum word requirement
                    cleaned_docs.append(cleaned)
            
            if len(cleaned_docs) < 2:
                logger.warning("Insufficient cleaned documents for TF-IDF analysis")
                return set()
            
            # **FIX 3: Dynamic parameter calculation based on actual document count**
            doc_count = len(cleaned_docs)
            min_df_value = max(1, min(3, doc_count // 20))  # More conservative approach
            max_df_value = min(0.9, max(0.6, 1.0 - (5 / doc_count))) if doc_count > 5 else 0.9
            
            logger.debug(f"TF-IDF parameters: doc_count={doc_count}, min_df={min_df_value}, max_df={max_df_value}")
            
            # **FIX 4: More robust TF-IDF configuration**
            vectorizer = TfidfVectorizer(
                max_features=200,  # Reduced for better performance
                min_df=min_df_value,
                max_df=max_df_value,
                ngram_range=(1, 2),
                token_pattern=r'\b[a-zA-Z가-힣][a-zA-Z가-힣0-9_.-]{1,}\b',  # More flexible pattern
                stop_words=None,  # We'll handle stopwords separately
                lowercase=True,
                strip_accents='unicode'
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms based on average TF-IDF scores
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_scores)[-50:]  # Top 50 terms
                
                domain_terms = set()
                for idx in top_indices:
                    term = feature_names[idx]
                    if self._is_meaningful_term(term):
                        domain_terms.add(term.lower())
                
                logger.debug(f"Extracted {len(domain_terms)} domain terms")
                return domain_terms
                
            except ValueError as ve:
                logger.warning(f"TF-IDF vectorization failed: {ve}")
                return set()
            
        except Exception as e:
            logger.error(f"Error extracting domain terms: {e}", exc_info=True)
            return set()
            
    def _extract_technical_patterns(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extract technical patterns from texts with enhanced error handling"""
        patterns = {
            'commands': [],
            'file_paths': [],
            'error_codes': [],
            'configuration': []
        }
        
        try:
            # **FIX 5: Safer text combination with size limits**
            combined_text = ' '.join(texts[:100])  # Limit to first 100 docs to prevent memory issues
            if len(combined_text) > 100000:  # Limit total text size
                combined_text = combined_text[:100000]
            
            # Extract command patterns
            command_patterns = re.findall(r'\b[a-z]+\s*-[a-z]+\b', combined_text, re.IGNORECASE)
            patterns['commands'] = list(set(command_patterns[:15]))  # Limit results
            
            # Extract file paths
            file_patterns = re.findall(r'[a-zA-Z]:\\[^\s]+|/[^\s]+', combined_text)
            patterns['file_paths'] = list(set(file_patterns[:10]))
            
            # Extract error codes
            error_patterns = re.findall(r'\berror[_\s]*code[_\s]*\d+\b|\bcode[_\s]*\d+\b', combined_text, re.IGNORECASE)
            patterns['error_codes'] = list(set(error_patterns[:10]))
            
            # Extract configuration terms
            config_patterns = re.findall(r'\bconfig\w*\b|\bsetup\w*\b|\binstall\w*\b', combined_text, re.IGNORECASE)
            patterns['configuration'] = list(set(config_patterns[:10]))
            
            logger.debug(f"Extracted technical patterns: {sum(len(v) for v in patterns.values())} total")
            
        except Exception as e:
            logger.error(f"Error extracting technical patterns: {e}", exc_info=True)
            # Return empty patterns on error
            patterns = {key: [] for key in patterns.keys()}
        
        return patterns
    
    def _extract_important_phrases(self, documents: List[str]) -> List[str]:
        """Extract important phrases using n-gram analysis with enhanced error handling"""
        try:
            # **FIX 6: Better document preprocessing for phrase extraction**
            cleaned_docs = []
            for doc in documents:
                cleaned = self._clean_for_analysis(doc)
                if cleaned and len(cleaned.split()) >= 5:  # Minimum for meaningful phrases
                    cleaned_docs.append(cleaned)
            
            if len(cleaned_docs) < 2:
                logger.warning("Insufficient documents for phrase extraction")
                return []
            
            phrases = []
            doc_count = len(cleaned_docs)
            min_df_value = max(1, min(2, doc_count // 30))
            
            if doc_count < min_df_value:
                return []
            
            # **FIX 7: More conservative bigram extraction**
            try:
                bigram_vectorizer = TfidfVectorizer(
                    ngram_range=(2, 2),
                    max_features=50,  # Reduced for better performance
                    min_df=min_df_value,
                    max_df=0.8,
                    token_pattern=r'\b[a-zA-Z가-힣][a-zA-Z가-힣0-9_.-]+\b',
                    lowercase=True
                )
                
                bigram_matrix = bigram_vectorizer.fit_transform(cleaned_docs)
                bigram_features = bigram_vectorizer.get_feature_names_out()
                bigram_scores = np.mean(bigram_matrix.toarray(), axis=0)
                
                # Get top bigrams
                top_bigram_indices = np.argsort(bigram_scores)[-15:]  # Top 15
                phrases.extend([bigram_features[i] for i in top_bigram_indices])
                
                logger.debug(f"Extracted {len(phrases)} important phrases")
                
            except ValueError as ve:
                logger.warning(f"Bigram extraction failed: {ve}")
                return []
            
            return phrases
            
        except Exception as e:
            logger.error(f"Error extracting important phrases: {e}", exc_info=True)
            return []
    
    def _analyze_structure_patterns(self, documents: List[str]) -> Dict[str, any]:
        """Analyze document structure patterns with enhanced error handling"""
        patterns = {
            'avg_doc_length': 0,
            'avg_sentences_per_doc': 0,
            'common_section_headers': [],
            'list_patterns': 0,
            'code_block_patterns': 0
        }
        
        try:
            if not documents:
                return patterns
            
            # **FIX 8: Safer pattern analysis with error handling**
            total_length = 0
            total_sentences = 0
            section_headers = []
            list_count = 0
            code_count = 0
            
            for doc in documents:
                try:
                    # Calculate length safely
                    doc_length = len(doc) if doc else 0
                    total_length += doc_length
                    
                    # Count sentences with better regex
                    sentences = len(re.findall(r'[.!?]+(?:\s|$)', doc)) if doc else 0
                    total_sentences += sentences
                    
                    # Extract headers safely
                    try:
                        headers = re.findall(r'^[A-Z][^.!?]*:$', doc, re.MULTILINE) if doc else []
                        section_headers.extend(headers[:5])  # Limit headers per document
                    except re.error:
                        pass
                    
                    # Count patterns safely
                    if doc:
                        list_count += len(re.findall(r'^\s*[-*•]\s+', doc, re.MULTILINE))
                        list_count += len(re.findall(r'^\s*\d+\.\s+', doc, re.MULTILINE))
                        code_count += len(re.findall(r'`[^`]+`', doc))
                        code_count += len(re.findall(r'```[\s\S]*?```', doc))
                        
                except Exception as doc_error:
                    logger.warning(f"Error analyzing document structure: {doc_error}")
                    continue
            
            doc_count = len(documents)
            patterns['avg_doc_length'] = total_length / doc_count if doc_count > 0 else 0
            patterns['avg_sentences_per_doc'] = total_sentences / doc_count if doc_count > 0 else 0
            patterns['common_section_headers'] = list(Counter(section_headers).most_common(5))
            patterns['list_patterns'] = list_count
            patterns['code_block_patterns'] = code_count
            
            logger.debug(f"Analyzed structure patterns for {doc_count} documents")
            
        except Exception as e:
            logger.error(f"Error analyzing structure patterns: {e}", exc_info=True)
        
        return patterns
    
    def _build_dynamic_stopwords(self, documents: List[str]) -> Set[str]:
        """Build dynamic stop words based on corpus analysis with enhanced error handling"""
        try:
            # **FIX 9: More robust stopword building**
            base_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Get language-specific stopwords safely
            try:
                english_stopwords = set(stopwords.words('english'))
                base_stopwords.update(english_stopwords)
            except LookupError:
                logger.warning("English stopwords not available")
            except Exception as e:
                logger.warning(f"Error loading stopwords: {e}")
            
            # Find words that appear in most documents
            word_doc_freq = defaultdict(int)
            total_docs = len(documents)
            
            for doc in documents:
                if not doc:
                    continue
                try:
                    # More robust word extraction
                    words = set(re.findall(r'\b[a-zA-Z가-힣]{2,8}\b', doc.lower()))  # Limit word length
                    for word in list(words)[:100]:  # Limit words per document
                        word_doc_freq[word] += 1
                except Exception as word_error:
                    logger.warning(f"Error processing words in document: {word_error}")
                    continue
            
            # Words appearing in more than 80% of documents are likely stop words
            dynamic_stopwords = base_stopwords.copy()
            for word, freq in word_doc_freq.items():
                if freq / total_docs > 0.8 and len(word) < 8:
                    dynamic_stopwords.add(word)
            
            # **FIX 10: Limit stopword set size**
            if len(dynamic_stopwords) > 150:
                sorted_stopwords = sorted(
                    [(word, word_doc_freq.get(word, 0)) for word in dynamic_stopwords],
                    key=lambda x: x[1], reverse=True
                )
                dynamic_stopwords = set([word for word, _ in sorted_stopwords[:150]])
            
            logger.debug(f"Built dynamic stopwords set with {len(dynamic_stopwords)} words")
            return dynamic_stopwords
            
        except Exception as e:
            logger.error(f"Error building dynamic stopwords: {e}", exc_info=True)
            return {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def _clean_for_analysis(self, text: str) -> str:
        """Clean text for analysis with enhanced error handling"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # **FIX 11: More robust text cleaning**
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove very short words and numbers-only tokens
            words = text.split()
            cleaned_words = []
            for word in words[:1000]:  # Limit words processed
                if (len(word) >= 2 and 
                    not word.isdigit() and 
                    re.match(r'^[a-zA-Z가-힣0-9_.-]+$', word)):
                    cleaned_words.append(word)
            
            result = ' '.join(cleaned_words)
            return result[:5000]  # Limit output length
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return ""
    
    def _is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful for domain analysis"""
        if not term or len(term) < 2:
            return False
        
        # Skip pure numbers
        if term.isdigit():
            return False
        
        # Skip very common words
        common_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'are', 'was', 'were'}
        if term.lower() in common_words:
            return False
        
        return True
    
    def _get_empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            'domain_terms': set(),
            'technical_patterns': {},
            'important_phrases': [],
            'structure_patterns': {},
            'dynamic_stopwords': set(),
            'corpus_size': 0,
            'analysis_time': time.time()
        }


class DynamicQueryProcessor:
    """Dynamic query processor that adapts to corpus content with enhanced error handling"""
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.analyzer = DynamicCorpusAnalyzer()
        self.last_corpus_update = 0
        self.update_interval = 3600  # Update every hour
        self.corpus_cache = None
        
    async def adaptive_preprocess_query(self, query: str) -> str:
        """Dynamically preprocess query based on current corpus with enhanced error handling"""
        if not query or not query.strip():
            return query
        
        try:
            # **FIX 12: Safe corpus analysis update**
            await self._update_corpus_analysis_if_needed()
            
            # Clean and enhance query
            processed_query = self._dynamic_query_cleaning(query)
            
            # Add context-aware expansions
            expanded_query = await self._context_aware_expansion(processed_query)
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error in adaptive preprocessing: {e}", exc_info=True)
            return self._basic_query_cleaning(query)  # Fallback to basic cleaning
    
    def _adaptive_preprocess_query(self, query: str) -> str:
        """Basic adaptive query preprocessing - synchronous version"""
        if not query or not query.strip():
            return query
        
        try:
            # Clean and enhance query
            processed_query = self._basic_query_cleaning(query)
            
            # Get corpus insights if available
            stats = self.analyzer.corpus_stats
            if not stats:
                return processed_query
            
            # Apply dynamic preprocessing based on corpus
            words = processed_query.split()
            enhanced_words = []
            
            domain_terms = stats.get('domain_terms', set())
            dynamic_stopwords = stats.get('dynamic_stopwords', set())
            
            for word in words:
                word_lower = word.lower()
                # Keep word if it's not a dynamic stopword or if it's a domain term
                if (word_lower not in dynamic_stopwords or 
                    word_lower in domain_terms):
                    enhanced_words.append(word)
            
            if not enhanced_words:  # If all words were filtered, keep original
                return processed_query
            
            return ' '.join(enhanced_words)
            
        except Exception as e:
            logger.error(f"Error in adaptive preprocessing: {e}")
            return self._basic_query_cleaning(query)
    
    def _basic_query_cleaning(self, query: str) -> str:
        """Basic query cleaning when no corpus analysis is available"""
        try:
            # Very minimal cleaning - just basic preprocessing
            query = re.sub(r'\s+', ' ', query.strip())
            
            # Only remove very basic stop words that are truly universal
            basic_stops = {'a', 'an', 'the', 'is', 'are', 'was', 'were'}
            words = query.split()
            
            # Only filter if we have enough words
            if len(words) > 3:
                filtered_words = [w for w in words if w.lower() not in basic_stops]
                if filtered_words:  # Only use filtered if we have words left
                    return ' '.join(filtered_words)
            
            return query
            
        except Exception as e:
            logger.error(f"Error in basic query cleaning: {e}")
            return query  # Return original query if cleaning fails
    
    async def _update_corpus_analysis_if_needed(self):
        """Update corpus analysis if needed with enhanced error handling"""
        current_time = time.time()
        if current_time - self.last_corpus_update > self.update_interval:
            try:
                # **FIX 13: Enhanced corpus document sampling with validation**
                documents = await self._sample_corpus_documents()
                if documents and len(documents) >= 2:  # Minimum documents required
                    self.analyzer.analyze_corpus(documents)
                    self.last_corpus_update = current_time
                    logger.info(f"Updated corpus analysis with {len(documents)} documents")
                else:
                    logger.warning(f"Insufficient documents for corpus analysis: {len(documents) if documents else 0}")
                    
            except Exception as e:
                logger.error(f"Error updating corpus analysis: {e}", exc_info=True)
    
    async def _sample_corpus_documents(self) -> List[str]:
        """Sample documents from vector store with enhanced error handling"""
        try:
            documents = []
            
            # **FIX 14: Enhanced vector store access with multiple fallback strategies**
            if not self.vector_store:
                logger.warning("Vector store not available for corpus sampling")
                return []
            
            # Strategy 1: Try get method
            try:
                if hasattr(self.vector_store, 'get'):
                    result = self.vector_store.get(limit=500)  # Reduced limit
                    if isinstance(result, dict) and 'documents' in result:
                        documents = result['documents']
                        logger.debug(f"Retrieved {len(documents)} documents using get() method")
            except Exception as e:
                logger.debug(f"Vector store get() method failed: {e}")
            
            # Strategy 2: Try _collection access
            if not documents:
                try:
                    if hasattr(self.vector_store, '_collection'):
                        result = self.vector_store._collection.get(limit=500)
                        if isinstance(result, dict) and 'documents' in result:
                            documents = result['documents']
                            logger.debug(f"Retrieved {len(documents)} documents using _collection.get()")
                except Exception as e:
                    logger.debug(f"Vector store _collection access failed: {e}")
            
            # Strategy 3: Try similarity search fallback
            if not documents:
                try:
                    generic_queries = ["information", "data", "system"]
                    for query in generic_queries:
                        try:
                            docs = self.vector_store.similarity_search(query, k=50)
                            documents.extend([doc.page_content for doc in docs if hasattr(doc, 'page_content')])
                            if len(documents) >= 100:  # Stop when we have enough
                                break
                        except Exception as search_error:
                            logger.debug(f"Similarity search failed for query '{query}': {search_error}")
                            continue
                    logger.debug(f"Retrieved {len(documents)} documents using similarity search fallback")
                except Exception as e:
                    logger.warning(f"All fallback methods failed: {e}")
            
            # **FIX 15: Enhanced document validation and deduplication**
            if documents:
                # Clean and validate documents
                valid_docs = []
                seen_hashes = set()
                
                for doc in documents:
                    if not doc or not isinstance(doc, str):
                        continue
                    
                    # Basic cleaning
                    cleaned = doc.strip()
                    if len(cleaned) < 50:  # Minimum length requirement
                        continue
                    
                    # Simple deduplication using hash of first 200 characters
                    doc_hash = hash(cleaned[:200])
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)
                    
                    valid_docs.append(cleaned)
                    
                    # Limit total documents processed
                    if len(valid_docs) >= 300:
                        break
                
                logger.info(f"Validated {len(valid_docs)} documents for corpus analysis")
                return valid_docs
            
            logger.warning("No documents retrieved from vector store")
            return []
            
        except Exception as e:
            logger.error(f"Error sampling corpus documents: {e}", exc_info=True)
            return []
    
    def _dynamic_query_cleaning(self, query: str) -> str:
        """Clean query using dynamic vocabulary with enhanced error handling"""
        if not query:
            return query
        
        try:
            stats = self.analyzer.corpus_stats
            if not stats:
                return self._basic_query_cleaning(query)
            
            # Remove dynamic stop words
            dynamic_stopwords = stats.get('dynamic_stopwords', set())
            words = query.split()
            filtered_words = []
            
            for word in words:
                word_lower = word.lower()
                # Keep word if it's not a dynamic stopword or if it's a domain term
                if (word_lower not in dynamic_stopwords or 
                    word_lower in stats.get('domain_terms', set())):
                    filtered_words.append(word)
            
            if not filtered_words:  # If all words were filtered, keep original
                return query
            
            return ' '.join(filtered_words)
            
        except Exception as e:
            logger.error(f"Error in dynamic query cleaning: {e}")
            return self._basic_query_cleaning(query)
    
    async def _context_aware_expansion(self, query: str) -> str:
        """Expand query based on corpus context with enhanced error handling"""
        try:
            stats = self.analyzer.corpus_stats
            if not stats:
                return query
            
            expansions = []
            query_lower = query.lower()
            
            # **FIX 16: Safer expansion logic with limits**
            # Add related domain terms
            domain_terms = stats.get('domain_terms', set())
            for term in list(domain_terms)[:20]:  # Limit terms checked
                if self._is_related_term(query_lower, term):
                    expansions.append(term)
                    if len(expansions) >= 2:  # Limit expansions
                        break
            
            # Add related technical patterns
            technical_patterns = stats.get('technical_patterns', {})
            for category, terms in technical_patterns.items():
                for term in terms[:5]:  # Limit terms per category
                    if self._is_related_term(query_lower, term.lower()):
                        expansions.append(term)
                        if len(expansions) >= 3:
                            break
                if len(expansions) >= 3:
                    break
            
            if expansions:
                # Add expansions to query (limit to avoid overly long queries)
                expanded = f"{query} {' '.join(expansions[:2])}"
                return expanded[:500]  # Limit total query length
            
            return query
            
        except Exception as e:
            logger.error(f"Error in context-aware expansion: {e}")
            return query
    
    def _is_related_term(self, query: str, term: str) -> bool:
        """Check if a term is related to the query with enhanced error handling"""
        if not query or not term:
            return False
        
        try:
            # Simple relatedness check
            query_words = set(query.split())
            term_words = set(term.split())
            
            # Check for word overlap
            if query_words.intersection(term_words):
                return True
            
            # Check for substring relationships
            if any(word in term for word in query_words if len(word) > 3):
                return True
            
            if any(word in query for word in term_words if len(word) > 3):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking term relatedness: {e}")
            return False
    
    def get_corpus_insights(self) -> Dict:
        """Get insights about the current corpus with enhanced error handling"""
        try:
            if not self.analyzer.corpus_stats:
                return {"status": "No corpus analysis available"}
            
            stats = self.analyzer.corpus_stats
            return {
                "status": "Available",
                "corpus_size": stats.get('corpus_size', 0),
                "domain_terms_count": len(stats.get('domain_terms', set())),
                "technical_patterns_count": sum(len(patterns) for patterns in stats.get('technical_patterns', {}).values()),
                "important_phrases_count": len(stats.get('important_phrases', [])),
                "dynamic_stopwords_count": len(stats.get('dynamic_stopwords', set())),
                "last_analysis": stats.get('analysis_time', 0),
                "sample_domain_terms": list(stats.get('domain_terms', set()))[:10],
                "sample_stopwords": list(stats.get('dynamic_stopwords', set()))[:10]
            }
            
        except Exception as e:
            logger.error(f"Error getting corpus insights: {e}")
            return {"status": "Error retrieving insights", "error": str(e)}


class EnhancedRLFeatureExtractor:
    """Enhanced RL feature extractor with dynamic corpus insights and error handling"""
    
    def __init__(self, dynamic_processor):
        self.dynamic_processor = dynamic_processor
    
    def extract_features(self, query_embedding: np.ndarray, docs: List, dynamic_processor) -> np.ndarray:
        """Extract enhanced features including dynamic corpus insights with error handling"""
        try:
            features = []
            
            # **FIX 17: Enhanced feature extraction with validation**
            # Validate inputs
            if not isinstance(query_embedding, np.ndarray) or query_embedding.size == 0:
                logger.warning("Invalid query embedding provided")
                return np.zeros(15)  # Return default feature vector
            
            if not docs:
                logger.warning("No documents provided for feature extraction")
                return np.zeros(15)
            
            # Document diversity features
            diversity_score = self._calculate_diversity_features(docs)
            features.extend(diversity_score)
            
            # Dynamic processor features
            if (hasattr(dynamic_processor, 'analyzer') and 
                dynamic_processor.analyzer.corpus_stats):
                domain_features = self._calculate_domain_features(
                    docs, 
                    dynamic_processor.analyzer.corpus_stats.get('domain_terms', set())
                )
                features.extend(domain_features)
            else:
                features.extend([0.0, 0.0])  # Placeholder
            
            # Content quality features
            quality_features = self._calculate_quality_features(docs)
            features.extend(quality_features)
            
            # Add dynamic corpus features
            corpus_features = self._extract_corpus_features(docs, dynamic_processor)
            features.extend(corpus_features)
            
            # **FIX 18: Ensure consistent feature size with validation**
            target_size = 15
            while len(features) < target_size:
                features.append(0.0)
            
            result = np.array(features[:target_size], dtype=np.float32)
            
            # Validate result
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("Invalid feature values detected, using default")
                return np.zeros(target_size, dtype=np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}", exc_info=True)
            return np.zeros(15, dtype=np.float32)  # Return safe default
    
    def _calculate_diversity_features(self, docs: List) -> List[float]:
        """Calculate diversity among documents with enhanced error handling"""
        try:
            if len(docs) < 2:
                return [0.0, 0.0]
            
            similarities = []
            # **FIX 19: Limit comparisons to prevent performance issues**
            max_comparisons = min(len(docs), 10)  # Limit to prevent O(n²) explosion
            
            for i in range(max_comparisons):
                for j in range(i+1, max_comparisons):
                    if i < len(docs) and j < len(docs):
                        try:
                            sim = self._simple_similarity(
                                docs[i].page_content, 
                                docs[j].page_content
                            )
                            similarities.append(sim)
                        except (AttributeError, IndexError) as e:
                            logger.warning(f"Error calculating similarity: {e}")
                            continue
            
            avg_sim = np.mean(similarities) if similarities else 0.0
            diversity = max(0.0, min(1.0, 1.0 - avg_sim))  # Clamp to [0,1]
            
            # Count unique sources safely
            unique_sources = len(set(
                doc.metadata.get('filename', f'doc_{i}') 
                for i, doc in enumerate(docs) 
                if hasattr(doc, 'metadata')
            ))
            
            return [diversity, float(unique_sources)]
            
        except Exception as e:
            logger.error(f"Error calculating diversity features: {e}")
            return [0.0, 0.0]
    
    def _calculate_domain_features(self, docs: List, domain_terms: set) -> List[float]:
        """Calculate domain-specific features with enhanced error handling"""
        try:
            if not domain_terms or not docs:
                return [0.0, 0.0]
            
            domain_scores = []
            for doc in docs:
                try:
                    content_lower = getattr(doc, 'page_content', '').lower()
                    if not content_lower:
                        domain_scores.append(0.0)
                        continue
                        
                    domain_count = sum(1 for term in domain_terms if term in content_lower)
                    domain_scores.append(float(domain_count))
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Error processing document for domain features: {e}")
                    domain_scores.append(0.0)
            
            avg_domain_score = np.mean(domain_scores) if domain_scores else 0.0
            max_domain_score = max(domain_scores) if domain_scores else 0.0
            
            # Normalize scores
            avg_domain_score = min(avg_domain_score / 10.0, 1.0)
            max_domain_score = min(max_domain_score / 10.0, 1.0)
            
            return [float(avg_domain_score), float(max_domain_score)]
            
        except Exception as e:
            logger.error(f"Error calculating domain features: {e}")
            return [0.0, 0.0]
    
    def _calculate_quality_features(self, docs: List) -> List[float]:
        """Calculate content quality features with enhanced error handling"""
        try:
            features = []
            
            # **FIX 20: Enhanced quality feature calculation**
            # Average document length
            lengths = []
            for doc in docs:
                try:
                    content = getattr(doc, 'page_content', '')
                    lengths.append(len(content) if content else 0)
                except (AttributeError, TypeError):
                    lengths.append(0)
            
            avg_length = np.mean(lengths) if lengths else 0.0
            features.append(min(avg_length / 1000.0, 1.0))  # Normalize and clamp
            
            # Structure indicators
            structure_scores = []
            for doc in docs:
                try:
                    content = getattr(doc, 'page_content', '')
                    if not content:
                        structure_scores.append(0)
                        continue
                        
                    structure_count = (
                        content.count('\n') + 
                        content.count(':') + 
                        content.count('-') + 
                        content.count('•')
                    )
                    structure_scores.append(structure_count)
                except (AttributeError, TypeError):
                    structure_scores.append(0)
            
            avg_structure = np.mean(structure_scores) if structure_scores else 0.0
            features.append(min(avg_structure / 10.0, 1.0))  # Normalize and clamp
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating quality features: {e}")
            return [0.0, 0.0]
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation with enhanced error handling"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # **FIX 21: More robust similarity calculation**
            words1 = set(text1.lower().split()[:100])  # Limit words processed
            words2 = set(text2.lower().split()[:100])
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _extract_corpus_features(self, docs: List, dynamic_processor) -> np.ndarray:
        """Extract features based on dynamic corpus analysis with enhanced error handling"""
        try:
            features = []
            
            if (hasattr(dynamic_processor, 'analyzer') and 
                dynamic_processor.analyzer.corpus_stats):
                stats = dynamic_processor.analyzer.corpus_stats
                
                # **FIX 22: Enhanced corpus feature extraction**
                # Technical pattern matches
                technical_patterns = stats.get('technical_patterns', {})
                tech_matches = 0
                
                for doc in docs:
                    try:
                        content = getattr(doc, 'page_content', '').lower()
                        if not content:
                            continue
                            
                        for category, terms in technical_patterns.items():
                            tech_matches += sum(
                                1 for term in terms 
                                if term and term.lower() in content
                            )
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Error processing document for tech patterns: {e}")
                        continue
                
                features.append(min(tech_matches / 10.0, 1.0))  # Normalize
                
                # Important phrase matches
                important_phrases = stats.get('important_phrases', [])
                phrase_matches = 0
                
                for doc in docs:
                    try:
                        content = getattr(doc, 'page_content', '').lower()
                        if not content:
                            continue
                            
                        phrase_matches += sum(
                            1 for phrase in important_phrases 
                            if phrase and phrase.lower() in content
                        )
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Error processing document for phrases: {e}")
                        continue
                
                features.append(min(phrase_matches / 5.0, 1.0))  # Normalize
            else:
                features.extend([0.0, 0.0])  # Default values
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting corpus features: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)