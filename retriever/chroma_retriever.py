import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class ChromaRetrieverConfig:
    """Configuration for ChromaDB retriever."""
    collection_name: str = "documents"
    persist_directory: str = "./data/chromadb"
    top_k: int = 10
    similarity_threshold: float = 0.3
    enable_bm25: bool = True
    enable_fuzzy: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    fuzzy_threshold: float = 0.7


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "source": self.source
        }


class ChromaRetrieverError(Exception):
    """Exception raised for retriever errors."""
    pass


class BM25:
    """
    BM25 scoring implementation for keyword-based retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str]) -> None:
        """Fit BM25 on a corpus of documents."""
        self.documents = [self._tokenize(doc) for doc in documents]
        self.n_docs = len(self.documents)
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        
        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Calculate IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for a document given a query."""
        if doc_index >= len(self.documents):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc = self.documents[doc_index]
        doc_length = self.doc_lengths[doc_index]
        
        # Count term frequencies in document
        term_freqs = Counter(doc)
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = term_freqs.get(term, 0)
            idf = self.idf[term]
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents matching the query."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class FuzzyMatcher:
    """
    Fuzzy string matching for approximate search.
    """
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0
        
        distance = FuzzyMatcher.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def find_fuzzy_matches(
        query: str,
        documents: List[str],
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find documents with fuzzy matching terms."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        scores = []
        
        for i, doc in enumerate(documents):
            doc_terms = set(re.findall(r'\b\w+\b', doc.lower()))
            
            best_matches = []
            for q_term in query_terms:
                best_score = 0.0
                for d_term in doc_terms:
                    sim = FuzzyMatcher.similarity(q_term, d_term)
                    if sim > best_score:
                        best_score = sim
                
                if best_score >= threshold:
                    best_matches.append(best_score)
            
            if best_matches:
                avg_score = sum(best_matches) / len(query_terms)
                scores.append((i, avg_score))
        
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class ChromaRetriever:
    """
    ChromaDB retriever with dense, BM25, and fuzzy search capabilities.
    
    Supports:
    - Dense (semantic) retrieval using embeddings
    - BM25 keyword-based retrieval
    - Fuzzy matching for typo tolerance
    - Combined/hybrid scoring
    """
    
    def __init__(self, config: Optional[ChromaRetrieverConfig] = None):
        """
        Initialize the ChromaDB retriever.
        
        Args:
            config: Retriever configuration
        """
        self.config = config or ChromaRetrieverConfig()
        self._vector_store = None
        self._embedder = None
        self._bm25 = None
        self._documents_cache: List[str] = []
        self._ids_cache: List[str] = []
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize retriever components."""
        try:
            from embeddings import VectorStore, VectorStoreConfig, Embedder
            
            # Initialize vector store
            vs_config = VectorStoreConfig(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory
            )
            self._vector_store = VectorStore(vs_config)
            
            # Initialize embedder
            self._embedder = Embedder()
            
            # Initialize BM25 if enabled
            if self.config.enable_bm25:
                self._bm25 = BM25()
                self._rebuild_bm25_index()
            
            logger.info(f"ChromaDB retriever initialized. Documents: {self._vector_store.count()}")
            
        except Exception as e:
            raise ChromaRetrieverError(f"Failed to initialize retriever: {str(e)}")
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from vector store."""
        if not self.config.enable_bm25 or not self._bm25:
            return
        
        try:
            # Get all documents from vector store
            results = self._vector_store.get(limit=10000)
            
            self._documents_cache = [r.content for r in results]
            self._ids_cache = [r.id for r in results]
            
            if self._documents_cache:
                self._bm25.fit(self._documents_cache)
                logger.info(f"BM25 index built with {len(self._documents_cache)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {str(e)}")
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform dense (embedding-based) search."""
        try:
            query_embedding = self._embedder.embed_query(query)
            
            results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                where=filters
            )
            
            return [
                RetrievalResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    score=r.score,
                    retrieval_method="dense",
                    source=r.metadata.get("source")
                )
                for r in results
                if r.score >= self.config.similarity_threshold
            ]
            
        except Exception as e:
            logger.error(f"Dense search failed: {str(e)}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform BM25 keyword search."""
        if not self.config.enable_bm25 or not self._bm25 or not self._documents_cache:
            return []
        
        try:
            bm25_results = self._bm25.search(query, top_k)
            
            results = []
            for doc_idx, score in bm25_results:
                if doc_idx < len(self._documents_cache) and score > 0:
                    # Get full document info from vector store
                    doc_id = self._ids_cache[doc_idx]
                    doc_results = self._vector_store.get(ids=[doc_id])
                    
                    if doc_results:
                        doc = doc_results[0]
                        results.append(RetrievalResult(
                            id=doc.id,
                            content=doc.content,
                            metadata=doc.metadata,
                            score=score,
                            retrieval_method="bm25",
                            source=doc.metadata.get("source")
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []
    
    def _fuzzy_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform fuzzy search."""
        if not self.config.enable_fuzzy or not self._documents_cache:
            return []
        
        try:
            fuzzy_results = FuzzyMatcher.find_fuzzy_matches(
                query,
                self._documents_cache,
                threshold=self.config.fuzzy_threshold,
                top_k=top_k
            )
            
            results = []
            for doc_idx, score in fuzzy_results:
                if doc_idx < len(self._documents_cache):
                    doc_id = self._ids_cache[doc_idx]
                    doc_results = self._vector_store.get(ids=[doc_id])
                    
                    if doc_results:
                        doc = doc_results[0]
                        results.append(RetrievalResult(
                            id=doc.id,
                            content=doc.content,
                            metadata=doc.metadata,
                            score=score,
                            retrieval_method="fuzzy",
                            source=doc.metadata.get("source")
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy search failed: {str(e)}")
            return []
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        methods: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            methods: Retrieval methods to use (dense, bm25, fuzzy)
            
        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.config.top_k
        methods = methods or ["dense", "bm25", "fuzzy"]
        
        all_results: Dict[str, RetrievalResult] = {}
        
        # Dense search
        if "dense" in methods:
            dense_results = self._dense_search(query, top_k * 2, filters)
            for r in dense_results:
                if r.id not in all_results:
                    r.score *= self.config.dense_weight
                    all_results[r.id] = r
                else:
                    all_results[r.id].score += r.score * self.config.dense_weight
        
        # BM25 search
        if "bm25" in methods and self.config.enable_bm25:
            bm25_results = self._bm25_search(query, top_k * 2)
            
            # Normalize BM25 scores
            if bm25_results:
                max_bm25 = max(r.score for r in bm25_results)
                for r in bm25_results:
                    normalized_score = (r.score / max_bm25) * self.config.bm25_weight
                    if r.id not in all_results:
                        r.score = normalized_score
                        all_results[r.id] = r
                    else:
                        all_results[r.id].score += normalized_score
        
        # Fuzzy search
        if "fuzzy" in methods and self.config.enable_fuzzy:
            fuzzy_results = self._fuzzy_search(query, top_k)
            for r in fuzzy_results:
                if r.id not in all_results:
                    r.score *= 0.5  # Lower weight for fuzzy
                    all_results[r.id] = r
                else:
                    all_results[r.id].score += r.score * 0.2
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        logger.info(f"Retrieved {len(sorted_results[:top_k])} results for query: {query[:50]}...")
        
        return sorted_results[:top_k]
    
    def search_by_similarity(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Search using only dense similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of results above threshold
        """
        top_k = top_k or self.config.top_k
        threshold = threshold or self.config.similarity_threshold
        
        results = self._dense_search(query, top_k)
        return [r for r in results if r.score >= threshold]
    
    def get_document(self, doc_id: str) -> Optional[RetrievalResult]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            RetrievalResult or None
        """
        try:
            results = self._vector_store.get(ids=[doc_id])
            if results:
                doc = results[0]
                return RetrievalResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=1.0,
                    retrieval_method="direct",
                    source=doc.metadata.get("source")
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {str(e)}")
            return None
    
    def refresh_index(self) -> None:
        """Refresh BM25 index after new documents are added."""
        self._rebuild_bm25_index()
    
    @property
    def document_count(self) -> int:
        """Get total document count."""
        return self._vector_store.count()


def create_chroma_retriever(
    collection_name: str = "documents",
    **kwargs
) -> ChromaRetriever:
    """
    Factory function to create a ChromaDB retriever.
    
    Args:
        collection_name: Name of the collection
        **kwargs: Additional configuration options
        
    Returns:
        Configured ChromaRetriever instance
    """
    config = ChromaRetrieverConfig(collection_name=collection_name, **kwargs)
    return ChromaRetriever(config)

