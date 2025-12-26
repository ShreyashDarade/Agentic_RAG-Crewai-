"""
Advanced Retriever - Production Grade with Milvus Cloud

Features:
- Milvus Cloud with HNSW indexing
- RRF (Reciprocal Rank Fusion)
- Cross-encoder re-ranking
- MMR diversity selection
- Multi-query retrieval
- Multi-modal search
"""

import logging
import re
import math
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRetrieverConfig:
    """Configuration for advanced retriever."""
    collection_name: str = "documents"
    
    # Basic retrieval
    top_k: int = 10
    similarity_threshold: float = 0.3
    
    # Multi-query settings
    enable_multi_query: bool = True
    num_query_variations: int = 3
    
    # RRF Fusion
    fusion_method: str = "rrf"
    rrf_k: int = 60
    
    # Re-ranking
    enable_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    
    # Diversity (MMR)
    enable_diversity: bool = True
    diversity_lambda: float = 0.5
    
    # BM25
    enable_bm25: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    
    # Fuzzy matching
    enable_fuzzy: bool = True
    fuzzy_threshold: float = 0.7
    
    # Query expansion
    enable_query_expansion: bool = False


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    source: Optional[str] = None
    content_type: str = "text"
    rerank_score: Optional[float] = None
    linked_elements: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "source": self.source,
            "content_type": self.content_type,
            "rerank_score": self.rerank_score,
            "linked_elements": self.linked_elements,
            "parent_id": self.parent_id
        }


class AdvancedRetrieverError(Exception):
    """Exception for retriever errors."""
    pass


class BM25Index:
    """BM25 implementation for keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str], doc_ids: List[str]) -> None:
        """Build BM25 index."""
        self.documents = [self._tokenize(doc) for doc in documents]
        self.doc_ids = doc_ids
        self.n_docs = len(self.documents)
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        
        # Document frequencies
        self.doc_freqs = {}
        for doc in self.documents:
            for term in set(doc):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # IDF scores
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score."""
        if doc_idx >= len(self.documents):
            return 0.0
        
        query_terms = self._tokenize(query)
        doc = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
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
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for matching documents."""
        scores = [(self.doc_ids[i], self.score(query, i)) for i in range(self.n_docs)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in scores[:top_k] if score > 0]


class CrossEncoderReranker:
    """Cross-encoder for re-ranking."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._initialize()
    
    def _initialize(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._available = True
        except ImportError:
            logger.warning("sentence-transformers not available for re-ranking")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Re-rank documents."""
        if not self._available or not documents:
            return [(i, 0.0) for i in range(len(documents))]
        
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._model.predict(pairs)
            results = [(i, float(scores[i])) for i in range(len(scores))]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k] if top_k else results
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return [(i, 0.0) for i in range(len(documents))]


class AdvancedRetriever:
    """
    Production-grade retriever with Milvus Cloud.
    
    Features:
    - Multi-strategy retrieval
    - RRF fusion
    - Cross-encoder re-ranking
    - MMR diversity
    """
    
    def __init__(self, config: Optional[AdvancedRetrieverConfig] = None):
        self.config = config or AdvancedRetrieverConfig()
        
        self._embedder = None
        self._vector_store = None
        self._bm25 = None
        self._reranker = None
        
        self._documents_cache: List[str] = []
        self._ids_cache: List[str] = []
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize components."""
        try:
            self._initialize_embedder()
            self._initialize_vector_store()
            
            if self.config.enable_bm25:
                self._bm25 = BM25Index(
                    k1=self.config.bm25_k1,
                    b=self.config.bm25_b
                )
                self._rebuild_bm25_index()
            
            if self.config.enable_rerank:
                self._reranker = CrossEncoderReranker(self.config.rerank_model)
            
            logger.info("Advanced retriever initialized")
            
        except Exception as e:
            raise AdvancedRetrieverError(f"Initialization failed: {str(e)}")
    
    def _initialize_embedder(self) -> None:
        """Initialize OpenAI embedder."""
        try:
            import os
            from embeddings import OpenAIEmbedder
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._embedder = OpenAIEmbedder(api_key=api_key)
                logger.info("OpenAI embedder initialized")
            else:
                raise ValueError("OPENAI_API_KEY not set")
        except Exception as e:
            logger.error(f"Embedder initialization failed: {e}")
            raise
    
    def _initialize_vector_store(self) -> None:
        """Initialize Milvus Cloud store."""
        try:
            from embeddings import MilvusCloudStore, MilvusCloudConfig
            
            config = MilvusCloudConfig(
                collection_name=self.config.collection_name
            )
            self._vector_store = MilvusCloudStore(config)
            logger.info("Milvus Cloud store initialized")
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from Milvus."""
        if not self.config.enable_bm25 or not self._bm25 or not self._vector_store:
            return
        
        try:
            results = self._vector_store.get(limit=10000)
            
            self._documents_cache = [r.content for r in results]
            self._ids_cache = [r.id for r in results]
            
            if self._documents_cache:
                self._bm25.fit(self._documents_cache, self._ids_cache)
                logger.info(f"BM25 index built with {len(self._documents_cache)} documents")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        content_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Dense vector search via Milvus."""
        if not self._embedder or not self._vector_store:
            return []
        
        try:
            query_embedding = self._embedder.embed_query(query)
            
            results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                content_type=content_type
            )
            
            return [
                RetrievalResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    score=r.score,
                    retrieval_method="dense",
                    source=r.metadata.get("source"),
                    content_type=r.content_type,
                    linked_elements=r.linked_elements,
                    parent_id=r.parent_id
                )
                for r in results
                if r.score >= self.config.similarity_threshold
            ]
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25 keyword search."""
        if not self.config.enable_bm25 or not self._bm25 or not self._documents_cache:
            return []
        
        try:
            bm25_results = self._bm25.search(query, top_k)
            results = []
            
            for doc_id, score in bm25_results:
                if score > 0:
                    try:
                        docs = self._vector_store.get(ids=[doc_id])
                        if docs:
                            doc = docs[0]
                            results.append(RetrievalResult(
                                id=doc.id,
                                content=doc.content,
                                metadata=doc.metadata,
                                score=score,
                                retrieval_method="bm25",
                                source=doc.metadata.get("source"),
                                content_type=doc.content_type,
                                linked_elements=doc.linked_elements,
                                parent_id=doc.parent_id
                            ))
                    except:
                        pass
            
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _rrf_fusion(
        self,
        result_lists: List[List[RetrievalResult]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, RetrievalResult] = {}
        
        for results in result_lists:
            for rank, result in enumerate(results, 1):
                rrf_scores[result.id] += 1.0 / (k + rank)
                if result.id not in result_map or result.score > result_map[result.id].score:
                    result_map[result.id] = result
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused = []
        for doc_id in sorted_ids:
            result = result_map[doc_id]
            result.score = rrf_scores[doc_id]
            result.retrieval_method = "rrf_fusion"
            fused.append(result)
        
        return fused
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Re-rank with cross-encoder."""
        if not self.config.enable_rerank or not self._reranker or not self._reranker.is_available:
            return results[:top_k]
        
        try:
            documents = [r.content for r in results]
            reranked = self._reranker.rerank(query, documents, top_k)
            
            reranked_results = []
            for idx, score in reranked:
                result = results[idx]
                result.rerank_score = score
                result.score = score
                reranked_results.append(result)
            
            return reranked_results
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
    
    def _apply_mmr(
        self,
        query_embedding: List[float],
        results: List[RetrievalResult],
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[RetrievalResult]:
        """Maximum Marginal Relevance for diversity."""
        if not self.config.enable_diversity or len(results) <= top_k:
            return results[:top_k]
        
        if not self._embedder:
            return results[:top_k]
        
        try:
            import numpy as np
            
            # Get document embeddings
            doc_embeddings = []
            for result in results:
                emb = self._embedder.embed_query(result.content)
                doc_embeddings.append(emb)
            
            doc_embeddings = np.array(doc_embeddings)
            query_emb = np.array(query_embedding)
            
            # Normalize
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1
            doc_embeddings = doc_embeddings / doc_norms
            
            # Query similarities
            query_sims = np.dot(doc_embeddings, query_emb)
            
            selected = []
            remaining = list(range(len(results)))
            
            for _ in range(min(top_k, len(results))):
                if not remaining:
                    break
                
                mmr_scores = []
                for idx in remaining:
                    relevance = query_sims[idx]
                    max_sim = 0.0
                    
                    for sel_idx in selected:
                        sim = np.dot(doc_embeddings[idx], doc_embeddings[sel_idx])
                        max_sim = max(max_sim, sim)
                    
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append((idx, mmr))
                
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(best_idx)
                remaining.remove(best_idx)
            
            return [results[i] for i in selected]
            
        except Exception as e:
            logger.error(f"MMR failed: {e}")
            return results[:top_k]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations."""
        if not self.config.enable_multi_query:
            return [query]
        
        variations = [query]
        
        if "?" in query:
            variations.append(query.replace("?", "").strip())
        
        if len(query.split()) < 5 and not query.lower().startswith(("what", "how", "why")):
            variations.append(f"What is {query}")
        
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "have", "has"}
        keywords = [w for w in query.lower().split() if w not in stop_words]
        if keywords:
            variations.append(" ".join(keywords))
        
        return variations[:self.config.num_query_variations]
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        methods: Optional[List[str]] = None,
        content_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Search with fusion and re-ranking.
        
        Args:
            query: Search query
            top_k: Number of results
            methods: Search methods to use
            content_type: Filter by type
            
        Returns:
            List of RetrievalResult objects
        """
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.config.top_k
        methods = methods or ["dense", "bm25"]
        
        # Generate query variations
        queries = self._generate_query_variations(query)
        
        all_result_lists = []
        
        for q in queries:
            result_lists = []
            
            if "dense" in methods:
                dense_results = self._dense_search(q, top_k * 2, content_type)
                if dense_results:
                    result_lists.append(dense_results)
            
            if "bm25" in methods and self.config.enable_bm25:
                bm25_results = self._bm25_search(q, top_k * 2)
                if bm25_results:
                    result_lists.append(bm25_results)
            
            all_result_lists.extend(result_lists)
        
        if not all_result_lists:
            return []
        
        # RRF Fusion
        fused = self._rrf_fusion(all_result_lists, self.config.rrf_k)
        
        # Re-ranking
        if self.config.enable_rerank and self._reranker:
            fused = self._rerank(query, fused, top_k * 2)
        
        # MMR Diversity
        if self.config.enable_diversity and self._embedder:
            query_embedding = self._embedder.embed_query(query)
            fused = self._apply_mmr(
                query_embedding,
                fused,
                top_k,
                self.config.diversity_lambda
            )
        
        logger.info(f"Retrieved {len(fused[:top_k])} results for: {query[:50]}...")
        return fused[:top_k]
    
    def search_by_type(
        self,
        query: str,
        content_type: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Search for specific content type."""
        return self.search(query, top_k=top_k, content_type=content_type)
    
    def get(self, ids: List[str]) -> List[RetrievalResult]:
        """Get documents by ID."""
        try:
            results = self._vector_store.get(ids=ids)
            return [
                RetrievalResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    score=1.0,
                    retrieval_method="direct",
                    content_type=r.content_type,
                    linked_elements=r.linked_elements,
                    parent_id=r.parent_id
                )
                for r in results
            ]
        except:
            return []
    
    def get_linked_elements(self, chunk_id: str) -> List[RetrievalResult]:
        """Get linked elements."""
        try:
            results = self._vector_store.get_linked_elements(chunk_id)
            return [
                RetrievalResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    score=1.0,
                    retrieval_method="linked",
                    content_type=r.content_type
                )
                for r in results
            ]
        except:
            return []
    
    def refresh_index(self) -> None:
        """Refresh BM25 index."""
        self._rebuild_bm25_index()
    
    @property
    def document_count(self) -> int:
        if self._vector_store:
            return self._vector_store.count()
        return 0


def create_advanced_retriever(
    collection_name: str = "documents",
    **kwargs
) -> AdvancedRetriever:
    """Factory function."""
    config = AdvancedRetrieverConfig(collection_name=collection_name, **kwargs)
    return AdvancedRetriever(config)
