"""
Query Caching - High-Performance Caching for RAG Operations

This module provides multi-tier caching for:
1. Query embeddings (avoid re-computing expensive embeddings)
2. Retrieval results (cache chunks for identical queries)
3. LLM responses (for deterministic queries)

Supports:
- In-memory LRU cache (fast, limited size)
- Redis cache (distributed, persistent)
- Semantic cache (near-duplicate queries)

Usage:
    from src.observability.caching import QueryCache, CacheConfig

    cache = QueryCache(CacheConfig(
        embedding_cache_size=10000,
        result_cache_ttl_seconds=3600,
    ))
    
    # Check cache before computing
    cached_embedding = cache.get_embedding(query)
    if cached_embedding is None:
        embedding = embedder.embed(query)
        cache.set_embedding(query, embedding)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, Tuple
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# CACHE CONFIG
# =============================================================================

@dataclass(frozen=True)
class CacheConfig:
    """Configuration for query cache."""
    
    # Embedding cache
    embedding_cache_size: int = 10000  # Max cached embeddings
    embedding_cache_enabled: bool = True
    
    # Retrieval result cache
    result_cache_size: int = 5000  # Max cached results
    result_cache_ttl_seconds: int = 3600  # 1 hour TTL
    result_cache_enabled: bool = True
    
    # Semantic cache (for near-duplicate queries)
    semantic_cache_enabled: bool = False
    semantic_similarity_threshold: float = 0.95
    
    # Redis config (optional)
    redis_url: Optional[str] = None
    redis_prefix: str = "rag:"
    
    # Metrics
    track_hit_rate: bool = True


# =============================================================================
# LRU CACHE IMPLEMENTATION
# =============================================================================

class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(self, max_size: int, default_ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()  # key -> (value, expiry)
        self._lock = threading.Lock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache, returns None if not found or expired."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, expiry = self._cache[key]
            
            # Check expiry
            if expiry > 0 and time.time() > expiry:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value
    
    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl if ttl else 0
        
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    def delete(self, key: str):
        """Delete key from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        removed = 0
        
        with self._lock:
            keys_to_remove = [
                k for k, (v, expiry) in self._cache.items()
                if expiry > 0 and now > expiry
            ]
            for key in keys_to_remove:
                del self._cache[key]
                removed += 1
        
        return removed
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Current number of cached items."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

def make_embedding_key(text: str, model: str = "default") -> str:
    """Generate cache key for embeddings."""
    content = f"{model}:{text}"
    return f"emb:{hashlib.sha256(content.encode()).hexdigest()[:32]}"


def make_result_key(
    query: str,
    control_type: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Generate cache key for retrieval results."""
    content = json.dumps({
        "query": query,
        "control_type": control_type,
        "top_k": top_k,
    }, sort_keys=True)
    return f"res:{hashlib.sha256(content.encode()).hexdigest()[:32]}"


def make_llm_key(
    prompt: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Generate cache key for LLM responses (only for deterministic calls)."""
    if temperature > 0:
        return ""  # Don't cache non-deterministic calls
    
    content = json.dumps({
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
    }, sort_keys=True)
    return f"llm:{hashlib.sha256(content.encode()).hexdigest()[:32]}"


# =============================================================================
# QUERY CACHE
# =============================================================================

@dataclass
class CacheEntry:
    """A cached item with metadata."""
    value: Any
    created_at: float
    access_count: int = 0
    size_bytes: int = 0


class QueryCache:
    """
    Multi-tier cache for RAG operations.
    
    Provides separate caches for:
    - Embeddings (long-lived, expensive to compute)
    - Retrieval results (medium-lived, depends on data freshness)
    - LLM responses (short-lived, only for deterministic queries)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize caches
        self._embedding_cache: Optional[LRUCache[List[float]]] = None
        self._result_cache: Optional[LRUCache[Dict]] = None
        self._llm_cache: Optional[LRUCache[str]] = None
        
        if self.config.embedding_cache_enabled:
            self._embedding_cache = LRUCache(
                max_size=self.config.embedding_cache_size,
                default_ttl_seconds=None,  # Embeddings don't expire
            )
        
        if self.config.result_cache_enabled:
            self._result_cache = LRUCache(
                max_size=self.config.result_cache_size,
                default_ttl_seconds=self.config.result_cache_ttl_seconds,
            )
        
        # LLM cache with short TTL
        self._llm_cache = LRUCache(
            max_size=1000,
            default_ttl_seconds=300,  # 5 min TTL
        )
        
        # Redis client (lazy init)
        self._redis = None
    
    # -------------------------------------------------------------------------
    # EMBEDDING CACHE
    # -------------------------------------------------------------------------
    
    def get_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached embedding for text."""
        if not self._embedding_cache:
            return None
        
        key = make_embedding_key(text, model)
        return self._embedding_cache.get(key)
    
    def set_embedding(self, text: str, embedding: List[float], model: str = "default"):
        """Cache an embedding."""
        if not self._embedding_cache:
            return
        
        key = make_embedding_key(text, model)
        self._embedding_cache.set(key, embedding)
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "default",
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Get cached embeddings for batch of texts.
        
        Returns:
            Tuple of (cached_embeddings, missing_indices)
        """
        if not self._embedding_cache:
            return [], list(range(len(texts)))
        
        cached = []
        missing = []
        
        for i, text in enumerate(texts):
            emb = self.get_embedding(text, model)
            if emb is not None:
                cached.append(emb)
            else:
                missing.append(i)
        
        return cached, missing
    
    def set_embeddings_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: str = "default",
    ):
        """Cache a batch of embeddings."""
        for text, emb in zip(texts, embeddings):
            self.set_embedding(text, emb, model)
    
    # -------------------------------------------------------------------------
    # RESULT CACHE
    # -------------------------------------------------------------------------
    
    def get_results(
        self,
        query: str,
        control_type: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[Dict]:
        """Get cached retrieval results."""
        if not self._result_cache:
            return None
        
        key = make_result_key(query, control_type, top_k)
        return self._result_cache.get(key)
    
    def set_results(
        self,
        query: str,
        results: Dict,
        control_type: Optional[str] = None,
        top_k: int = 5,
    ):
        """Cache retrieval results."""
        if not self._result_cache:
            return
        
        key = make_result_key(query, control_type, top_k)
        self._result_cache.set(key, results)
    
    # -------------------------------------------------------------------------
    # LLM CACHE
    # -------------------------------------------------------------------------
    
    def get_llm_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Get cached LLM response (only for deterministic calls)."""
        if not self._llm_cache or temperature > 0:
            return None
        
        key = make_llm_key(prompt, model, temperature)
        if not key:
            return None
        
        return self._llm_cache.get(key)
    
    def set_llm_response(
        self,
        prompt: str,
        response: str,
        model: str,
        temperature: float = 0.0,
    ):
        """Cache LLM response (only for deterministic calls)."""
        if not self._llm_cache or temperature > 0:
            return
        
        key = make_llm_key(prompt, model, temperature)
        if key:
            self._llm_cache.set(key, response)
    
    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    
    def invalidate_results(self):
        """Invalidate all result cache (e.g., after data refresh)."""
        if self._result_cache:
            self._result_cache.clear()
            logger.info("Result cache invalidated")
    
    def cleanup(self):
        """Cleanup expired entries from all caches."""
        removed = 0
        if self._embedding_cache:
            removed += self._embedding_cache.cleanup_expired()
        if self._result_cache:
            removed += self._result_cache.cleanup_expired()
        if self._llm_cache:
            removed += self._llm_cache.cleanup_expired()
        
        if removed > 0:
            logger.debug(f"Cache cleanup removed {removed} expired entries")
        return removed
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}
        
        if self._embedding_cache:
            stats["embedding_cache"] = self._embedding_cache.stats()
        
        if self._result_cache:
            stats["result_cache"] = self._result_cache.stats()
        
        if self._llm_cache:
            stats["llm_cache"] = self._llm_cache.stats()
        
        # Calculate aggregate hit rate
        total_hits = sum(
            c.hits for c in [self._embedding_cache, self._result_cache, self._llm_cache]
            if c is not None
        )
        total_misses = sum(
            c.misses for c in [self._embedding_cache, self._result_cache, self._llm_cache]
            if c is not None
        )
        total = total_hits + total_misses
        stats["aggregate_hit_rate"] = round(total_hits / total, 4) if total > 0 else 0.0
        
        return stats


# =============================================================================
# BATCH EMBEDDING HELPER
# =============================================================================

class BatchEmbedder:
    """
    Efficient batch embedding with caching.
    
    Collects texts and embeds them in batches for efficiency,
    while using cache to avoid recomputing existing embeddings.
    """
    
    def __init__(
        self,
        embed_fn: callable,
        cache: Optional[QueryCache] = None,
        batch_size: int = 32,
        model: str = "default",
    ):
        """
        Args:
            embed_fn: Function that takes List[str] and returns List[List[float]]
            cache: Optional QueryCache for caching embeddings
            batch_size: Max texts per batch
            model: Model identifier for cache keys
        """
        self.embed_fn = embed_fn
        self.cache = cache
        self.batch_size = batch_size
        self.model = model
        
        # Pending texts for batch embedding
        self._pending: List[Tuple[str, int]] = []  # (text, original_index)
        self._lock = threading.Lock()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts efficiently with caching.
        
        Checks cache first, then batches remaining texts.
        """
        if not texts:
            return []
        
        results: List[Optional[List[float]]] = [None] * len(texts)
        to_embed: List[Tuple[str, int]] = []  # (text, original_index)
        
        # Check cache
        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get_embedding(text, self.model)
                if cached is not None:
                    results[i] = cached
                    continue
            
            to_embed.append((text, i))
        
        # Batch embed remaining
        if to_embed:
            for batch_start in range(0, len(to_embed), self.batch_size):
                batch = to_embed[batch_start:batch_start + self.batch_size]
                batch_texts = [t for t, _ in batch]
                batch_indices = [i for _, i in batch]
                
                embeddings = self.embed_fn(batch_texts)
                
                for emb, text, idx in zip(embeddings, batch_texts, batch_indices):
                    results[idx] = emb
                    
                    # Cache
                    if self.cache:
                        self.cache.set_embedding(text, emb, self.model)
        
        return results  # type: ignore


# =============================================================================
# CONTEXT WINDOW OPTIMIZER
# =============================================================================

@dataclass
class ContextWindowConfig:
    """Configuration for context window optimization."""
    
    max_tokens: int = 4096  # Max tokens for context
    reserved_tokens: int = 1024  # Reserved for prompt template + response
    chars_per_token: float = 4.0  # Approximate chars per token
    min_chunks: int = 2  # Minimum chunks to include
    max_chunks: int = 10  # Maximum chunks to include


class ContextWindowOptimizer:
    """
    Dynamically adjusts chunk count based on LLM context limits.
    
    Ensures we don't exceed context window while maximizing relevant context.
    """
    
    def __init__(self, config: Optional[ContextWindowConfig] = None):
        self.config = config or ContextWindowConfig()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) / self.config.chars_per_token)
    
    def optimize_chunks(
        self,
        chunks: List[Dict],
        query: str,
        prompt_template_tokens: int = 200,
    ) -> List[Dict]:
        """
        Select optimal chunks that fit within context window.
        
        Args:
            chunks: List of chunk dicts with 'content' key
            query: Original query (contributes to token count)
            prompt_template_tokens: Estimated tokens for prompt template
        
        Returns:
            Optimized list of chunks
        """
        if not chunks:
            return []
        
        # Calculate available token budget
        available = (
            self.config.max_tokens
            - self.config.reserved_tokens
            - prompt_template_tokens
            - self.estimate_tokens(query)
        )
        
        if available <= 0:
            logger.warning("No token budget available for chunks")
            return chunks[:self.config.min_chunks]
        
        # Greedily add chunks up to budget
        selected = []
        used_tokens = 0
        
        for chunk in chunks:
            content = chunk.get('content', chunk.get('text', ''))
            chunk_tokens = self.estimate_tokens(content)
            
            if used_tokens + chunk_tokens <= available:
                selected.append(chunk)
                used_tokens += chunk_tokens
                
                if len(selected) >= self.config.max_chunks:
                    break
            elif len(selected) < self.config.min_chunks:
                # Include minimum chunks even if over budget (truncate later)
                selected.append(chunk)
                used_tokens += chunk_tokens
        
        logger.debug(
            f"Context optimization: {len(chunks)} chunks -> {len(selected)} chunks, "
            f"~{used_tokens} tokens (budget: {available})"
        )
        
        return selected
    
    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        preserve_end: bool = False,
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            preserve_end: If True, keep end of text (useful for recent context)
        """
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        max_chars = int(max_tokens * self.config.chars_per_token)
        
        if preserve_end:
            return "... " + text[-max_chars:]
        else:
            return text[:max_chars] + " ..."


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CacheConfig",
    "QueryCache",
    "LRUCache",
    "BatchEmbedder",
    "ContextWindowConfig",
    "ContextWindowOptimizer",
    "make_embedding_key",
    "make_result_key",
    "make_llm_key",
]
