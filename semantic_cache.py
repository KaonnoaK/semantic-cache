"""
Semantic Cache for RAG pipeline.

CONCEPT:
Traditional cache = exact string match. "reset password" ≠ "change password"
Semantic cache = vector similarity match. Both map to nearby vectors → cache hit.

HOW IT WORKS:
1. Every answered query gets stored as (vector, answer) pair
2. New query → embed → compare against all cached vectors
3. If closest cached vector has cosine similarity > threshold → cache hit
4. Else → cache miss → run full RAG pipeline → store result

WHY COSINE SIMILARITY (not L2 distance):
Cosine measures the ANGLE between vectors, not their magnitude.
"reset password" and "change my password" point in similar directions
in embedding space even if their magnitudes differ.
cos(θ) = (A·B) / (|A| × |B|)
Range: -1 (opposite) to 1 (identical). We want > 0.90 for high confidence.

WHY THIS MATTERS FOR NVIDIA/GOOGLE:
Production LLM serving (vLLM, TensorRT-LLM) implements semantic caching
to reduce GPU compute costs. A cache hit = zero GPU usage = massive savings
at scale. Google processes billions of queries/day — even 10% cache hit rate
saves enormous compute.
"""

import numpy as np
import faiss
import time
import json
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime


class SemanticCache:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.92):
        """
        threshold: cosine similarity above which we consider a cache hit.
        0.92 = very similar meaning, very low false positive rate.
        We'll experiment with this value — it's the key hyperparameter.
        """
        self.threshold = threshold
        self.model_name = model_name
        self.dim = 384  # MiniLM embedding dimension

        print(f"Loading embedding model for cache: {model_name}")
        self.model = SentenceTransformer(model_name)

        # FAISS index for fast similarity search over cached queries
        # IndexFlatIP = Inner Product (dot product) — equivalent to cosine
        # similarity when vectors are normalized (which we do below)
        self.index = faiss.IndexFlatIP(self.dim)

        # Parallel list — index i in FAISS → entry i here
        self.entries = []  # list of {query, answer, timestamp, hits}

        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time_saved_s': 0,
            'avg_llm_time_s': 98.5  # from your Q4 experiment baseline
        }

        print(f"Semantic cache initialized (threshold={threshold})")

    def _embed_and_normalize(self, text):
        """
        Embed text and L2-normalize the vector.
        WHY NORMALIZE: makes dot product equivalent to cosine similarity.
        ||v|| = 1 for all cached vectors → dot product = cos(angle between them)
        """
        vec = self.model.encode([text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(vec)  # in-place normalization
        return vec

    def get(self, query):
        """
        Check cache for a semantically similar query.
        Returns (answer, similarity, latency_ms) if hit, else (None, 0, latency_ms).
        """
        t0 = time.perf_counter()
        self.stats['total_queries'] += 1

        if self.index.ntotal == 0:
            # Cache is empty
            self.stats['cache_misses'] += 1
            return None, 0.0, (time.perf_counter() - t0) * 1000

        query_vec = self._embed_and_normalize(query)

        # Search for most similar cached query
        similarities, indices = self.index.search(query_vec, k=1)
        similarity = float(similarities[0][0])
        idx = int(indices[0][0])

        cache_time_ms = (time.perf_counter() - t0) * 1000

        if similarity >= self.threshold:
            # CACHE HIT
            self.stats['cache_hits'] += 1
            self.stats['total_time_saved_s'] += self.stats['avg_llm_time_s']
            self.entries[idx]['hits'] += 1
            self.entries[idx]['last_hit'] = datetime.now().isoformat()

            print(f"  [CACHE HIT]  similarity={similarity:.4f} | "
                  f"matched: '{self.entries[idx]['query'][:50]}' | "
                  f"lookup={cache_time_ms:.1f}ms")
            return self.entries[idx]['answer'], similarity, cache_time_ms
        else:
            # CACHE MISS
            self.stats['cache_misses'] += 1
            print(f"  [CACHE MISS] similarity={similarity:.4f} < {self.threshold} | "
                  f"lookup={cache_time_ms:.1f}ms")
            return None, similarity, cache_time_ms

    def put(self, query, answer):
        """Store a new (query, answer) pair in the cache."""
        query_vec = self._embed_and_normalize(query)
        self.index.add(query_vec)
        self.entries.append({
            'query': query,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'hits': 0,
            'last_hit': None
        })
        print(f"  [CACHE STORE] '{query[:60]}' (cache size: {self.index.ntotal})")

    def hit_rate(self):
        if self.stats['total_queries'] == 0:
            return 0
        return self.stats['cache_hits'] / self.stats['total_queries']

    def summary(self):
        n = self.stats['total_queries']
        hits = self.stats['cache_hits']
        misses = self.stats['cache_misses']
        saved = self.stats['total_time_saved_s']
        rate = self.hit_rate()

        avg_with_cache = (
            rate * 0.005 +           # cache hit = ~5ms
            (1 - rate) * self.stats['avg_llm_time_s']  # miss = full pipeline
        )

        print(f"\n{'='*55}")
        print(f"SEMANTIC CACHE SUMMARY")
        print(f"  Total queries:    {n}")
        print(f"  Cache hits:       {hits} ({rate*100:.1f}%)")
        print(f"  Cache misses:     {misses}")
        print(f"  Time saved:       {saved:.1f}s")
        print(f"  Avg latency without cache: {self.stats['avg_llm_time_s']:.1f}s")
        print(f"  Avg latency with cache:    {avg_with_cache:.1f}s")
        print(f"  Effective speedup:         {self.stats['avg_llm_time_s']/avg_with_cache:.1f}×")
        print(f"{'='*55}")

    def save(self, path='data/cache'):
        """Persist cache to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f'{path}/cache.faiss')
        with open(f'{path}/entries.json', 'w') as f:
            json.dump(self.entries, f, indent=2)
        with open(f'{path}/stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Cache saved → {path}/ ({self.index.ntotal} entries)")

    def load(self, path='data/cache'):
        """Load cache from disk."""
        if not os.path.exists(f'{path}/cache.faiss'):
            print("No cache found on disk — starting fresh.")
            return
        self.index = faiss.read_index(f'{path}/cache.faiss')
        with open(f'{path}/entries.json') as f:
            self.entries = json.load(f)
        with open(f'{path}/stats.json') as f:
            self.stats = json.load(f)
        print(f"Cache loaded — {self.index.ntotal} entries")
