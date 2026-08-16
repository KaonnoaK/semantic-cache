"""
Full RAG pipeline with semantic cache layer.

FLOW:
Query → Cache check (5ms) → HIT: return instantly
                          → MISS: hybrid retrieval + LLM (98s) → store in cache
"""
import time
from sentence_transformers import SentenceTransformer
from src.semantic_cache import SemanticCache
from src.hybrid_retriever import load_chunks, build_bm25, hybrid_retrieve
from src.llm import generate


class CachedRAG:
    def __init__(self, threshold=0.92, llm_model='models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'):
        self.cache = SemanticCache(threshold=threshold)
        self.llm_model = llm_model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index, self.chunks = load_chunks('minilm_flat_300')
        self.bm25 = build_bm25(self.chunks)

        # Track all query latencies for comparison
        self.query_log = []

    def query(self, question):
        """
        Answer a question using cache-first strategy.
        Logs full timing breakdown for analysis.
        """
        print(f"\n{'='*55}")
        print(f"Query: {question}")
        t_start = time.perf_counter()

        # Step 1 — check cache
        cached_answer, similarity, cache_time_ms = self.cache.get(question)

        if cached_answer is not None:
            total_time = time.perf_counter() - t_start
            print(f"  Answer: {cached_answer[:200]}...")
            print(f"  Total time: {total_time*1000:.1f}ms (CACHE HIT)")

            self.query_log.append({
                'query': question,
                'source': 'cache',
                'similarity': similarity,
                'cache_time_ms': cache_time_ms,
                'llm_time_s': 0,
                'total_time_s': total_time,
                'answer': cached_answer
            })
            return cached_answer

        # Step 2 — cache miss: run full RAG pipeline
        print(f"  Running full RAG pipeline...")
        retrieved, embed_time, retrieve_time = hybrid_retrieve(
            question, self.index, self.chunks,
            self.bm25, self.embed_model, k=3, alpha=0.7
        )
        answer, llm_time = generate(question, retrieved, model_path=self.llm_model)
        total_time = time.perf_counter() - t_start

        print(f"  Answer: {answer[:200]}...")
        print(f"  Total time: {total_time:.1f}s (CACHE MISS — full pipeline)")

        # Step 3 — store in cache for future
        self.cache.put(question, answer)

        self.query_log.append({
            'query': question,
            'source': 'rag',
            'similarity': similarity,
            'cache_time_ms': cache_time_ms,
            'llm_time_s': llm_time,
            'total_time_s': total_time,
            'answer': answer
        })
        return answer

    def summary(self):
        self.cache.summary()

        if self.query_log:
            hits = [q for q in self.query_log if q['source'] == 'cache']
            misses = [q for q in self.query_log if q['source'] == 'rag']
            print(f"\n  Query breakdown:")
            print(f"    Cache hits:  {len(hits)} queries | avg {sum(q['total_time_s']*1000 for q in hits)/max(len(hits),1):.0f}ms each")
            print(f"    Cache misses:{len(misses)} queries | avg {sum(q['total_time_s'] for q in misses)/max(len(misses),1):.1f}s each")
