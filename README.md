
# Semantic Cache for LLM Inference Cost Reduction

> *A vector similarity-based caching layer that eliminates redundant LLM inference for semantically equivalent queries.*

---

## The Problem

LLM inference = 99.7% of RAG pipeline latency (98.5s/query locally). Traditional exact-match caching fails for natural language — "how to create LDAP account" and "how to make a new LDAP account" are treated as different queries despite identical intent.

---

## The Solution

Embed queries into vector space. Match on **meaning** (cosine similarity) rather than string equality. If a new query is semantically close enough to a cached one → return instantly. Skip the LLM entirely.

Query → embed → cosine similarity vs cache
├── similarity ≥ 0.80 → return cached answer (~12ms)
└── similarity < 0.80 → full LLM pipeline (98.5s) → store in cache

---

## Key Finding — Natural Similarity Gap

MiniLM-L6-v2 creates a clean separation:

| Query Type | Similarity Range |
|---|---|
| Exact repeats | 1.000 |
| Paraphrases | 0.731 – 0.931 |
| Unrelated queries | 0.122 – 0.332 |

Gap of 0.40 units between paraphrases and unrelated queries — allows confident threshold placement at 0.80 with zero false positive risk.

---

## Threshold Sensitivity Results

| Threshold | Hit Rate | False Positives | Speedup |
|---|---|---|---|
| 0.70 | 100% | 0% | 1.7× |
| **0.80** ★ | **90%** | **0%** | **1.6×** |
| 0.85 | 50% | 0% | 1.2× |
| 0.90+ | ≤20% | 0% | ~1.0× |

★ Optimal — maximum hit rate, zero false positives.

---

## Real-World Results (13 queries, threshold=0.80)

- Cache hit rate: **69.2%**
- False positives: **0**
- Avg latency without cache: 98.5s
- Avg latency with cache: **30.3s**
- **Effective speedup: 3.2×**

---

## Stack

Python · FAISS IndexFlatIP · sentence-transformers (MiniLM-L6-v2) · llama.cpp

---
