"""
Semantic Cache Experiment — Threshold Sensitivity Study.

RESEARCH QUESTION:
What similarity threshold maximizes cache utility while preserving answer accuracy?

Too low (0.70) → wrong answers served from cache (false positives)
Too high (0.99) → cache never hits (false negatives)
Sweet spot      → maximum time savings without quality loss

EXPERIMENT DESIGN:
- Round 1: seed cache with 10 original queries (all cache misses)
- Round 2: run 20 paraphrased/similar queries (should be cache hits)
- Vary threshold: 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.99
- Measure: hit rate, false positive rate, time saved per threshold
"""
import csv
import json
import time
import os
from src.semantic_cache import SemanticCache
from src.hybrid_retriever import load_chunks, build_bm25, hybrid_retrieve
from src.llm import generate
from sentence_transformers import SentenceTransformer

# Original queries — used to seed the cache
SEED_QUERIES = [
    "how to create LDAP account",
    "how to create bhairav account",
    "how to create sandesh account",
    "3rd floor switch is down what to do",
    "how to do bhairav ip blocking",
    "adding member to mailing list",
    "what to do after preliminary shortlist in phd admission",
    "how to setup the TA preference form",
    "steps for configuring DHCP snooping or stopping DHCP relay",
    "wifi not working troubleshooting"
]

# Paraphrased versions — SHOULD hit the cache
PARAPHRASED_QUERIES = [
    ("how to make a new LDAP account", "how to create LDAP account"),
    ("create new bhairav user", "how to create bhairav account"),
    ("add sandesh user account", "how to create sandesh account"),
    ("switch on third floor not responding", "3rd floor switch is down what to do"),
    ("block IP address on bhairav server", "how to do bhairav ip blocking"),
    ("add someone to email list", "adding member to mailing list"),
    ("phd admission after shortlist steps", "what to do after preliminary shortlist in phd admission"),
    ("setup TA form preference", "how to setup the TA preference form"),
    ("configure DHCP snooping on switch", "steps for configuring DHCP snooping or stopping DHCP relay"),
    ("internet not working wireless", "wifi not working troubleshooting"),
]

# Unrelated queries — should NOT hit the cache (true negatives)
UNRELATED_QUERIES = [
    "how to book a meeting room",
    "what is the lab printer IP",
    "how to mount NFS share",
    "GPU server usage policy",
    "VPN setup for home access",
]

def run_threshold_experiment(thresholds=[0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.99]):
    """
    For each threshold, measure:
    - Hit rate on paraphrased queries (want HIGH)
    - False positive rate on unrelated queries (want LOW)
    - Effective speedup (derived from hit rate)
    """
    print("Loading RAG components for seeding...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, chunks = load_chunks('minilm_flat_300')
    bm25 = build_bm25(chunks)

    # Step 1 — seed cache once with Q4 model answers
    print("\nSeeding cache with 10 base queries (this takes ~15 min)...")
    seed_cache = SemanticCache(threshold=0.50)  # low threshold just for seeding

    seed_answers = {}
    for query in SEED_QUERIES:
        retrieved, _, _ = hybrid_retrieve(
            query, index, chunks, bm25, embed_model, k=3, alpha=0.7
        )
        answer, llm_time = generate(
            query, retrieved,
            model_path='models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
        )
        seed_cache.put(query, answer)
        seed_answers[query] = answer
        print(f"  Seeded: '{query[:50]}' ({llm_time:.1f}s)")

    # Save seed answers
    os.makedirs('results', exist_ok=True)
    with open('results/seed_answers.json', 'w') as f:
        json.dump(seed_answers, f, indent=2)
    print(f"\nCache seeded with {len(seed_answers)} entries.")

    # Step 2 — test each threshold
    results = []

    for threshold in thresholds:
        print(f"\n{'='*55}")
        print(f"Testing threshold = {threshold}")

        # Create fresh cache with this threshold, pre-loaded with seed answers
        cache = SemanticCache(threshold=threshold)
        for query, answer in seed_answers.items():
            cache.put(query, answer)

        # Test paraphrased queries (should HIT)
        paraphrase_hits = 0
        for para_query, original in PARAPHRASED_QUERIES:
            answer, sim, _ = cache.get(para_query)
            hit = answer is not None
            paraphrase_hits += int(hit)
            print(f"  Para: '{para_query[:40]}' → sim={sim:.3f} → {'HIT' if hit else 'MISS'}")

        # Test unrelated queries (should MISS)
        false_positives = 0
        for unrelated in UNRELATED_QUERIES:
            answer, sim, _ = cache.get(unrelated)
            fp = answer is not None
            false_positives += int(fp)
            print(f"  Unrel: '{unrelated[:40]}' → sim={sim:.3f} → {'FP!' if fp else 'OK'}")

        hit_rate = paraphrase_hits / len(PARAPHRASED_QUERIES)
        fp_rate = false_positives / len(UNRELATED_QUERIES)

        # Effective avg latency with this threshold
        # Assumes 40% of real queries are paraphrases of previous ones
        effective_hit_rate = hit_rate * 0.4
        avg_latency = effective_hit_rate * 0.005 + (1 - effective_hit_rate) * 98.5

        record = {
            'threshold': threshold,
            'paraphrase_hit_rate': round(hit_rate, 3),
            'false_positive_rate': round(fp_rate, 3),
            'paraphrase_hits': paraphrase_hits,
            'false_positives': false_positives,
            'effective_speedup': round(98.5 / avg_latency, 2)
        }
        results.append(record)
        print(f"  Hit rate: {hit_rate:.1%} | FP rate: {fp_rate:.1%} | Speedup: {98.5/avg_latency:.1f}×")

    # Save results
    with open('results/cache_threshold_experiment.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*55}")
    print("THRESHOLD EXPERIMENT RESULTS:")
    print(f"{'Threshold':>10} {'Hit Rate':>10} {'FP Rate':>10} {'Speedup':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['threshold']:>10} {r['paraphrase_hit_rate']:>9.1%} "
              f"{r['false_positive_rate']:>9.1%} {r['effective_speedup']:>9.1f}×")

    print(f"\nSaved → results/cache_threshold_experiment.csv")
    return results

if __name__ == '__main__':
    run_threshold_experiment()
