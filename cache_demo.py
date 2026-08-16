"""
Real-world cache simulation at optimal threshold (0.80).
Shows actual cache behavior on a realistic query workload.
"""
import time
import json
from src.semantic_cache import SemanticCache

# Load pre-computed seed answers
with open('results/seed_answers.json') as f:
    seed_answers = json.load(f)

# Build cache at optimal threshold
cache = SemanticCache(threshold=0.80)
for query, answer in seed_answers.items():
    cache.put(query, answer)

# Simulate a realistic day of sysadmin queries
# Mix of: exact repeats, paraphrases, new queries
workload = [
    # Paraphrases — should HIT
    "how to make a new LDAP account",
    "create new bhairav user",
    "add sandesh user account",
    "block IP address on bhairav server",
    "internet not working wireless",
    "add someone to email list",
    "switch on third floor not responding",
    "setup TA form preference",
    # Exact repeats — should HIT at 1.0 similarity
    "how to create LDAP account",
    "wifi not working troubleshooting",
    # New queries — should MISS
    "how to book a meeting room",
    "GPU server usage policy",
    "how to mount NFS share",
]

print("Simulating realistic sysadmin query workload...")
print(f"Cache threshold: 0.80 | Entries: {cache.index.ntotal}\n")

for query in workload:
    t0 = time.perf_counter()
    answer, sim, lookup_ms = cache.get(query)
    total_ms = (time.perf_counter() - t0) * 1000

    status = "HIT" if answer else "MISS"
    print(f"[{status}] sim={sim:.3f} | {total_ms:.0f}ms | '{query[:50]}'")

cache.summary()
