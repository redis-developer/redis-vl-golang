"""Benchmark RedisVL for Python against a running Redis.

Mirrors benchmarks/gobench/main.go exactly (same schema, dataset shape, and
workloads) so the two implementations can be compared:

  1. load:       bulk-load N documents (batches of 500)
  2. sequential: single-threaded KNN queries, latency percentiles
  3. concurrent: C threads issuing KNN queries, aggregate QPS

Standalone: requires only the redisvl package (pip install redisvl).
Use python3/pip3 on systems without a `python`/`pip` alias, e.g.:

  python3 -m venv .venv && .venv/bin/pip install redisvl
  .venv/bin/python benchmarks/pybench/bench.py --docs 10000

Or simply: make bench-py-deps && make bench-py (handles the fallback).
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=int, default=10000)
    p.add_argument("--dims", type=int, default=384)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--conc-queries", type=int, default=3200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--url", default=os.environ.get("REDIS_URL", "redis://localhost:6379"))
    return p.parse_args()


def random_unit_vector(rng, dims):
    v = np.array([rng.uniform(-1, 1) for _ in range(dims)], dtype=np.float32)
    return v / np.linalg.norm(v)


def build_query(vec, i, k):
    q = VectorQuery(
        vector=vec.tolist(),
        vector_field_name="embedding",
        return_fields=["content", "category", "price"],
        num_results=k,
    )
    if i % 2 == 0:
        q.set_filter(Tag("category") == f"cat{i % 10}")
    return q


def wait_indexed(index, want, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = index.info()
        num_docs = int(info.get("num_docs", 0))
        pct = float(info.get("percent_indexed", 1))
        if num_docs >= want and pct >= 1:
            return
        time.sleep(0.2)
    raise TimeoutError(f"index did not finish indexing {want} docs in time")


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    args = parse_args()

    schema = {
        "index": {"name": "bench-py", "prefix": "bench-py", "storage_type": "hash"},
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "price", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": args.dims,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
    index = SearchIndex.from_dict(schema, redis_url=args.url)
    index.create(overwrite=True, drop=True)

    # --- dataset (generation excluded from timings) ---
    rng = random.Random(42)
    records = [
        {
            "doc_id": str(i),
            "content": f"document {i} about topic {i % 10} with some benchmark filler text",
            "category": f"cat{i % 10}",
            "price": float(i % 1000),
            "embedding": random_unit_vector(rng, args.dims).tobytes(),
        }
        for i in range(args.docs)
    ]
    query_vectors = [random_unit_vector(rng, args.dims) for _ in range(args.queries)]

    try:
        # --- 1. load ---
        load_start = time.perf_counter()
        index.load(records, id_field="doc_id", batch_size=500)
        load_secs = time.perf_counter() - load_start

        wait_indexed(index, args.docs)

        # --- 2. sequential queries ---
        latencies = []
        for i in range(args.queries):
            start = time.perf_counter()
            index.query(build_query(query_vectors[i], i, args.k))
            latencies.append((time.perf_counter() - start) * 1000)

        # --- 3. concurrent queries ---
        conc_latencies = [0.0] * args.conc_queries

        def one(i):
            start = time.perf_counter()
            try:
                index.query(build_query(query_vectors[i % len(query_vectors)], i, args.k))
            except Exception as e:  # log and continue, like the Go runner
                print(f"query error: {e}", file=sys.stderr)
            conc_latencies[i] = (time.perf_counter() - start) * 1000

        conc_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(one, range(args.conc_queries)))
        conc_secs = time.perf_counter() - conc_start
    finally:
        index.delete(drop=True)
        index.disconnect()

    latencies.sort()
    conc_latencies.sort()
    mean = sum(latencies) / len(latencies)

    result = {
        "impl": "python",
        "docs": args.docs,
        "load_secs": round(load_secs, 2),
        "load_docs_per_sec": round(args.docs / load_secs, 2),
        "seq_queries": len(latencies),
        "seq_mean_ms": round(mean, 3),
        "seq_p50_ms": round(percentile(latencies, 0.50), 3),
        "seq_p95_ms": round(percentile(latencies, 0.95), 3),
        "seq_p99_ms": round(percentile(latencies, 0.99), 3),
        "conc_queries": args.conc_queries,
        "concurrency": args.concurrency,
        "conc_secs": round(conc_secs, 2),
        "conc_qps": round(args.conc_queries / conc_secs, 2),
        "conc_p50_ms": round(percentile(conc_latencies, 0.50), 3),
        "conc_p99_ms": round(percentile(conc_latencies, 0.99), 3),
    }

    print("== RedisVL for Python ==")
    print(f"load:       {args.docs} docs in {load_secs:.2f}s ({args.docs / load_secs:.0f} docs/s)")
    print(
        f"sequential: {len(latencies)} queries  mean {mean:.3f}ms  "
        f"p50 {percentile(latencies, 0.50):.3f}ms  p95 {percentile(latencies, 0.95):.3f}ms  "
        f"p99 {percentile(latencies, 0.99):.3f}ms"
    )
    print(
        f"concurrent: {args.conc_queries} queries x {args.concurrency} workers in {conc_secs:.2f}s "
        f"=> {args.conc_queries / conc_secs:.0f} qps  "
        f"p50 {percentile(conc_latencies, 0.50):.3f}ms  p99 {percentile(conc_latencies, 0.99):.3f}ms"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
