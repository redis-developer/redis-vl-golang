# RedisVL Go vs Python benchmarks

Two mirror-image benchmark programs — `gobench/main.go` (Go) and
`pybench/bench.py` (Python) — run the same workloads against the same Redis
so the client libraries can be compared head to head. The vector search itself executes
inside Redis and is identical for both; what these benchmarks measure is
the *client-side* cost each library adds: query building, vector
serialization, reply parsing, validation, and concurrency behavior.

## Workloads

Both programs use an identical schema (text + tag + numeric + 384-dim
float32 HNSW/cosine vector, hash storage) and dataset shape (seeded random
unit vectors; generation is excluded from all timings). Half the queries
carry a tag filter, half are pure KNN, k=10.

1. **load** — bulk-load 10,000 documents in pipelined batches of 500;
   reports docs/sec. Both wait for `percent_indexed == 1` before querying.
2. **sequential** — 500 single-threaded queries; reports mean/p50/p95/p99
   latency. Dominated by Redis + one round trip; differences here are the
   per-call client overhead.
3. **concurrent** — 3,200 queries across 32 workers (goroutines vs
   threads); reports aggregate QPS and latency percentiles. This is where
   the GIL vs goroutines difference shows.

## Running

Requires a running Redis 8 / Redis Stack, e.g.:

```bash
docker run -d -p 6379:6379 redis:8.8.0
```

Each program uses its own index name (`bench-go` / `bench-py`) and cleans
up after itself.

```bash
# Go (from this repository's root)
make bench-go

# Python
make bench-py-deps   # one-time: creates benchmarks/.venv and installs redisvl
make bench-py
```

`bench-py-deps` installs redisvl into a local virtualenv
(`benchmarks/pybench/.venv`, gitignored), which works on PEP 668
externally-managed Pythons (Homebrew, Debian). `bench-py` uses that venv
when present, otherwise whatever `python3`/`python` is active — so an
already-activated environment with redisvl installed works too.

Flags/env: `-docs/--docs`, `-dims/--dims`, `-queries/--queries`,
`-concurrency/--concurrency`, `-conc-queries/--conc-queries`, `-k/--k`,
and `REDIS_URL`.

Each program prints a human-readable summary plus one JSON line with all
metrics, so runs can be collected and diffed programmatically.

## Fair-comparison notes

- Run both against the same Redis instance, same machine, ideally several
  times, discarding the first run (connection pool warmup, OS caches).
- Redis itself is often the bottleneck in the concurrent phase; if both
  implementations converge on the same QPS, Redis is saturated — lower
  `--concurrency` or use a larger `--docs` to shift work to the clients,
  or run Redis on a separate machine to include network serialization.
- Python concurrency uses threads (redis-py releases the GIL on socket
  I/O, but reply parsing and query building still serialize on the GIL).
  This is representative of a typical sync deployment; an asyncio variant
  would change the shape but not the GIL constraint.
- Vectors are distribution-identical (seeded uniform, unit-normalized) but
  not bit-identical across languages; index size, HNSW parameters, and
  result counts are the same, which is what matters for performance.
