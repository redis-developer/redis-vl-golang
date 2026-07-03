<div align="center">
    <img width="300" src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" alt="Redis">
    <h1>Redis Vector Library</h1>
    <p><strong>The AI-native Redis Go client</strong></p>
</div>

<div align="center">

[![Go Reference](https://pkg.go.dev/badge/github.com/redis-developer/redis-vl-golang.svg)](https://pkg.go.dev/github.com/redis-developer/redis-vl-golang)
[![CI](https://github.com/redis-developer/redis-vl-golang/actions/workflows/ci.yml/badge.svg)](https://github.com/redis-developer/redis-vl-golang/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/badge/go-%3E%3D1.25-00ADD8?logo=go)

</div>

---

## Introduction

Redis Vector Library (RedisVL) for Golang is the Go client for AI applications built on Redis, ported from the production-ready [Python library](https://github.com/redis/redis-vl-python). **Lightning-fast vector search meets enterprise-grade reliability.**

Perfect for building **RAG pipelines** with real-time retrieval, **AI agents** with memory and semantic routing, and **recommendation systems** with fast search and reranking.

<div align="center">

| **🎯 Core Capabilities** | **🚀 AI Extensions** | **🛠️ Dev Utilities** |
|:---:|:---:|:---:|
| **[Index Management](#index-management)**<br/>*Schema design, data loading, CRUD ops* | **[Semantic Caching](#semantic-caching)**<br/>*Reduce LLM costs & boost throughput* | **[CLI](#command-line-interface)**<br/>*Index management from terminal* |
| **[Vector Search](#retrieval)**<br/>*Similarity search with metadata filters* | **[LLM Memory](#llm-memory)**<br/>*Agentic AI context management* | **Context-Native API**<br/>*Every operation takes a `context.Context`* |
| **[Complex Filtering](#retrieval)**<br/>*Combine multiple filter types* | **[Semantic Routing](#semantic-routing)**<br/>*Intelligent query classification* | **[Vectorizers](#vectorizers)**<br/>*6+ embedding provider integrations* |
| **[Hybrid Search](#retrieval)**<br/>*Combine semantic & full-text signals* | **[Embedding Caching](#embedding-caching)**<br/>*Cache embeddings for efficiency* | **[Rerankers](#rerankers)**<br/>*Improve search result relevancy* |
|  |  | **[MCP Server](#mcp-server)**<br/>*Expose an existing Redis index to MCP clients* |

</div>

# 💪 Getting Started

## Installation

Install `redis-vl-golang` into your Go (>=1.23) module using `go get`:

```bash
go get github.com/redis-developer/redis-vl-golang
```

## Redis

Choose from multiple Redis deployment options:

<details>
<summary><b>Redis Cloud</b> - Managed cloud database (free tier available)</summary>

[Redis Cloud](https://redis.io/try-free) offers a fully managed Redis service with a free tier, perfect for getting started quickly.

</details>

<details>
<summary><b>Docker</b> - Local development</summary>

Run Redis locally using Docker:

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

This runs Redis 8+ with built-in vector search capabilities.

</details>

<details>
<summary><b>Redis Enterprise</b> - Commercial, self-hosted database</summary>

[Redis Enterprise](https://redis.io/enterprise/) provides enterprise-grade features for production deployments.

</details>

<details>
<summary><b>Azure Managed Redis</b> - Fully managed Redis Enterprise on Azure</summary>

[Azure Managed Redis](https://azure.microsoft.com/en-us/products/managed-redis) provides fully managed Redis Enterprise on Microsoft Azure.

</details>

> 💡 **Tip**: Enhance your experience and observability with the free [Redis Insight GUI](https://redis.io/insight/).

# Overview

## Index Management

1. **Design a schema** for your use case that models your dataset with built-in Redis indexable fields (*e.g. text, tags, numerics, geo, and vectors*).

    <details>
    <summary><b>Load schema from YAML file</b></summary>

    ```yaml
    index:
      name: user-idx
      prefix: user
      storage_type: json

    fields:
      - name: user
        type: tag
      - name: credit_score
        type: tag
      - name: job_title
        type: text
        attrs:
          sortable: true
      - name: embedding
        type: vector
        attrs:
          algorithm: flat
          dims: 4
          distance_metric: cosine
          datatype: float32
    ```

    ```go
    import "github.com/redis-developer/redis-vl-golang/schema"

    s, err := schema.FromYAMLFile("schemas/schema.yaml")
    ```

    </details>

    <details>
    <summary><b>Build schema programmatically</b></summary>

    ```go
    import "github.com/redis-developer/redis-vl-golang/schema"

    embedding, err := schema.NewVectorField("embedding", schema.VectorAttrs{
        Algorithm:      schema.Flat, // or schema.HNSW, schema.SVSVamana
        Dims:           4,
        DistanceMetric: schema.Cosine,
        Datatype:       "float32",
    })

    s, err := schema.NewIndexSchema(
        schema.IndexInfo{
            Name:        "user-idx",
            Prefixes:    []string{"user"},
            StorageType: schema.JSON,
        },
        schema.NewTagField("user"),
        schema.NewTagField("credit_score"),
        schema.NewTextField("job_title", schema.TextAttrs{
            BaseAttrs: schema.BaseAttrs{Sortable: true},
        }),
        embedding,
    )
    ```

    </details>

2. **Create a SearchIndex** with an input schema to perform admin and search operations on your index in Redis:

    ```go
    import redisvl "github.com/redis-developer/redis-vl-golang"

    // Define the index
    index, err := redisvl.NewSearchIndexFromURL(s, "redis://localhost:6379")
    defer index.Close()

    // Create the index in Redis
    err = index.Create(ctx)
    ```

    > Every method takes a `context.Context`, so a single index type covers both the sync and async use cases of the Python library. You can also bind to an existing [go-redis](https://github.com/redis/go-redis) client with `redisvl.NewSearchIndex(s, client)`.

    > 💡 For SVS-VAMANA indexes, `schema.RecommendCompression(dims, schema.Balanced)` suggests compression settings tuned for your vector dimensionality, and `schema.EstimateMemorySavings` quantifies the tradeoff.

3. **Load and fetch** data to/from your Redis instance:

    ```go
    import "github.com/redis-developer/redis-vl-golang/vectors"

    emb, _ := vectors.ToBuffer([]float64{0.23, 0.49, -0.18, 0.95}, vectors.Float32)
    data := map[string]any{"user": "john", "credit_score": "high", "embedding": emb}

    // load documents, specify the "user" field as the id
    keys, err := index.Load(ctx, []map[string]any{data}, redisvl.LoadOptions{IDField: "user"})

    // fetch by id
    john, err := index.Fetch(ctx, "john")
    ```

## Retrieval

Define queries and perform advanced searches over your indices, including vector search, complex filtering, and hybrid search combining semantic and full-text signals.

<details>
<summary><b>Quick Reference: Query Types</b></summary>

| Query Type | Use Case | Description |
|:---|:---|:---|
| `VectorQuery` | Semantic similarity search | Find similar vectors with optional filters |
| `VectorRangeQuery` | Distance-based search | Vector search within a defined distance range |
| `FilterQuery` | Metadata filtering | Filter and search using metadata fields |
| `TextQuery` | Full-text search | BM25-based keyword search with field weighting |
| `HybridQuery` | Combined search | Combine semantic + full-text signals (Redis 8.4.0+) |
| `AggregateHybridQuery` | Combined search | Hybrid scoring via aggregation (earlier Redis versions) |
| `MultiVectorQuery` | Multi-field vector search | Weighted scoring across multiple vector fields |
| `CountQuery` | Counting records | Count documents matching filter criteria |

</details>

### Vector Search

- `VectorQuery` - Flexible vector queries with customizable filters enabling semantic search:

    ```go
    import "github.com/redis-developer/redis-vl-golang/query"

    q := query.NewVectorQuery("embedding", []float64{0.16, -0.34, 0.98, 0.23}).
        NumResults(3).
        EfRuntime(100) // HNSW: higher for better recall

    // run the vector search query against the embedding field
    results, err := index.Query(ctx, q)
    ```

- `VectorRangeQuery` - Vector search within a defined range paired with customizable filters

### Complex Filtering

Build complex filtering queries by combining multiple filter types (tags, numerics, text, geo, timestamps) using logical operators:

```go
import (
    "github.com/redis-developer/redis-vl-golang/filter"
    "github.com/redis-developer/redis-vl-golang/query"
)

// Combine multiple filter types
tagFilter := filter.Tag("user").Eq("john")
priceFilter := filter.Num("price").Ge(100)

// Create complex filtering query with combined filters
q := query.NewVectorQuery("embedding", []float64{0.16, -0.34, 0.98, 0.23}).
    Filter(tagFilter.And(priceFilter)).
    NumResults(10)

results, err := index.Query(ctx, q)
```

- `FilterQuery` - Standard search using filters and full-text search
- `CountQuery` - Count the number of indexed records given attributes
- `TextQuery` - Full-text search with support for field weighting and BM25 scoring

> The fluent builder produces query strings identical to the Python library's operator-overloading DSL (`Tag("user") == "john"` ⇢ `filter.Tag("user").Eq("john")`).

### Hybrid Search

Combine semantic (vector) search with full-text (BM25) search signals for improved search quality:

- `HybridQuery` - Native hybrid search combining text and vector similarity (Redis 8.4.0+):

    ```go
    hq := query.NewHybridQuery("running shoes", "description",
        []float64{0.1, 0.2, 0.3}, "embedding").
        CombineLinear(0.3). // or CombineRRF(window, constant)
        NumResults(10)

    results, err := index.Hybrid(ctx, hq)
    ```

- `AggregateHybridQuery` - Hybrid search using aggregation (compatible with earlier Redis versions)
- `MultiVectorQuery` - Search over multiple vector fields simultaneously with weighted score combination

> `AggregateHybridQuery` requires FT.AGGREGATE `SCORER`/`ADDSCORES` support (Redis 8.x); `HybridQuery` requires FT.HYBRID (Redis 8.4+). Learn more about [hybrid search](https://docs.redisvl.com/en/stable/user_guide/11_advanced_queries.html#hybrid-queries-combining-text-and-vector-search).

## Dev Utilities

### Vectorizers

Integrate with popular embedding providers to greatly simplify the process of vectorizing unstructured data for your index and queries.

<details>
<summary><b>Supported Vectorizer Providers</b></summary>

- AzureOpenAI (`vectorize.NewAzureOpenAIVectorizer`)
- Cohere (`vectorize.NewCohereVectorizer`)
- Custom (`vectorize.Func` — wrap any embedding function)
- HuggingFace (`hf.New` — local in-process models via ONNX Runtime, separate module)
- Mistral (`vectorize.NewMistralVectorizer`)
- Ollama (`vectorize.NewOllamaVectorizer` — local models)
- OpenAI (`vectorize.NewOpenAIVectorizer`)
- VoyageAI (`vectorize.NewVoyageAIVectorizer`)

</details>

```go
import "github.com/redis-developer/redis-vl-golang/extensions/vectorize"

// set COHERE_API_KEY in your environment
co, err := vectorize.NewCohereVectorizer(ctx, vectorize.CohereConfig{})

// query-side embedding (input_type "search_query")
embedding, err := co.ForQueries().Embed(ctx, "What is the capital city of France?")

// document-side embeddings (input_type "search_document")
embeddings, err := co.EmbedMany(ctx, []string{
    "my document chunk content",
    "my other document chunk content",
})
```

> Embedding dimensions are auto-detected at construction, batching and retry with backoff are built in, and API keys default to the standard environment variables (`OPENAI_API_KEY`, `COHERE_API_KEY`, ...).

#### Local embeddings (HuggingFace + ONNX Runtime)

Run sentence-transformer models **in-process** — no API key, no per-call cost, data never leaves your machine. Like Python's `HFTextVectorizer`, the model is downloaded from the Hugging Face Hub on first use and cached locally:

```go
import "github.com/redis-developer/redis-vl-golang/extensions/vectorize/hf"

vec, err := hf.New(ctx, hf.Config{
    Model: "sentence-transformers/all-MiniLM-L6-v2",
})
defer vec.Close()

embedding, err := vec.Embed(ctx, "Hello, world!")
```

The `hf` package is a **separate Go module** (`go get github.com/redis-developer/redis-vl-golang/extensions/vectorize/hf`) because it binds to ONNX Runtime via cgo; the core library stays pure Go. It requires the [onnxruntime shared library](https://onnxruntime.ai/docs/install/) (point `ONNXRUNTIME_LIB_PATH` or `Config.ONNXRuntimePath` at it) and works with any BERT-family sentence-transformers model that ships an ONNX export — including `redis/langcache-embed-v1`. Pooling and normalization follow each model's sentence-transformers configuration, so embeddings match Python's output.

### Rerankers

Integrate with popular reranking providers to improve the relevancy of the initial search results from Redis:

```go
import "github.com/redis-developer/redis-vl-golang/extensions/rerank"

// Cohere (rerank-english-v3.0 default) or VoyageAI
r, err := rerank.NewCohereReranker(rerank.CohereConfig{Limit: 3})

results, err := r.Rank(ctx, "query text", []string{"doc one", "doc two", "doc three"})
```

Or rerank **locally** with a Hugging Face cross-encoder (no API key; same `hf` module and requirements as local embeddings):

```go
import "github.com/redis-developer/redis-vl-golang/extensions/vectorize/hf"

// cross-encoder/ms-marco-MiniLM-L-6-v2 by default
ce, err := hf.NewCrossEncoder(ctx, hf.CrossEncoderConfig{Limit: 3})
defer ce.Close()

results, err := ce.Rank(ctx, "query text", []string{"doc one", "doc two", "doc three"})
```

## Extensions

**RedisVL Extensions** provide production-ready modules implementing best practices and design patterns for working with LLM memory and agents.

> All semantic extensions take an explicit `vectorize.Vectorizer` — pick a hosted provider above or wrap your own embedding function with `vectorize.Func`.

### Semantic Caching

Increase application throughput and reduce the cost of using LLM models in production by leveraging previously generated knowledge with the `SemanticCache`.

<details>
<summary><b>Example: Semantic Cache Usage</b></summary>

```go
import "github.com/redis-developer/redis-vl-golang/extensions/cache"

// init cache with TTL and semantic distance threshold
llmcache, err := cache.NewSemanticCache(ctx, client, vectorizer, cache.SemanticCacheOptions{
    Name:              "llmcache",
    TTL:               360 * time.Second,
    DistanceThreshold: 0.1, // Redis COSINE distance [0-2], lower is stricter
})

// store user queries and LLM responses in the semantic cache
key, err := llmcache.Store(ctx, "What is the capital city of France?", "Paris")

// quickly check the cache with a slightly different prompt (before invoking an LLM)
hits, err := llmcache.Check(ctx, "What is France's capital city?")
fmt.Println(hits[0].Response)
```

```stdout
>>> Paris
```

</details>

> ☁️ Using the managed [LangCache](https://redis.io/docs/latest/develop/ai/langcache/) service instead? `cache.NewLangCache` provides the same Check/Store surface with server-side embeddings — no vectorizer required.

### Embedding Caching

Reduce computational costs and improve performance by caching embedding vectors with their associated text and metadata using the `EmbeddingsCache`.

<details>
<summary><b>Example: Embedding Cache Usage</b></summary>

```go
import "github.com/redis-developer/redis-vl-golang/extensions/cache"

// Initialize embedding cache
embedCache := cache.NewEmbeddingsCache(client, cache.EmbeddingsCacheOptions{
    Name: "embed_cache",
    TTL:  time.Hour,
})

// Wrap any vectorizer with the cache: repeated texts embed only once
cached := cache.NewCachedVectorizer(vectorizer, embedCache)

emb, err := cached.Embed(ctx, "What is machine learning?") // computes + caches
emb, err = cached.Embed(ctx, "What is machine learning?")  // served from Redis
```

</details>

### LLM Memory

Improve personalization and accuracy of LLM responses by providing user conversation context. Manage access to memory data using recency or relevancy, *powered by vector search* with the `MessageHistory` and `SemanticMessageHistory` types.

<details>
<summary><b>Example: Message History Usage</b></summary>

```go
import "github.com/redis-developer/redis-vl-golang/extensions/history"

h, err := history.NewSemanticMessageHistory(ctx, client, "my-session", vectorizer,
    history.SemanticMessageHistoryOptions{DistanceThreshold: 0.7})

// Supports roles: system, user, assistant, llm, tool
// Optional metadata field for additional context
err = h.AddMessages(ctx, []history.Message{
    {Role: "user", Content: "hello, how are you?"},
    {Role: "llm", Content: "I'm doing fine, thanks."},
    {Role: "user", Content: "what is the weather going to be today?"},
    {Role: "llm", Content: "I don't know", Metadata: map[string]any{"model": "gpt-4"}},
})

// Get recent chat history
recent, err := h.GetRecent(ctx, history.GetRecentOptions{TopK: 1})
// >>> [{Role: "llm", Content: "I don't know", ...}]

// Get relevant chat history (powered by vector search)
relevant, err := h.GetRelevant(ctx, "weather", history.GetRelevantOptions{TopK: 1})
// >>> [{Role: "user", Content: "what is the weather going to be today?"}]

// Filter messages by role
userMsgs, err := h.GetRecent(ctx, history.GetRecentOptions{Roles: []string{"user"}})
```

</details>

### Semantic Routing

Build fast decision models that run directly in Redis and route user queries to the nearest "route" or "topic".

<details>
<summary><b>Example: Semantic Router Usage</b></summary>

```go
import "github.com/redis-developer/redis-vl-golang/extensions/router"

routes := []router.Route{
    {
        Name:              "greeting",
        References:        []string{"hello", "hi"},
        Metadata:          map[string]any{"type": "greeting"},
        DistanceThreshold: 0.3,
    },
    {
        Name:              "farewell",
        References:        []string{"bye", "goodbye"},
        Metadata:          map[string]any{"type": "farewell"},
        DistanceThreshold: 0.3,
    },
}

// build semantic router from routes
sr, err := router.NewSemanticRouter(ctx, client, "topic-router", routes, vectorizer)

match, err := sr.Route(ctx, "Hi, good morning")
// >>> RouteMatch{Name: "greeting", Distance: 0.273891836405}
```

</details>

## Command Line Interface

Create, destroy, and manage Redis index configurations from a purpose-built CLI interface: `rvl`.

```bash
$ go build -o bin/rvl ./cmd/rvl   # or: make build-rvl
$ rvl --help

rvl - RedisVL command line tool

Commands:
  index create  -s schema.yaml [--overwrite]   Create a new index from a schema file
  index info    -i <name> [--json]             Show details about an index
  index listall [--json]                       List all indexes
  index delete  -i <name>                      Delete an index, keep its data
  index destroy -i <name>                      Delete an index and drop its data
  stats         -i <name> | -s schema.yaml     Display index statistics
  mcp           --config mcp.yaml              Run the RedisVL MCP server
  version       [-s]                           Print the library version
```

The Redis connection URL defaults to the `REDIS_URL` environment variable, then `redis://localhost:6379`; override per command with `-u`.

### MCP Server

RedisVL includes an MCP server that lets MCP-compatible clients search or upsert data in an existing Redis index through a small, stable tool contract.

The server:

- connects to one existing Redis Search index
- uses the configured vectorizer for query embedding and optional upsert embedding
- exposes `search-records` and, unless read-only mode is enabled, `upsert-records`
- supports stdio (default) and Streamable HTTP transports

Run it over stdio (default):

```bash
rvl mcp --config examples/mcp-config.yaml
```

Run it over Streamable HTTP for remote clients:

```bash
rvl mcp --config examples/mcp-config.yaml --transport streamable-http --host 127.0.0.1 --port 8000
```

Use `--read-only` when clients should only search:

```bash
rvl mcp --config examples/mcp-config.yaml --read-only
```

See [examples/mcp-config.yaml](examples/mcp-config.yaml) for the configuration format. Built on the official [MCP Go SDK](https://github.com/modelcontextprotocol/go-sdk). Unlike the Python server, this port has no built-in JWT auth — HTTP binds to non-loopback hosts are refused unless `--allow-unauthenticated` is passed — and does not support SSE transport or schema overrides.

## 🚀 Why RedisVL?

Redis is a proven, high-performance database that excels at real-time workloads. With RedisVL for Golang, you get a Go client that makes Redis's vector search, caching, and session management capabilities easily accessible for AI applications.

Built on the [go-redis](https://github.com/redis/go-redis) client, RedisVL provides an intuitive interface for vector search, LLM caching, and conversational AI memory - all the core components needed for modern AI workloads.

## Differences from the Python library

This is a feature-parity port of the Python library's core, extensions, providers, aggregation queries, and CLI. Notable differences:

- **Filter DSL**: Go has no operator overloading, so `Tag("user") == "john" & Num("price") >= 100` becomes `filter.Tag("user").Eq("john").And(filter.Num("price").Ge(100))`. Rendered query strings are identical.
- **One API instead of sync + async**: every method takes a `context.Context`; there is no separate `AsyncSearchIndex`.
- **No default local embedding model**: Python's semantic extensions default to a HuggingFace vectorizer; in Go all semantic extensions require an explicit `vectorize.Vectorizer`. For local models use the [`hf` module](#local-embeddings-huggingface--onnx-runtime) (in-process, ONNX Runtime) or Ollama.
- **Not included**: the VertexAI and Bedrock providers (require cloud SDKs; use `vectorize.Func` to wrap them). `SQLQuery` is not part of the port scope: in Python it is a thin adapter over the separate [sql-redis](https://pypi.org/project/sql-redis/) package, which has no Go equivalent yet — an adapter can be added if one appears. The MCP server is included but omits JWT auth, SSE transport, and schema overrides.
- **bfloat16/float16** vector encodings are implemented natively (no `ml_dtypes` needed).
- Operations return errors — nothing panics. Missing indexes can be detected with `errors.Is(err, redisvl.ErrIndexNotFound)`.

## 😁 Helpful Links

For additional help, check out the following resources:

- [Getting Started Guide](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html) *(Python docs; concepts apply directly)*
- [API Reference](https://docs.redisvl.com/en/stable/api/index.html)
- [Redis AI Recipes](https://github.com/redis-developer/redis-ai-resources)

## 🫱🏼‍🫲🏽 Contributing

Please help us by contributing PRs, opening GitHub issues for bugs or new feature ideas, improving documentation, or increasing test coverage.

```bash
make fmt && go vet ./... && go test ./...
```

Integration tests use [testcontainers-go](https://golang.testcontainers.org/) — with Docker running, `go test ./...` starts a `redis:8.8.0` container automatically (mirroring the Python library's testcontainers setup). Override the image with `REDIS_IMAGE`, or set `REDIS_URL` to test against an external server instead; without Docker or `REDIS_URL`, integration tests skip.

## 🚧 Maintenance

This project is supported by [Redis, Inc](https://redis.io) on a good faith effort basis. To report bugs, request features, or receive assistance, please file an issue.
