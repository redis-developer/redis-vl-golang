# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Release tags cover both Go modules in this repository: the core module
(`vX.Y.Z`) and the HF vectorizer module (`extensions/vectorize/hf/vX.Y.Z`).

## [Unreleased]

### Added

- GoReleaser pipeline: `v*` release tags now attach prebuilt `rvl` binaries
  (macOS/Linux/Windows, amd64+arm64) to the GitHub Release
- API keys and tokens are trimmed of stray whitespace (copy-paste safety)

### Changed

- MCP example config now returns document metadata fields from
  `search-records` by default

## [0.1.0] - 2026-07-03

The first release of RedisVL for Golang, ported from
[redis-vl-python](https://github.com/redis/redis-vl-python).

### Added

- Schema-driven index management (`SearchIndex`) for Redis Hash and JSON
  storage, with YAML and programmatic schema definitions
- Query builders: vector KNN, vector range, filter, count, full-text,
  hybrid (`FT.HYBRID` and `FT.AGGREGATE`-based), and multi-vector queries
- Fluent filter DSL (tag, numeric, text, geo, timestamp) with rendered
  query strings byte-identical to the Python library
- Batch operations (`BatchSearch`, `BatchQuery`, `FetchMany`), pagination,
  load-time schema validation, and the SVS-VAMANA compression advisor
- AI extensions: `SemanticCache`, `EmbeddingsCache`, `CachedVectorizer`,
  `MessageHistory` / `SemanticMessageHistory`, and `SemanticRouter`
- Managed [LangCache](https://redis.io/docs/latest/develop/ai/langcache/)
  client with the same Check/Store surface as the semantic cache
- Vectorizers: OpenAI, Azure OpenAI, Cohere, Mistral, VoyageAI, Ollama,
  and a custom-function adapter
- Local in-process embeddings and cross-encoder reranking via ONNX Runtime
  (separate `extensions/vectorize/hf` module), with output parity verified
  against Python sentence-transformers
- Rerankers: Cohere and VoyageAI
- The `rvl` command-line interface (index management, stats, MCP server)
- MCP server (stdio and streamable HTTP) exposing `search-records` and
  `upsert-records` tools
- Integration testing via testcontainers-go (pinned to `redis:8.8.0`)
- Go-vs-Python benchmark harness (`benchmarks/`)
- Antora documentation site published to GitHub Pages

[Unreleased]: https://github.com/redis-developer/redis-vl-golang/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/redis-developer/redis-vl-golang/releases/tag/v0.1.0
