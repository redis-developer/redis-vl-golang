// Package mcpserver exposes an existing Redis search index to MCP clients
// through the official MCP Go SDK, mirroring the Python `rvl mcp` server:
// it connects to one index, embeds queries with a configured vectorizer,
// and provides `search-records` and (unless read-only) `upsert-records`
// tools over stdio or Streamable HTTP.
//
// Compared to the Python server, this port supports the hosted vectorizer
// providers (no HuggingFace), does not implement JWT auth (HTTP binds to
// non-loopback hosts are refused unless explicitly allowed), and does not
// support schema overrides.
package mcpserver

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/redis-developer/redis-vl-golang/vectors"
)

// Config is the YAML configuration for the MCP server (subset of the
// Python MCPConfig with matching field names).
type Config struct {
	Server   ServerConfig  `yaml:"server"`
	Index    IndexConfig   `yaml:"index"`
	Runtime  RuntimeConfig `yaml:"runtime"`
	ReadOnly bool          `yaml:"read_only"`
}

// ServerConfig holds connection settings.
type ServerConfig struct {
	// RedisURL is the Redis connection URL (required; REDIS_URL env var is
	// used when empty).
	RedisURL string `yaml:"redis_url"`
}

// IndexConfig binds the server to one existing Redis search index.
type IndexConfig struct {
	// RedisName is the name of the existing index (required).
	RedisName string `yaml:"redis_name"`
	// Vectorizer configures query/upsert embedding (optional; without it
	// only text search is available).
	Vectorizer *VectorizerConfig `yaml:"vectorizer"`
}

// VectorizerConfig selects an embedding provider.
type VectorizerConfig struct {
	// Class is one of: openai, azure_openai, cohere, mistral, voyageai,
	// ollama.
	Class string `yaml:"class"`
	// Model is the embedding model name (provider default when empty).
	Model string `yaml:"model"`
	// Dims skips the dimension probe when set.
	Dims int `yaml:"dims"`
}

// RuntimeConfig tunes search and upsert behavior (field names match the
// Python MCPRuntimeConfig).
type RuntimeConfig struct {
	// TextFieldName is the text field used for full-text search fallback.
	TextFieldName string `yaml:"text_field_name"`
	// VectorFieldName is the vector field used for semantic search and
	// upsert embedding (required when a vectorizer is configured).
	VectorFieldName string `yaml:"vector_field_name"`
	// DefaultEmbedTextField is the record field embedded during upsert.
	DefaultEmbedTextField string `yaml:"default_embed_text_field"`
	// DefaultLimit is the search result count when the client omits limit
	// (default 10).
	DefaultLimit int `yaml:"default_limit"`
	// MaxLimit caps the client-provided limit (default 100).
	MaxLimit int `yaml:"max_limit"`
	// MaxUpsertRecords caps records per upsert call (default 64).
	MaxUpsertRecords int `yaml:"max_upsert_records"`
	// SkipEmbeddingIfPresent skips embedding records that already carry a
	// vector field value (default true).
	SkipEmbeddingIfPresent *bool `yaml:"skip_embedding_if_present"`
	// ReturnFields restricts the fields returned by search (default: all
	// fields except the vector field).
	ReturnFields []string `yaml:"return_fields"`
	// Dtype is the vector datatype of the index field (default float32).
	Dtype string `yaml:"dtype"`
}

// LoadConfig reads and validates a YAML config file.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("mcp config %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("invalid mcp config yaml: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Validate applies defaults and checks required fields.
func (c *Config) Validate() error {
	if c.Server.RedisURL == "" {
		c.Server.RedisURL = os.Getenv("REDIS_URL")
	}
	if c.Server.RedisURL == "" {
		c.Server.RedisURL = "redis://localhost:6379"
	}
	if c.Index.RedisName == "" {
		return fmt.Errorf("mcp config: index.redis_name is required")
	}
	if v := c.Index.Vectorizer; v != nil {
		switch strings.ToLower(v.Class) {
		case "openai", "azure_openai", "cohere", "mistral", "voyageai", "ollama":
		case "":
			return fmt.Errorf("mcp config: index.vectorizer.class is required when a vectorizer is configured")
		default:
			return fmt.Errorf("mcp config: unsupported vectorizer class %q (supported: openai, azure_openai, cohere, mistral, voyageai, ollama)", v.Class)
		}
		if c.Runtime.VectorFieldName == "" {
			return fmt.Errorf("mcp config: runtime.vector_field_name is required when a vectorizer is configured")
		}
	}
	if c.Index.Vectorizer == nil && c.Runtime.TextFieldName == "" {
		return fmt.Errorf("mcp config: configure index.vectorizer (semantic search) or runtime.text_field_name (text search)")
	}
	if c.Runtime.DefaultLimit <= 0 {
		c.Runtime.DefaultLimit = 10
	}
	if c.Runtime.MaxLimit <= 0 {
		c.Runtime.MaxLimit = 100
	}
	if c.Runtime.MaxUpsertRecords <= 0 {
		c.Runtime.MaxUpsertRecords = 64
	}
	if c.Runtime.SkipEmbeddingIfPresent == nil {
		t := true
		c.Runtime.SkipEmbeddingIfPresent = &t
	}
	if c.Runtime.Dtype == "" {
		c.Runtime.Dtype = string(vectors.Float32)
	}
	if _, err := vectors.Parse(c.Runtime.Dtype); err != nil {
		return fmt.Errorf("mcp config: %w", err)
	}
	return nil
}

// clampLimit resolves the effective search limit from a client-provided
// value.
func (r *RuntimeConfig) clampLimit(requested int) int {
	if requested <= 0 {
		return r.DefaultLimit
	}
	if requested > r.MaxLimit {
		return r.MaxLimit
	}
	return requested
}
