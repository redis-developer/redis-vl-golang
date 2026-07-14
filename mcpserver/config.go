// Package mcpserver exposes an existing Redis search index to MCP clients
// through the official MCP Go SDK, mirroring the Python `rvl mcp` server:
// it connects to one index, embeds queries with a configured vectorizer,
// and provides `search-records` and (unless read-only) `upsert-records`
// tools over stdio, SSE, or Streamable HTTP.
//
// HTTP transports can be protected with JWT bearer authentication
// (server.auth in the config, or REDISVL_MCP_AUTH_* environment
// variables), including per-tool read/write scope gating. Field-level
// schema overrides fill gaps in FT.INFO inspection, matching the Python
// server's schema_overrides semantics.
package mcpserver

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/redis/redis-vl-golang/vectors"
)

// Config is the YAML configuration for the MCP server (subset of the
// Python MCPConfig with matching field names).
type Config struct {
	Server   ServerConfig  `yaml:"server"`
	Index    IndexConfig   `yaml:"index"`
	Runtime  RuntimeConfig `yaml:"runtime"`
	ReadOnly bool          `yaml:"read_only"`

	// dtypeDefaulted records that runtime.dtype was defaulted rather than
	// configured, so schema inspection may refine it. Set-once so that
	// repeated Validate calls stay idempotent.
	dtypeDefaulted bool
}

// ServerConfig holds connection settings.
type ServerConfig struct {
	// RedisURL is the Redis connection URL (required; REDIS_URL env var is
	// used when empty).
	RedisURL string `yaml:"redis_url"`
	// Auth configures bearer-token authentication for HTTP transports
	// (optional; stdio is never authenticated). REDISVL_MCP_AUTH_*
	// environment variables override this block.
	Auth *AuthConfig `yaml:"auth"`
}

// AuthConfig configures JWT bearer authentication for the MCP server's
// HTTP transports (port of the Python MCPAuthConfig). Auth defaults to
// "none". When Type is "jwt" the server validates incoming bearer tokens
// against either a JWKS endpoint or a static public key, checking issuer,
// audience, required claims, and required scopes.
type AuthConfig struct {
	// Type is "none" (default) or "jwt".
	Type string `yaml:"type"`

	// JWKSURI fetches signing keys from a JWKS endpoint. Exactly one of
	// JWKSURI or PublicKey is required for type "jwt".
	JWKSURI string `yaml:"jwks_uri"`
	// PublicKey is a static PEM-encoded public key (RSA, EC, or Ed25519).
	PublicKey string `yaml:"public_key"`
	// Issuer is the required token issuer (required for type "jwt").
	Issuer string `yaml:"issuer"`
	// Audience is the required token audience (required for type "jwt";
	// RFC 8707 binds tokens to this server).
	Audience string `yaml:"audience"`
	// Algorithm restricts the accepted signing algorithm (e.g. RS256).
	// When empty, common asymmetric algorithms are accepted.
	Algorithm string `yaml:"algorithm"`
	// RequiredScopes must all be present on every token.
	RequiredScopes []string `yaml:"required_scopes"`
	// RequiredClaims must be present on every token. Defaults to exp and
	// iat so a token without an expiration (which would never expire) is
	// rejected.
	RequiredClaims []string `yaml:"required_claims"`
	// BaseURL is advertised as the protected-resource metadata URL in
	// WWW-Authenticate challenges (optional).
	BaseURL string `yaml:"base_url"`

	// ReadScope gates the search tool; WriteScope gates the upsert tool
	// (both optional; enforced per tool call).
	ReadScope  string `yaml:"read_scope"`
	WriteScope string `yaml:"write_scope"`
	// AuthorizationClaim is the token claim carrying authorization values
	// for the read/write gates. Standard OAuth uses "scp"/"scope"; some
	// IdPs (for example Azure AD / Entra) carry app roles in "roles".
	AuthorizationClaim string `yaml:"authorization_claim"`
}

// Enabled reports whether JWT authentication is configured.
func (a *AuthConfig) Enabled() bool { return a != nil && a.Type == "jwt" }

// readScope returns the configured search-tool scope gate ("" when unset
// or auth is disabled).
func (a *AuthConfig) readScope() string {
	if a == nil {
		return ""
	}
	return a.ReadScope
}

// writeScope returns the configured upsert-tool scope gate ("" when unset
// or auth is disabled).
func (a *AuthConfig) writeScope() string {
	if a == nil {
		return ""
	}
	return a.WriteScope
}

// IndexConfig binds the server to one existing Redis search index.
type IndexConfig struct {
	// RedisName is the name of the existing index (required).
	RedisName string `yaml:"redis_name"`
	// Vectorizer configures query/upsert embedding (optional; without it
	// only text search is available).
	Vectorizer *VectorizerConfig `yaml:"vectorizer"`
	// SchemaOverrides patch attrs of fields discovered from FT.INFO
	// (optional). Overrides cannot add fields or change a discovered
	// field's type or path.
	SchemaOverrides *SchemaOverrides `yaml:"schema_overrides"`
}

// SchemaOverrides is a set of field-level schema patches used to fill
// FT.INFO inspection gaps (port of the Python MCPSchemaOverrides).
type SchemaOverrides struct {
	Fields []SchemaOverrideField `yaml:"fields"`
}

// SchemaOverrideField patches one already-discovered field.
type SchemaOverrideField struct {
	// Name of the discovered field (required).
	Name string `yaml:"name"`
	// Type must match the discovered field type (required).
	Type string `yaml:"type"`
	// Path must match the discovered JSON path when set.
	Path string `yaml:"path"`
	// Attrs are merged over the discovered attrs (e.g. dims, datatype,
	// distance_metric for vector fields missing them in FT.INFO).
	Attrs map[string]any `yaml:"attrs"`
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

// authEnvOverrides merges REDISVL_MCP_AUTH_* environment variables over
// the YAML auth block (env wins, matching the Python server). An explicit
// REDISVL_MCP_AUTH_TYPE=none disables auth regardless of YAML.
func authEnvOverrides(yamlAuth *AuthConfig) *AuthConfig {
	env := func(name string) string { return strings.TrimSpace(os.Getenv("REDISVL_MCP_AUTH_" + name)) }
	csv := func(v string) []string {
		var out []string
		for _, item := range strings.Split(v, ",") {
			if s := strings.TrimSpace(item); s != "" {
				out = append(out, s)
			}
		}
		return out
	}

	if env("TYPE") == "none" {
		return nil
	}

	var a AuthConfig
	if yamlAuth != nil {
		a = *yamlAuth
	}
	set := func(dst *string, name string) {
		if v := env(name); v != "" {
			*dst = v
		}
	}
	set(&a.Type, "TYPE")
	set(&a.JWKSURI, "JWKS_URI")
	set(&a.PublicKey, "PUBLIC_KEY")
	set(&a.Issuer, "ISSUER")
	set(&a.Audience, "AUDIENCE")
	set(&a.Algorithm, "ALGORITHM")
	set(&a.BaseURL, "BASE_URL")
	set(&a.ReadScope, "READ_SCOPE")
	set(&a.WriteScope, "WRITE_SCOPE")
	set(&a.AuthorizationClaim, "AUTHORIZATION_CLAIM")
	if v := env("REQUIRED_SCOPES"); v != "" {
		a.RequiredScopes = csv(v)
	}
	if v := env("REQUIRED_CLAIMS"); v != "" {
		a.RequiredClaims = csv(v)
	}

	if a.Type == "" && a.JWKSURI == "" && a.PublicKey == "" && a.Issuer == "" &&
		a.Audience == "" && a.Algorithm == "" && a.BaseURL == "" &&
		a.ReadScope == "" && a.WriteScope == "" && a.AuthorizationClaim == "" &&
		len(a.RequiredScopes) == 0 && len(a.RequiredClaims) == 0 {
		return nil
	}
	return &a
}

// validateAuth applies defaults and enforces the Python server's JWT
// configuration rules.
func validateAuth(a *AuthConfig) error {
	if a == nil {
		return nil
	}
	jwtFieldsSet := a.JWKSURI != "" || a.PublicKey != "" || a.Issuer != "" ||
		a.Audience != "" || len(a.RequiredScopes) > 0 ||
		a.ReadScope != "" || a.WriteScope != ""

	switch a.Type {
	case "", "none":
		// Fail loudly rather than silently running unauthenticated when an
		// auth block looks like JWT but forgot to set the type.
		if jwtFieldsSet {
			return fmt.Errorf("mcp config: auth has JWT settings but type is not 'jwt'; set type: jwt to enable authentication")
		}
		return nil
	case "jwt":
	default:
		return fmt.Errorf("mcp config: unsupported auth type %q (supported: none, jwt)", a.Type)
	}

	if (a.JWKSURI != "") == (a.PublicKey != "") {
		return fmt.Errorf("mcp config: auth type 'jwt' requires exactly one of jwks_uri or public_key")
	}
	if strings.HasPrefix(strings.ToUpper(a.Algorithm), "HS") {
		return fmt.Errorf("mcp config: symmetric algorithm %q is not supported; use an asymmetric algorithm (RS*, PS*, ES*, EdDSA)", a.Algorithm)
	}
	if a.Issuer == "" {
		return fmt.Errorf("mcp config: auth type 'jwt' requires issuer; without it the verifier would accept tokens from any issuer")
	}
	if a.Audience == "" {
		return fmt.Errorf("mcp config: auth type 'jwt' requires audience to bind tokens to this server")
	}
	if a.RequiredClaims == nil {
		a.RequiredClaims = []string{"exp", "iat"}
	}
	if a.AuthorizationClaim == "" {
		a.AuthorizationClaim = "scp"
	}
	return nil
}

// validateSchemaOverrides checks override fragments for required fields.
func validateSchemaOverrides(o *SchemaOverrides) error {
	if o == nil {
		return nil
	}
	for i, f := range o.Fields {
		if f.Name == "" {
			return fmt.Errorf("mcp config: schema_overrides.fields[%d].name is required", i)
		}
		if f.Type == "" {
			return fmt.Errorf("mcp config: schema_overrides.fields[%d].type is required", i)
		}
	}
	return nil
}

// Validate applies defaults and checks required fields.
func (c *Config) Validate() error {
	if c.Server.RedisURL == "" {
		c.Server.RedisURL = os.Getenv("REDIS_URL")
	}
	if c.Server.RedisURL == "" {
		c.Server.RedisURL = "redis://localhost:6379"
	}
	c.Server.Auth = authEnvOverrides(c.Server.Auth)
	if err := validateAuth(c.Server.Auth); err != nil {
		return err
	}
	if err := validateSchemaOverrides(c.Index.SchemaOverrides); err != nil {
		return err
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
		c.dtypeDefaulted = true
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
