package mcpserver

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mcp.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadConfigDefaults(t *testing.T) {
	path := writeConfig(t, `
server:
  redis_url: redis://example:6379
index:
  redis_name: movies
  vectorizer:
    class: openai
    model: text-embedding-3-small
runtime:
  vector_field_name: embedding
  default_embed_text_field: description
`)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.RedisURL != "redis://example:6379" || cfg.Index.RedisName != "movies" {
		t.Errorf("cfg = %+v", cfg)
	}
	// defaults applied
	if cfg.Runtime.DefaultLimit != 10 || cfg.Runtime.MaxLimit != 100 ||
		cfg.Runtime.MaxUpsertRecords != 64 || cfg.Runtime.Dtype != "float32" {
		t.Errorf("runtime defaults = %+v", cfg.Runtime)
	}
	if cfg.Runtime.SkipEmbeddingIfPresent == nil || !*cfg.Runtime.SkipEmbeddingIfPresent {
		t.Error("skip_embedding_if_present should default to true")
	}
	if cfg.ReadOnly {
		t.Error("read_only should default to false")
	}
}

func TestLoadConfigValidation(t *testing.T) {
	cases := []struct {
		name string
		yaml string
	}{
		{"missing index name", `
server: {redis_url: redis://x}
runtime: {text_field_name: description}
`},
		{"vectorizer without vector field", `
index:
  redis_name: idx
  vectorizer: {class: openai}
`},
		{"unsupported vectorizer class", `
index:
  redis_name: idx
  vectorizer: {class: huggingface}
runtime: {vector_field_name: embedding}
`},
		{"neither vectorizer nor text field", `
index: {redis_name: idx}
`},
		{"invalid dtype", `
index: {redis_name: idx}
runtime: {text_field_name: d, dtype: float128}
`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := LoadConfig(writeConfig(t, c.yaml)); err == nil {
				t.Errorf("expected validation error")
			}
		})
	}
}

func TestClampLimit(t *testing.T) {
	rt := RuntimeConfig{DefaultLimit: 10, MaxLimit: 100}
	cases := map[int]int{0: 10, -5: 10, 7: 7, 100: 100, 500: 100}
	for in, want := range cases {
		if got := rt.clampLimit(in); got != want {
			t.Errorf("clampLimit(%d) = %d, want %d", in, got, want)
		}
	}
}

func TestTextOnlyConfig(t *testing.T) {
	path := writeConfig(t, `
index: {redis_name: docs}
runtime: {text_field_name: content}
`)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Index.Vectorizer != nil {
		t.Error("vectorizer should be nil for text-only config")
	}
}
