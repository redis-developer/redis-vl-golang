package cache

import (
	"testing"
)

func TestParseEmbeddingEntry(t *testing.T) {
	data := map[string]string{
		entryIDField:    "abc",
		textContent:     "hello",
		modelNameField:  "test-model",
		embeddingField:  "[0.1,0.2]",
		insertedAtField: "1705320000.5",
		metadataField:   `{"k":"v"}`,
	}
	entry, err := parseEmbeddingEntry(data)
	if err != nil {
		t.Fatal(err)
	}
	if entry.EntryID != "abc" || entry.Content != "hello" || entry.ModelName != "test-model" {
		t.Errorf("entry = %+v", entry)
	}
	if len(entry.Embedding) != 2 || entry.Embedding[1] != 0.2 {
		t.Errorf("embedding = %v", entry.Embedding)
	}
	if entry.InsertedAt != 1705320000.5 || entry.Metadata["k"] != "v" {
		t.Errorf("entry = %+v", entry)
	}
}

func TestParseEmbeddingEntryBadJSON(t *testing.T) {
	if _, err := parseEmbeddingEntry(map[string]string{embeddingField: "not-json"}); err == nil {
		t.Error("expected error for invalid embedding json")
	}
}

func TestParseCacheHit(t *testing.T) {
	doc := map[string]any{
		"id":              "llmcache:xyz",
		entryIDField:      "xyz",
		promptField:       "what is the capital of France?",
		responseField:     "Paris",
		"vector_distance": "0.05",
		insertedAtField:   "1705320000",
		updatedAtField:    "1705320001",
		metadataField:     `{"model":"gpt"}`,
		"label":           "geo", // filterable extra field
	}
	hit := parseCacheHit(doc)
	if hit.Key != "llmcache:xyz" || hit.EntryID != "xyz" || hit.Response != "Paris" {
		t.Errorf("hit = %+v", hit)
	}
	if hit.VectorDistance != 0.05 || hit.InsertedAt != 1705320000 || hit.UpdatedAt != 1705320001 {
		t.Errorf("hit = %+v", hit)
	}
	if hit.Metadata["model"] != "gpt" {
		t.Errorf("metadata = %v", hit.Metadata)
	}
	if hit.Filters["label"] != "geo" {
		t.Errorf("filters = %v", hit.Filters)
	}
}

func TestEmbeddingsCacheKeyDeterminism(t *testing.T) {
	c := NewEmbeddingsCache(nil)
	k1 := c.Key("hello", "model-a")
	k2 := c.Key("hello", "model-a")
	k3 := c.Key("hello", "model-b")
	if k1 != k2 {
		t.Error("keys must be deterministic")
	}
	if k1 == k3 {
		t.Error("different models must produce different keys")
	}
	if len(k1) != len("embedcache:")+64 { // sha256 hex
		t.Errorf("unexpected key %q", k1)
	}
}
