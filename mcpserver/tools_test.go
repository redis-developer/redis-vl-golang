package mcpserver

import (
	"context"
	"testing"

	redisvl "github.com/redis-developer/redis-vl-golang"
	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/schema"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// testServer builds a Server without touching Redis (schema-only paths).
func testServer(t *testing.T, storage schema.StorageType, readOnly bool) *Server {
	t.Helper()
	s, err := schema.NewIndexSchema(schema.IndexInfo{
		Name: "test-idx", Prefixes: []string{"test-idx"}, StorageType: storage,
	})
	if err != nil {
		t.Fatal(err)
	}
	skip := true
	cfg := &Config{
		Index: IndexConfig{RedisName: "test-idx"},
		Runtime: RuntimeConfig{
			VectorFieldName:        "embedding",
			DefaultEmbedTextField:  "description",
			DefaultLimit:           10,
			MaxLimit:               100,
			MaxUpsertRecords:       2,
			SkipEmbeddingIfPresent: &skip,
			Dtype:                  "float32",
		},
	}
	return &Server{
		cfg:      cfg,
		index:    redisvl.NewSearchIndex(s, nil),
		dtype:    vectors.Float32,
		readOnly: readOnly,
		vectorizer: &vectorize.Func{
			Dimensions: 2,
			EmbedFunc: func(_ context.Context, text string) ([]float64, error) {
				return []float64{float64(len(text)), 1}, nil
			},
		},
	}
}

func TestEmbedRecordsHashStorage(t *testing.T) {
	s := testServer(t, schema.Hash, false)
	records := []map[string]any{
		{"description": "hello"},
		{"description": "world", "embedding": []byte{1, 2, 3, 4, 5, 6, 7, 8}}, // pre-embedded: skipped
		{"title": "no text field"}, // no embed source: skipped
	}
	out, err := s.embedRecords(context.Background(), records)
	if err != nil {
		t.Fatal(err)
	}
	blob, ok := out[0]["embedding"].([]byte)
	if !ok || len(blob) != 8 { // 2 float32
		t.Errorf("record 0 embedding = %T %v", out[0]["embedding"], out[0]["embedding"])
	}
	if b := out[1]["embedding"].([]byte); len(b) != 8 || b[0] != 1 {
		t.Errorf("pre-embedded record should be untouched: %v", b)
	}
	if _, ok := out[2]["embedding"]; ok {
		t.Error("record without text field should not be embedded")
	}
}

func TestEmbedRecordsJSONStorage(t *testing.T) {
	s := testServer(t, schema.JSON, false)
	out, err := s.embedRecords(context.Background(), []map[string]any{
		{"description": "hi"},
	})
	if err != nil {
		t.Fatal(err)
	}
	emb, ok := out[0]["embedding"].([]float64)
	if !ok || len(emb) != 2 {
		t.Errorf("JSON storage should store float slice, got %T", out[0]["embedding"])
	}
}

func TestDoUpsertValidation(t *testing.T) {
	// read-only refusal
	s := testServer(t, schema.Hash, true)
	if _, err := s.doUpsert(context.Background(), UpsertArgs{
		Records: []map[string]any{{"description": "x"}},
	}); err == nil {
		t.Error("expected read-only error")
	}

	s = testServer(t, schema.Hash, false)
	if _, err := s.doUpsert(context.Background(), UpsertArgs{}); err == nil {
		t.Error("expected empty-records error")
	}
	// exceeds MaxUpsertRecords (2)
	if _, err := s.doUpsert(context.Background(), UpsertArgs{
		Records: []map[string]any{{"a": 1}, {"a": 2}, {"a": 3}},
	}); err == nil {
		t.Error("expected too-many-records error")
	}
}

func TestSanitizeRecords(t *testing.T) {
	s := testServer(t, schema.Hash, false)
	out := s.sanitizeRecords([]map[string]any{
		{"id": "1", "title": "x", "embedding": "\x00\x01binary"},
	})
	if _, ok := out[0]["embedding"]; ok {
		t.Error("vector field should be stripped from results")
	}
	if out[0]["title"] != "x" {
		t.Errorf("other fields should survive: %v", out[0])
	}
}

func TestDoSearchValidation(t *testing.T) {
	s := testServer(t, schema.Hash, false)
	if _, err := s.doSearch(context.Background(), SearchArgs{Query: ""}); err == nil {
		t.Error("expected empty-query error")
	}
}
