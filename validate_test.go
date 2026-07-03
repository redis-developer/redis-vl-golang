package redisvl

import (
	"strings"
	"testing"

	"github.com/redis-developer/redis-vl-golang/schema"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

func validationIndex(t *testing.T, storage schema.StorageType) *SearchIndex {
	t.Helper()
	vf, err := schema.NewVectorField("embedding", schema.VectorAttrs{
		Dims: 3, Algorithm: schema.Flat, Datatype: "float32",
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: "validate-idx", StorageType: storage},
		schema.NewTagField("user"),
		schema.NewNumericField("age"),
		schema.NewTextField("bio"),
		vf,
	)
	if err != nil {
		t.Fatal(err)
	}
	return NewSearchIndex(s, nil)
}

func TestValidateRecordHash(t *testing.T) {
	idx := validationIndex(t, schema.Hash)
	goodVec, _ := vectors.ToBuffer([]float64{1, 2, 3}, vectors.Float32)

	// valid record
	if err := idx.validateRecord(map[string]any{
		"user": "john", "age": 33, "bio": "hello", "embedding": goodVec,
	}); err != nil {
		t.Errorf("valid record rejected: %v", err)
	}

	// wrong vector size (2 dims instead of 3)
	shortVec, _ := vectors.ToBuffer([]float64{1, 2}, vectors.Float32)
	err := idx.validateRecord(map[string]any{"embedding": shortVec})
	if err == nil || !strings.Contains(err.Error(), "12 bytes") {
		t.Errorf("short vector: err = %v", err)
	}

	// float slice instead of bytes on hash storage
	if err := idx.validateRecord(map[string]any{"embedding": []float64{1, 2, 3}}); err == nil {
		t.Error("hash storage should reject non-[]byte vectors")
	}

	// non-numeric age
	if err := idx.validateRecord(map[string]any{"age": "not-a-number"}); err == nil {
		t.Error("expected numeric validation error")
	}
	// numeric string age is fine
	if err := idx.validateRecord(map[string]any{"age": "42"}); err != nil {
		t.Errorf("numeric string rejected: %v", err)
	}

	// missing fields are fine
	if err := idx.validateRecord(map[string]any{"user": "solo"}); err != nil {
		t.Errorf("partial record rejected: %v", err)
	}
}

func TestValidateRecordJSON(t *testing.T) {
	idx := validationIndex(t, schema.JSON)

	// JSON storage takes float slices
	if err := idx.validateRecord(map[string]any{"embedding": []float64{1, 2, 3}}); err != nil {
		t.Errorf("valid JSON vector rejected: %v", err)
	}
	// wrong dims
	if err := idx.validateRecord(map[string]any{"embedding": []float64{1, 2}}); err == nil {
		t.Error("expected dims validation error")
	}
	// []any from json.Unmarshal works
	if err := idx.validateRecord(map[string]any{"embedding": []any{1.0, 2.0, 3.0}}); err != nil {
		t.Errorf("[]any vector rejected: %v", err)
	}
	// bytes on JSON storage rejected
	if err := idx.validateRecord(map[string]any{"embedding": []byte{1, 2, 3}}); err == nil {
		t.Error("JSON storage should reject []byte vectors")
	}
}
