package main

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// Regression test: FT.INFO replies over RESP3 contain nested map[any]any
// values that encoding/json cannot marshal directly.
func TestNormalizeJSONMakesRESP3Encodable(t *testing.T) {
	resp3 := map[string]any{
		"index_name": "movies",
		"index_definition": map[any]any{
			"key_type":  "HASH",
			"prefixes":  []any{"movie"},
			"nested":    map[any]any{"a": int64(1)},
			"more_list": []any{map[any]any{"b": "c"}},
		},
	}

	// raw encode must fail (this is the bug condition)
	if _, err := json.Marshal(resp3); err == nil {
		t.Fatal("expected raw RESP3 map to be unmarshalable — test setup wrong")
	}

	out, err := json.Marshal(normalizeJSON(resp3))
	if err != nil {
		t.Fatalf("normalized encode failed: %v", err)
	}
	var back map[string]any
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatal(err)
	}
	def, ok := back["index_definition"].(map[string]any)
	if !ok || def["key_type"] != "HASH" {
		t.Errorf("normalized structure wrong: %s", out)
	}
}

// Regression test: RediSearch reports averages as NaN on empty indexes;
// encoding/json rejects non-finite floats.
func TestNormalizeJSONHandlesNonFiniteFloats(t *testing.T) {
	stats := map[string]any{
		"num_docs":             int64(0),
		"offsets_per_term_avg": math.NaN(),
		"records_per_doc_avg":  math.Inf(1),
		"bytes_per_record_avg": math.Inf(-1),
		"percent_indexed":      1.0,
		"nested":               map[any]any{"avg": math.NaN()},
	}
	if _, err := json.Marshal(stats); err == nil {
		t.Fatal("expected raw NaN map to be unmarshalable — test setup wrong")
	}
	out, err := json.Marshal(normalizeJSON(stats))
	if err != nil {
		t.Fatalf("normalized encode failed: %v", err)
	}
	s := string(out)
	if !strings.Contains(s, `"offsets_per_term_avg":"NaN"`) {
		t.Errorf("NaN not stringified: %s", s)
	}
	if !strings.Contains(s, `"percent_indexed":1`) {
		t.Errorf("finite float mangled: %s", s)
	}
}
