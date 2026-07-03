package schema

import (
	"reflect"
	"testing"
)

const sampleYAML = `
version: '0.1.0'

index:
    name: user-index
    prefix: user
    key_separator: ":"
    storage_type: json

fields:
    - name: user
      type: tag
    - name: credit_score
      type: tag
    - name: embedding
      type: vector
      attrs:
        algorithm: flat
        dims: 3
        distance_metric: cosine
        datatype: float32
`

func TestFromYAML(t *testing.T) {
	s, err := FromYAML([]byte(sampleYAML))
	if err != nil {
		t.Fatal(err)
	}
	if s.Index.Name != "user-index" {
		t.Errorf("name = %q", s.Index.Name)
	}
	if !reflect.DeepEqual(s.Index.Prefixes, []string{"user"}) {
		t.Errorf("prefixes = %v", s.Index.Prefixes)
	}
	if s.Index.StorageType != JSON {
		t.Errorf("storage = %q", s.Index.StorageType)
	}
	if !reflect.DeepEqual(s.FieldNames(), []string{"user", "credit_score", "embedding"}) {
		t.Errorf("fields = %v", s.FieldNames())
	}
	// JSON storage defaults paths
	if got := s.Field("user").Path; got != "$.user" {
		t.Errorf("path = %q", got)
	}
	v := s.Field("embedding").Vector
	if v == nil {
		t.Fatal("embedding is not a vector field")
	}
	if v.Dims != 3 || v.Algorithm != Flat || v.DistanceMetric != Cosine || v.Datatype != "float32" {
		t.Errorf("vector attrs = %+v", v)
	}
}

func TestYAMLRoundtrip(t *testing.T) {
	s, err := FromYAML([]byte(sampleYAML))
	if err != nil {
		t.Fatal(err)
	}
	data, err := s.ToYAML()
	if err != nil {
		t.Fatal(err)
	}
	s2, err := FromYAML(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.FieldNames(), s2.FieldNames()) {
		t.Errorf("field names differ: %v vs %v", s.FieldNames(), s2.FieldNames())
	}
	if !reflect.DeepEqual(s.Index, s2.Index) {
		t.Errorf("index info differs: %+v vs %+v", s.Index, s2.Index)
	}
}

func TestDuplicateFieldNames(t *testing.T) {
	s, err := NewIndexSchema(IndexInfo{Name: "idx"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.AddField(NewTagField("user")); err != nil {
		t.Fatal(err)
	}
	if err := s.AddField(NewTextField("user")); err == nil {
		t.Error("expected duplicate field error")
	}
}

func TestHashStorageIgnoresPath(t *testing.T) {
	s, _ := NewIndexSchema(IndexInfo{Name: "idx", StorageType: Hash})
	f := NewTagField("user")
	f.Path = "$.user"
	if err := s.AddField(f); err != nil {
		t.Fatal(err)
	}
	if got := s.Field("user").Path; got != "" {
		t.Errorf("path should be cleared for hash storage, got %q", got)
	}
}

func TestVectorValidation(t *testing.T) {
	// SVS-VAMANA rejects float64
	_, err := NewVectorField("v", VectorAttrs{
		Dims: 4, Algorithm: SVSVamana, Datatype: "float64",
	})
	if err == nil {
		t.Error("expected SVS datatype error")
	}
	// reduce requires LeanVec compression
	_, err = NewVectorField("v", VectorAttrs{
		Dims: 4, Algorithm: SVSVamana, Datatype: "float32",
		Reduce: IntPtr(2), Compression: LVQ8,
	})
	if err == nil {
		t.Error("expected reduce/compression error")
	}
	// reduce must be < dims
	_, err = NewVectorField("v", VectorAttrs{
		Dims: 4, Algorithm: SVSVamana, Datatype: "float32",
		Reduce: IntPtr(8), Compression: LeanVec4x8,
	})
	if err == nil {
		t.Error("expected reduce < dims error")
	}
	// missing algorithm
	bad := VectorAttrs{Dims: 4}
	if err := bad.Validate(); err == nil {
		t.Error("expected unknown algorithm error")
	}
}

func TestRedisArgs(t *testing.T) {
	f := NewTagField("user")
	args, err := f.RedisArgs()
	if err != nil {
		t.Fatal(err)
	}
	want := []any{"user", "TAG", "SEPARATOR", ","}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("tag args = %v, want %v", args, want)
	}

	vf, err := NewVectorField("embedding", VectorAttrs{Dims: 3, Algorithm: Flat})
	if err != nil {
		t.Fatal(err)
	}
	args, err = vf.RedisArgs()
	if err != nil {
		t.Fatal(err)
	}
	want = []any{
		"embedding", "VECTOR", "FLAT", 6,
		"TYPE", "FLOAT32", "DIM", 3, "DISTANCE_METRIC", "COSINE",
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("vector args = %v, want %v", args, want)
	}

	// text with modifiers in canonical order
	tf := NewTextField("title", TextAttrs{
		BaseAttrs: BaseAttrs{Sortable: true, IndexMissing: true},
		UNF:       true,
	})
	args, err = tf.RedisArgs()
	if err != nil {
		t.Fatal(err)
	}
	want = []any{"title", "TEXT", "INDEXMISSING", "SORTABLE", "UNF"}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("text args = %v, want %v", args, want)
	}

	// JSON path alias
	jf := NewTagField("user")
	jf.Path = "$.user"
	args, err = jf.RedisArgs()
	if err != nil {
		t.Fatal(err)
	}
	want = []any{"$.user", "AS", "user", "TAG", "SEPARATOR", ","}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("json tag args = %v, want %v", args, want)
	}
}

func TestInferFieldType(t *testing.T) {
	cases := []struct {
		value any
		want  FieldType
	}{
		{42, TypeNumeric},
		{3.14, TypeNumeric},
		{"123.4", TypeNumeric},
		{"37.7749,-122.4194", TypeGeo},
		{"hello world", TypeText},
		{[]string{"a", "b"}, TypeTag},
	}
	for _, c := range cases {
		if got := InferFieldType(c.value); got != c.want {
			t.Errorf("InferFieldType(%v) = %q, want %q", c.value, got, c.want)
		}
	}
}
