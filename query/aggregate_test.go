package query

import (
	"reflect"
	"testing"

	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/vectors"
)

func TestAggregateHybridQueryString(t *testing.T) {
	q := NewAggregateHybridQuery("capital of france", "description", []float64{0.1}, "embedding")
	want := "(~@description:(capital | france))=>[KNN 10 @embedding $vector AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}

	// with filter
	q = NewAggregateHybridQuery("hello", "description", []float64{0.1}, "embedding").
		Filter(filter.Tag("brand").Eq("nike"))
	want = "(~@description:(hello) AND @brand:{nike})=>[KNN 10 @embedding $vector AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestAggregateHybridArgs(t *testing.T) {
	q := NewAggregateHybridQuery("hello", "description", []float64{0.1}, "embedding").
		Alpha(0.5).
		NumResults(3).
		ReturnFields("title")
	args := q.AggregateArgs("idx")

	blob, _ := vectors.ToBuffer([]float64{0.1}, vectors.Float32)
	want := []any{
		"FT.AGGREGATE", "idx",
		"(~@description:(hello))=>[KNN 3 @embedding $vector AS vector_distance]",
		"LOAD", 1, "title",
		"SCORER", "BM25STD", "ADDSCORES",
		"APPLY", "(2 - @vector_distance)/2", "AS", "vector_similarity",
		"APPLY", "@__score", "AS", "text_score",
		"APPLY", "0.5*@text_score + 0.5*@vector_similarity", "AS", "hybrid_score",
		"SORTBY", 2, "@hybrid_score", "DESC", "MAX", 3,
		"PARAMS", 2, "vector", blob,
		"DIALECT", 2,
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("args mismatch\ngot:  %v\nwant: %v", args, want)
	}
}

func TestMultiVectorQuery(t *testing.T) {
	q, err := NewMultiVectorQuery(
		Vector{Values: []float64{0.1}, FieldName: "text_vector", Weight: 0.7},
		Vector{Values: []float64{0.2}, FieldName: "image_vector", Weight: 0.3, MaxDistance: 0.5},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "@text_vector:[VECTOR_RANGE 2 $vector_0]=>{$YIELD_DISTANCE_AS: distance_0}" +
		" AND @image_vector:[VECTOR_RANGE 0.5 $vector_1]=>{$YIELD_DISTANCE_AS: distance_1}"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}

	// with filter the range query is parenthesized
	q.Filter(filter.Num("price").Lt(100))
	wantFiltered := "(" + want + ") AND (@price:[-inf (100])"
	if got := q.QueryString(); got != wantFiltered {
		t.Errorf("got %q\nwant %q", got, wantFiltered)
	}

	args := q.AggregateArgs("idx")
	// spot-check the scoring pipeline
	found := map[string]bool{}
	for i, a := range args {
		s, ok := a.(string)
		if !ok {
			continue
		}
		switch s {
		case "(2 - @distance_0)/2", "(2 - @distance_1)/2",
			"@score_0 * 0.7 + @score_1 * 0.3", "@combined_score":
			found[s] = true
		case "PARAMS":
			if args[i+1] != 4 {
				t.Errorf("PARAMS count = %v, want 4", args[i+1])
			}
		}
	}
	for _, key := range []string{
		"(2 - @distance_0)/2", "(2 - @distance_1)/2",
		"@score_0 * 0.7 + @score_1 * 0.3", "@combined_score",
	} {
		if !found[key] {
			t.Errorf("args missing %q in %v", key, args)
		}
	}
}

func TestMultiVectorQueryValidation(t *testing.T) {
	if _, err := NewMultiVectorQuery(); err == nil {
		t.Error("expected error for no vectors")
	}
	if _, err := NewMultiVectorQuery(Vector{Values: []float64{1}}); err == nil {
		t.Error("expected error for missing field name")
	}
	if _, err := NewMultiVectorQuery(Vector{Values: []float64{1}, FieldName: "v", MaxDistance: 3}); err == nil {
		t.Error("expected error for out-of-range max distance")
	}
}

func TestHybridArgs(t *testing.T) {
	q := NewHybridQuery("hello world", "description", []float64{0.1}, "embedding").
		UseKNN(0). // keep default EF_RUNTIME 10
		CombineLinear(0.5).
		YieldCombinedScoreAs("hybrid_score").
		ReturnFields("title").
		NumResults(5)

	args, err := q.HybridArgs("idx")
	if err != nil {
		t.Fatal(err)
	}
	blob, _ := vectors.ToBuffer([]float64{0.1}, vectors.Float32)
	want := []any{
		"FT.HYBRID", "idx",
		"SEARCH", "(~@description:(hello | world))",
		"SCORER", "BM25STD",
		"VSIM", "@embedding", "$vector",
		"KNN", 4, "K", 5, "EF_RUNTIME", 10,
		"COMBINE", "LINEAR", 6, "ALPHA", "0.5", "BETA", "0.5",
		"YIELD_SCORE_AS", "hybrid_score",
		"LOAD", 1, "@title",
		"LIMIT", 0, 5,
		"PARAMS", 2, "vector", blob,
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("args mismatch\ngot:  %v\nwant: %v", args, want)
	}
}

func TestHybridArgsValidation(t *testing.T) {
	// empty text
	q := NewHybridQuery("   ", "description", []float64{0.1}, "embedding")
	if _, err := q.HybridArgs("idx"); err == nil {
		t.Error("expected empty-text error")
	}
	// RANGE without radius
	q = NewHybridQuery("hi", "description", []float64{0.1}, "embedding")
	q.searchMethod = Range
	if _, err := q.HybridArgs("idx"); err == nil {
		t.Error("expected missing-radius error")
	}
}
