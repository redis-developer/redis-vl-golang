package query

import (
	"reflect"
	"strings"
	"testing"

	"github.com/redis/redis-vl-golang/filter"
)

func TestVectorQueryString(t *testing.T) {
	q := NewVectorQuery("embedding", []float64{0.1, 0.2, 0.3})
	want := "*=>[KNN 10 @embedding $vector AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	q = NewVectorQuery("embedding", []float64{0.1}).
		NumResults(5).
		Filter(filter.Tag("brand").Eq("nike"))
	want = "@brand:{nike}=>[KNN 5 @embedding $vector AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	q = NewVectorQuery("embedding", []float64{0.1}).EfRuntime(20)
	want = "*=>[KNN 10 @embedding $vector EF_RUNTIME $EF AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	q = NewVectorQuery("embedding", []float64{0.1}).
		SetHybridPolicy(Batches).BatchSize(50)
	want = "*=>[KNN 10 @embedding $vector HYBRID_POLICY BATCHES BATCH_SIZE 50 AS vector_distance]"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestVectorQueryParams(t *testing.T) {
	q := NewVectorQuery("embedding", []float64{0.1, 0.2}).EfRuntime(20)
	params := q.Params()
	if _, ok := params["vector"].([]byte); !ok {
		t.Error("params missing vector bytes")
	}
	if params["EF"] != 20 {
		t.Errorf("EF = %v", params["EF"])
	}
	if blob := params["vector"].([]byte); len(blob) != 8 { // 2 x float32
		t.Errorf("vector blob length = %d", len(blob))
	}
}

func TestVectorRangeQueryString(t *testing.T) {
	q := NewVectorRangeQuery("embedding", []float64{0.1})
	want := "@embedding:[VECTOR_RANGE $distance_threshold $vector]=>{$YIELD_DISTANCE_AS: vector_distance}"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	q = NewVectorRangeQuery("embedding", []float64{0.1}).
		Filter(filter.Tag("brand").Eq("nike")).
		Epsilon(0.05)
	want = "(@embedding:[VECTOR_RANGE $distance_threshold $vector]=>{$YIELD_DISTANCE_AS: vector_distance; $EPSILON: 0.05} @brand:{nike})"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	params := q.Params()
	if params["distance_threshold"] != 0.2 {
		t.Errorf("distance_threshold = %v", params["distance_threshold"])
	}
}

func TestNormalizedRangeThreshold(t *testing.T) {
	q := NewVectorRangeQuery("embedding", []float64{0.1}).
		NormalizeVectorDistance().
		DistanceThreshold(0.9)
	// 0.9 similarity -> 0.2 cosine distance (2 - 2*0.9)
	got, ok := q.Params()["distance_threshold"].(float64)
	if !ok || got < 0.199 || got > 0.201 {
		t.Errorf("denormalized threshold = %v", q.Params()["distance_threshold"])
	}
}

func TestFilterAndCountQueries(t *testing.T) {
	fq := NewFilterQuery(filter.Tag("brand").Eq("nike")).ReturnFields("brand", "price")
	if got := fq.QueryString(); got != "@brand:{nike}" {
		t.Errorf("got %q", got)
	}
	args := BuildSearchArgs("idx", fq)
	want := []any{
		"FT.SEARCH", "idx", "@brand:{nike}",
		"RETURN", 2, "brand", "price",
		"LIMIT", 0, 10, "DIALECT", 2,
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("args = %v\nwant %v", args, want)
	}

	cq := NewCountQuery(nil)
	args = BuildSearchArgs("idx", cq)
	want = []any{"FT.SEARCH", "idx", "*", "NOCONTENT", "LIMIT", 0, 0, "DIALECT", 2}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("count args = %v\nwant %v", args, want)
	}
}

func TestBuildSearchArgsVector(t *testing.T) {
	q := NewVectorQuery("embedding", []float64{0.1}).NumResults(3)
	args := BuildSearchArgs("idx", q)

	// spot-check structure: ... RETURN 1 vector_distance SORTBY vector_distance ASC
	// LIMIT 0 3 PARAMS 2 vector <blob> DIALECT 2
	str := make([]string, 0, len(args))
	for _, a := range args {
		if s, ok := a.(string); ok {
			str = append(str, s)
		} else {
			str = append(str, "#")
		}
	}
	joined := strings.Join(str, " ")
	wantSubstrings := []string{
		"FT.SEARCH idx",
		"RETURN # vector_distance",
		"SORTBY vector_distance ASC",
		"LIMIT # #",
		"PARAMS # vector",
		"DIALECT #",
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(joined, sub) {
			t.Errorf("args missing %q in %q", sub, joined)
		}
	}
}

func TestTextQuery(t *testing.T) {
	q := NewTextQuery("the hello world", "description")
	want := "@description:(hello | world)"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// punctuation is escaped, stopwords removed, lowercased
	q = NewTextQuery("The quick-fox", "description")
	want = `@description:(quick\-fox)`
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// filter combination
	q = NewTextQuery("hello", "description").Filter(filter.Tag("brand").Eq("nike"))
	want = "@description:(hello) AND @brand:{nike}"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// field weights
	q = NewTextQuery("hello", "title").FieldWeights(map[string]float64{"title": 2})
	want = "@title:(hello) => { $weight: 2 }"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// text weights
	q = NewTextQuery("hello world", "description").
		TextWeights(map[string]float64{"world": 2.5})
	want = "@description:(hello | world=>{$weight:2.5})"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// no stopword removal
	q = NewTextQuery("the hello", "description").Stopwords()
	want = "@description:(the | hello)"
	if got := q.QueryString(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestTextQueryScorerOptions(t *testing.T) {
	q := NewTextQuery("hello", "description")
	o := q.Options()
	if o.Scorer != "BM25STD" || !o.WithScores {
		t.Errorf("default scorer options wrong: %+v", o)
	}
	args := BuildSearchArgs("idx", q)
	found := false
	for i, a := range args {
		if a == "SCORER" && i+1 < len(args) && args[i+1] == "BM25STD" {
			found = true
		}
	}
	if !found {
		t.Errorf("SCORER BM25STD not in args: %v", args)
	}
}
