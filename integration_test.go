package redisvl

// Integration tests run against REDIS_URL when set, and otherwise start a
// Redis 8 container automatically via testcontainers-go (skipped when
// Docker is unavailable):
//
//	go test ./...                                    # testcontainers
//	REDIS_URL=redis://localhost:6379 go test ./...   # external server

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/internal/redistest"
	"github.com/redis/redis-vl-golang/query"
	"github.com/redis/redis-vl-golang/schema"
	"github.com/redis/redis-vl-golang/vectors"
)

func testIndex(t *testing.T) (*SearchIndex, context.Context) {
	t.Helper()
	url := redistest.URL(t)
	ctx := context.Background()

	vf, err := schema.NewVectorField("embedding", schema.VectorAttrs{
		Dims: 3, Algorithm: schema.Flat, Datatype: "float32",
		DistanceMetric: schema.Cosine,
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := schema.NewIndexSchema(schema.IndexInfo{
		Name:     "rvlgo-test",
		Prefixes: []string{"rvlgo-test"},
	},
		schema.NewTagField("user"),
		schema.NewTagField("credit_score"),
		schema.NewNumericField("age"),
		vf,
	)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := NewSearchIndexFromURL(s, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = idx.Delete(ctx, true)
		_ = idx.Close()
	})
	return idx, ctx
}

func TestIntegrationEndToEnd(t *testing.T) {
	idx, ctx := testIndex(t)

	if err := idx.Create(ctx, CreateOptions{Overwrite: true, Drop: true}); err != nil {
		t.Fatal(err)
	}
	exists, err := idx.Exists(ctx)
	if err != nil || !exists {
		t.Fatalf("exists=%v err=%v", exists, err)
	}

	vec := func(vals ...float64) []byte {
		b, err := vectors.ToBuffer(vals, vectors.Float32)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	data := []map[string]any{
		{"user": "john", "credit_score": "high", "age": 33, "embedding": vec(0.1, 0.1, 0.5)},
		{"user": "mary", "credit_score": "low", "age": 14, "embedding": vec(0.1, 0.1, 0.5)},
		{"user": "joe", "credit_score": "medium", "age": 35, "embedding": vec(0.9, 0.9, 0.1)},
	}
	keys, err := idx.Load(ctx, data, LoadOptions{IDField: "user"})
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 3 {
		t.Fatalf("keys = %v", keys)
	}

	// fetch
	doc, err := idx.Fetch(ctx, "john")
	if err != nil || doc == nil {
		t.Fatalf("fetch: doc=%v err=%v", doc, err)
	}

	// count
	total, err := idx.Count(ctx, query.NewCountQuery(nil))
	if err != nil || total != 3 {
		t.Fatalf("count=%d err=%v", total, err)
	}

	// vector query with filter
	q := query.NewVectorQuery("embedding", []float64{0.1, 0.1, 0.5}).
		NumResults(3).
		Filter(filter.Tag("credit_score").Eq("high")).
		ReturnFields("user", "credit_score")
	docs, err := idx.Query(ctx, q)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0]["user"] != "john" {
		t.Fatalf("docs = %v", docs)
	}
	if _, ok := docs[0][query.DistanceID]; !ok {
		t.Errorf("missing vector distance: %v", docs[0])
	}

	// range query
	rq := query.NewVectorRangeQuery("embedding", []float64{0.1, 0.1, 0.5}).
		DistanceThreshold(0.2).
		ReturnFields("user")
	docs, err = idx.Query(ctx, rq)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("range docs = %v", docs)
	}

	// filter query
	fq := query.NewFilterQuery(filter.Num("age").Gt(30)).ReturnFields("user", "age")
	docs, err = idx.Query(ctx, fq)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("filter docs = %v", docs)
	}

	// clear
	n, err := idx.Clear(ctx)
	if err != nil || n != 3 {
		t.Fatalf("clear=%d err=%v", n, err)
	}
}

func TestIntegrationBatchAndFetchMany(t *testing.T) {
	idx, ctx := testIndex(t)
	if err := idx.Create(ctx, CreateOptions{Overwrite: true, Drop: true}); err != nil {
		t.Fatal(err)
	}

	vec := func(vals ...float64) []byte {
		b, err := vectors.ToBuffer(vals, vectors.Float32)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	data := []map[string]any{
		{"user": "john", "credit_score": "high", "age": 33, "embedding": vec(0.1, 0.1, 0.5)},
		{"user": "mary", "credit_score": "low", "age": 14, "embedding": vec(0.1, 0.1, 0.5)},
		{"user": "joe", "credit_score": "medium", "age": 35, "embedding": vec(0.9, 0.9, 0.1)},
	}
	// exercise the validation path live
	if _, err := idx.Load(ctx, data, LoadOptions{IDField: "user", Validate: true}); err != nil {
		t.Fatal(err)
	}

	// validation rejects a malformed vector before writing
	if _, err := idx.Load(ctx, []map[string]any{
		{"user": "bad", "embedding": []byte{1, 2}},
	}, LoadOptions{IDField: "user", Validate: true}); err == nil {
		t.Error("expected validation error for malformed vector")
	}

	// FetchMany preserves order and marks misses as nil
	docs, err := idx.FetchMany(ctx, "john", "nobody", "mary")
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 3 || docs[0] == nil || docs[1] != nil || docs[2] == nil {
		t.Fatalf("fetchMany = %v", docs)
	}
	if docs[0]["credit_score"] != "high" {
		t.Errorf("doc 0 = %v", docs[0])
	}

	// BatchQuery runs mixed queries in one pipeline, in order
	queries := []query.Query{
		query.NewFilterQuery(filter.Num("age").Gt(30)).ReturnFields("user"),
		query.NewFilterQuery(filter.Tag("credit_score").Eq("low")).ReturnFields("user"),
		query.NewVectorQuery("embedding", []float64{0.9, 0.9, 0.1}).
			NumResults(1).ReturnFields("user"),
	}
	results, err := idx.BatchQuery(ctx, queries)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("batch results = %d", len(results))
	}
	if len(results[0]) != 2 {
		t.Errorf("query 0: %v", results[0])
	}
	if len(results[1]) != 1 || results[1][0]["user"] != "mary" {
		t.Errorf("query 1: %v", results[1])
	}
	if len(results[2]) != 1 || results[2][0]["user"] != "joe" {
		t.Errorf("query 2: %v", results[2])
	}
}

func TestIntegrationAggregateAndHybrid(t *testing.T) {
	url := redistest.URL(t)
	ctx := context.Background()

	vf1, err := schema.NewVectorField("embedding", schema.VectorAttrs{
		Dims: 2, Algorithm: schema.Flat, DistanceMetric: schema.Cosine,
	})
	if err != nil {
		t.Fatal(err)
	}
	vf2, err := schema.NewVectorField("image_embedding", schema.VectorAttrs{
		Dims: 2, Algorithm: schema.Flat, DistanceMetric: schema.Cosine,
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: "rvlgo-agg-test", Prefixes: []string{"rvlgo-agg-test"}},
		schema.NewTextField("description"),
		vf1, vf2,
	)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := NewSearchIndexFromURL(s, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = idx.Delete(ctx, true)
		_ = idx.Close()
	})
	if err := idx.Create(ctx, CreateOptions{Overwrite: true, Drop: true}); err != nil {
		t.Fatal(err)
	}

	vec := func(vals ...float64) []byte {
		b, err := vectors.ToBuffer(vals, vectors.Float32)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	_, err = idx.Load(ctx, []map[string]any{
		{"description": "the capital of france is paris", "embedding": vec(1, 0), "image_embedding": vec(1, 0)},
		{"description": "bread baking with sourdough starter", "embedding": vec(0, 1), "image_embedding": vec(0, 1)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// unsupportedServer reports errors caused by the Redis server not
	// knowing an argument/command yet (older RediSearch versions), so the
	// corresponding subtest can skip instead of fail.
	unsupportedServer := func(err error) bool {
		msg := strings.ToLower(err.Error())
		return strings.Contains(msg, "unknown argument") ||
			strings.Contains(msg, "unknown command") ||
			strings.Contains(msg, "unrecognized")
	}

	t.Run("MultiVectorQuery", func(t *testing.T) {
		// weight both fields toward doc 2
		mq, err := query.NewMultiVectorQuery(
			query.Vector{Values: []float64{0, 1}, FieldName: "embedding", Weight: 0.5},
			query.Vector{Values: []float64{0, 1}, FieldName: "image_embedding", Weight: 0.5},
		)
		if err != nil {
			t.Fatal(err)
		}
		rows, err := idx.Aggregate(ctx, mq.ReturnFields("description"))
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) == 0 {
			t.Fatal("no multi-vector rows")
		}
		if d := fmt.Sprint(rows[0]["description"]); d != "bread baking with sourdough starter" {
			t.Errorf("top row = %v", rows[0])
		}
		if _, ok := rows[0]["combined_score"]; !ok {
			t.Errorf("missing combined_score: %v", rows[0])
		}
	})

	t.Run("AggregateHybridQuery", func(t *testing.T) {
		// text and vector both point at doc 1; requires SCORER/ADDSCORES
		// support in FT.AGGREGATE (Redis 8.x)
		aq := query.NewAggregateHybridQuery("capital france", "description", []float64{1, 0}, "embedding").
			ReturnFields("description")
		rows, err := idx.Aggregate(ctx, aq)
		if err != nil {
			if unsupportedServer(err) {
				t.Skipf("FT.AGGREGATE SCORER/ADDSCORES not supported by this Redis server: %v", err)
			}
			t.Fatal(err)
		}
		if len(rows) == 0 {
			t.Fatal("no aggregate rows")
		}
		if d := fmt.Sprint(rows[0]["description"]); d != "the capital of france is paris" {
			t.Errorf("top row = %v", rows[0])
		}
		if _, ok := rows[0]["hybrid_score"]; !ok {
			t.Errorf("missing hybrid_score: %v", rows[0])
		}
	})

	t.Run("HybridQuery", func(t *testing.T) {
		// native FT.HYBRID requires Redis >= 8.4
		hq := query.NewHybridQuery("capital france", "description", []float64{1, 0}, "embedding").
			CombineLinear(0.5).
			ReturnFields("description")
		rows, err := idx.Hybrid(ctx, hq)
		if err != nil {
			if unsupportedServer(err) {
				t.Skipf("FT.HYBRID not supported by this Redis server (< 8.4): %v", err)
			}
			t.Fatal(err)
		}
		if len(rows) == 0 {
			t.Fatal("no hybrid rows")
		}
	})
}
