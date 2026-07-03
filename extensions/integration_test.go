// Package extensions_test holds end-to-end tests for the extension
// packages. They require a running Redis Stack / Redis 8 instance and are
// skipped unless REDIS_URL is set.
package extensions_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/redis/go-redis/v9"

	"github.com/redis-developer/redis-vl-golang/extensions/cache"
	"github.com/redis-developer/redis-vl-golang/extensions/history"
	"github.com/redis-developer/redis-vl-golang/extensions/router"
	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/internal/redistest"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// fakeVectorizer produces deterministic one-hot 8-dim embeddings keyed on
// the first word of the text: texts sharing a first word are identical
// (cosine distance 0) and different first words are orthogonal (distance 1).
func fakeVectorizer() vectorize.Vectorizer {
	axes := map[string]int{
		"capital": 0, "zebra": 1, "hello": 2, "goodbye": 3,
		"prompt": 4, "response": 5,
	}
	return &vectorize.Func{
		Model:      "fake",
		Dimensions: 8,
		DataType:   vectors.Float32,
		EmbedFunc: func(_ context.Context, text string) ([]float64, error) {
			word := strings.ToLower(strings.Fields(text + " x")[0])
			axis, ok := axes[word]
			if !ok {
				axis = 6 + len(word)%2
			}
			v := make([]float64, 8)
			v[axis] = 1
			return v, nil
		},
	}
}

func client(t *testing.T) redis.UniversalClient {
	t.Helper()
	url := redistest.URL(t)
	opts, err := redis.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}
	c := redis.NewClient(opts)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestIntegrationEmbeddingsCache(t *testing.T) {
	ctx := context.Background()
	c := cache.NewEmbeddingsCache(client(t), cache.EmbeddingsCacheOptions{Name: "rvlgo-embedcache"})
	t.Cleanup(func() { _ = c.Clear(ctx) })

	key, err := c.Set(ctx, "hello world", "test-model", []float64{0.1, 0.2}, map[string]any{"v": "1"})
	if err != nil {
		t.Fatal(err)
	}
	entry, err := c.Get(ctx, "hello world", "test-model")
	if err != nil || entry == nil {
		t.Fatalf("get: entry=%v err=%v", entry, err)
	}
	if entry.Content != "hello world" || len(entry.Embedding) != 2 || entry.Metadata["v"] != "1" {
		t.Errorf("entry = %+v", entry)
	}
	exists, err := c.Exists(ctx, "hello world", "test-model")
	if err != nil || !exists {
		t.Errorf("exists=%v err=%v", exists, err)
	}
	// batch
	entries, err := c.MGetByKeys(ctx, key, "missing-key")
	if err != nil {
		t.Fatal(err)
	}
	if entries[0] == nil || entries[1] != nil {
		t.Errorf("mget = %v", entries)
	}
	if err := c.Drop(ctx, "hello world", "test-model"); err != nil {
		t.Fatal(err)
	}
	if entry, _ := c.Get(ctx, "hello world", "test-model"); entry != nil {
		t.Error("entry should be dropped")
	}
}

func TestIntegrationSemanticCache(t *testing.T) {
	ctx := context.Background()
	sc, err := cache.NewSemanticCache(ctx, client(t), fakeVectorizer(),
		cache.SemanticCacheOptions{Name: "rvlgo-llmcache", Overwrite: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sc.Delete(ctx) })

	if _, err := sc.Store(ctx, "capital of France", "Paris",
		cache.StoreOptions{Metadata: map[string]any{"topic": "geo"}}); err != nil {
		t.Fatal(err)
	}

	// same first word -> same fake embedding -> hit at distance ~0
	hits, err := sc.Check(ctx, "capital cities quiz")
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Response != "Paris" || hits[0].Metadata["topic"] != "geo" {
		t.Fatalf("hits = %+v", hits)
	}

	// unrelated first word -> miss
	misses, err := sc.Check(ctx, "zebra stripes")
	if err != nil {
		t.Fatal(err)
	}
	if len(misses) != 0 {
		t.Errorf("expected miss, got %+v", misses)
	}

	// update refreshes metadata
	if err := sc.Update(ctx, hits[0].Key, map[string]any{
		"metadata": map[string]any{"hit_count": 1},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestIntegrationMessageHistory(t *testing.T) {
	ctx := context.Background()
	h, err := history.NewMessageHistory(ctx, client(t), "rvlgo-history")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Delete(ctx) })

	for i := 0; i < 3; i++ {
		if err := h.Store(ctx, fmt.Sprintf("prompt %d", i), fmt.Sprintf("response %d", i)); err != nil {
			t.Fatal(err)
		}
	}
	n, err := h.Count(ctx)
	if err != nil || n != 6 {
		t.Fatalf("count=%d err=%v", n, err)
	}
	recent, err := h.GetRecent(ctx, history.GetRecentOptions{TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	// user/llm pairs can share a float timestamp, so assert set membership
	// rather than exact order within the newest exchange
	if len(recent) != 2 {
		t.Fatalf("recent = %+v", recent)
	}
	got := map[string]bool{recent[0].Content: true, recent[1].Content: true}
	if !got["prompt 2"] || !got["response 2"] {
		t.Errorf("recent = %+v", recent)
	}
	// role filter
	userMsgs, err := h.GetRecent(ctx, history.GetRecentOptions{TopK: 10, Roles: []string{"user"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(userMsgs) != 3 {
		t.Errorf("user messages = %+v", userMsgs)
	}
}

func TestIntegrationCachedVectorizer(t *testing.T) {
	ctx := context.Background()
	embedCache := cache.NewEmbeddingsCache(client(t), cache.EmbeddingsCacheOptions{Name: "rvlgo-cachedvec"})
	t.Cleanup(func() { _ = embedCache.Clear(ctx) })

	var calls int
	inner := &vectorize.Func{
		Model:      "counting",
		Dimensions: 8,
		DataType:   vectors.Float32,
		EmbedFunc: func(_ context.Context, text string) ([]float64, error) {
			calls++
			v := make([]float64, 8)
			v[len(text)%8] = 1
			return v, nil
		},
	}
	cv := cache.NewCachedVectorizer(inner, embedCache)

	// first embed: miss -> inner call; second: cache hit -> no inner call
	first, err := cv.Embed(ctx, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	second, err := cv.Embed(ctx, "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("inner calls = %d, want 1", calls)
	}
	if len(first) != len(second) || first[3] != second[3] {
		t.Errorf("cached embedding differs: %v vs %v", first, second)
	}

	// EmbedMany: one cached, two misses -> inner sees only the misses
	calls = 0
	out, err := cv.EmbedMany(ctx, []string{"hello world", "alpha", "beta"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 || out[0] == nil || out[1] == nil || out[2] == nil {
		t.Fatalf("embedMany = %v", out)
	}
	if calls != 2 {
		t.Errorf("inner calls = %d, want 2 (only misses)", calls)
	}

	// everything cached now: zero inner calls
	calls = 0
	if _, err := cv.EmbedMany(ctx, []string{"hello world", "alpha", "beta"}); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Errorf("inner calls = %d, want 0 (all cached)", calls)
	}
}

func TestIntegrationSemanticRouter(t *testing.T) {
	ctx := context.Background()
	routes := []router.Route{
		{Name: "greeting", References: []string{"hello there", "hello friend"}, DistanceThreshold: 0.3},
		{Name: "farewell", References: []string{"goodbye now", "goodbye friend"}, DistanceThreshold: 0.3},
	}
	sr, err := router.NewSemanticRouter(ctx, client(t), "rvlgo-router", routes, fakeVectorizer(),
		router.SemanticRouterOptions{Overwrite: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sr.Delete(ctx) })

	match, err := sr.Route(ctx, "hello everyone")
	if err != nil {
		t.Fatal(err)
	}
	if match.Name != "greeting" {
		t.Errorf("match = %+v", match)
	}

	match, err = sr.Route(ctx, "goodbye everyone")
	if err != nil {
		t.Fatal(err)
	}
	if match.Name != "farewell" {
		t.Errorf("match = %+v", match)
	}

	// unrelated statement should not match any route
	match, err = sr.Route(ctx, "zebra stripes")
	if err != nil {
		t.Fatal(err)
	}
	if match.Name != "" {
		t.Errorf("expected no match, got %+v", match)
	}
}
