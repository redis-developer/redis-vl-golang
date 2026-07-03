package cache

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func newTestLangCache(t *testing.T, srvURL string, opts ...func(*LangCacheConfig)) *LangCache {
	t.Helper()
	cfg := LangCacheConfig{
		ServerURL: srvURL,
		CacheID:   "cache-123",
		APIKey:    "test-key",
	}
	for _, o := range opts {
		o(&cfg)
	}
	lc, err := NewLangCache(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return lc
}

func TestLangCacheStore(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.Method + " " + r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"entryId": "entry-1"})
	}))
	defer srv.Close()

	lc := newTestLangCache(t, srv.URL)
	id, err := lc.Store(context.Background(), "What is semantic caching?", "It caches by meaning.",
		LangCacheStoreOptions{
			Attributes: map[string]string{"topic": "ai"},
			TTL:        90 * time.Second,
		})
	if err != nil {
		t.Fatal(err)
	}
	if id != "entry-1" {
		t.Errorf("entry id = %q", id)
	}
	if gotPath != "POST /v1/caches/cache-123/entries" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("auth = %q", gotAuth)
	}
	if gotBody["prompt"] != "What is semantic caching?" || gotBody["response"] != "It caches by meaning." {
		t.Errorf("body = %v", gotBody)
	}
	if gotBody["ttlMillis"] != float64(90000) {
		t.Errorf("ttlMillis = %v", gotBody["ttlMillis"])
	}
	attrs, _ := gotBody["attributes"].(map[string]any)
	if attrs["topic"] != "ai" {
		t.Errorf("attributes = %v", attrs)
	}
}

func TestLangCacheCheck(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.Method + " " + r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "e1", "prompt": "p1", "response": "r1", "similarity": 0.95,
					"attributes": map[string]string{"topic": "ai"}},
				{"id": "e2", "prompt": "p2", "response": "r2", "similarity": 0.91},
			},
		})
	}))
	defer srv.Close()

	lc := newTestLangCache(t, srv.URL)
	hits, err := lc.Check(context.Background(), "what is caching?", LangCacheCheckOptions{
		NumResults:        2,
		DistanceThreshold: 0.1, // normalized scale -> similarityThreshold 0.9
		Attributes:        map[string]string{"topic": "ai"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "POST /v1/caches/cache-123/entries/search" {
		t.Errorf("path = %q", gotPath)
	}
	if math.Abs(gotBody["similarityThreshold"].(float64)-0.9) > 1e-9 {
		t.Errorf("similarityThreshold = %v", gotBody["similarityThreshold"])
	}
	strategies, _ := gotBody["searchStrategies"].([]any)
	if len(strategies) != 1 || strategies[0] != "semantic" {
		t.Errorf("searchStrategies = %v", strategies)
	}

	if len(hits) != 2 {
		t.Fatalf("hits = %d", len(hits))
	}
	// normalized distance = 1 - similarity
	if math.Abs(hits[0].VectorDistance-0.05) > 1e-9 {
		t.Errorf("distance = %v", hits[0].VectorDistance)
	}
	if hits[0].EntryID != "e1" || hits[0].Response != "r1" || hits[0].Metadata["topic"] != "ai" {
		t.Errorf("hit 0 = %+v", hits[0])
	}
	if hits[1].Metadata != nil {
		t.Errorf("hit 1 metadata should be nil: %+v", hits[1].Metadata)
	}
}

func TestLangCacheCheckRedisScaleAndLimit(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "e1", "similarity": 0.9},
				{"id": "e2", "similarity": 0.8},
				{"id": "e3", "similarity": 0.7},
			},
		})
	}))
	defer srv.Close()

	lc := newTestLangCache(t, srv.URL, func(c *LangCacheConfig) {
		c.DistanceScale = "redis"
		c.UseExactSearch = true
	})
	// redis-scale distance 0.2 -> similarityThreshold (2-0.2)/2 = 0.9
	hits, err := lc.Check(context.Background(), "q", LangCacheCheckOptions{DistanceThreshold: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(gotBody["similarityThreshold"].(float64)-0.9) > 1e-9 {
		t.Errorf("similarityThreshold = %v", gotBody["similarityThreshold"])
	}
	strategies, _ := gotBody["searchStrategies"].([]any)
	if len(strategies) != 2 || strategies[0] != "exact" || strategies[1] != "semantic" {
		t.Errorf("searchStrategies = %v", strategies)
	}
	// NumResults defaults to 1 even when the server returns more
	if len(hits) != 1 {
		t.Fatalf("hits = %d, want 1", len(hits))
	}
	// redis-scale distance = 2 - 2*0.9 = 0.2
	if math.Abs(hits[0].VectorDistance-0.2) > 1e-9 {
		t.Errorf("distance = %v", hits[0].VectorDistance)
	}
}

func TestLangCacheDeleteAndFlush(t *testing.T) {
	var paths []string
	var deleteBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.Method+" "+r.URL.Path)
		if r.Method == http.MethodDelete && r.URL.Path == "/v1/caches/cache-123/entries" {
			_ = json.NewDecoder(r.Body).Decode(&deleteBody)
			_ = json.NewEncoder(w).Encode(map[string]any{"deletedEntriesCount": 7})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	lc := newTestLangCache(t, srv.URL)
	ctx := context.Background()

	if err := lc.DeleteByID(ctx, "entry-9"); err != nil {
		t.Fatal(err)
	}
	n, err := lc.DeleteQuery(ctx, map[string]string{"topic": "ai"})
	if err != nil {
		t.Fatal(err)
	}
	if n != 7 {
		t.Errorf("deleted = %d", n)
	}
	attrs, _ := deleteBody["attributes"].(map[string]any)
	if attrs["topic"] != "ai" {
		t.Errorf("delete attributes = %v", deleteBody)
	}
	if err := lc.Flush(ctx); err != nil {
		t.Fatal(err)
	}

	want := []string{
		"DELETE /v1/caches/cache-123/entries/entry-9",
		"DELETE /v1/caches/cache-123/entries",
		"POST /v1/caches/cache-123/flush",
	}
	if len(paths) != 3 || paths[0] != want[0] || paths[1] != want[1] || paths[2] != want[2] {
		t.Errorf("paths = %v", paths)
	}
}

func TestLangCacheRetryAndErrors(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"entryId": "e"})
	}))
	defer srv.Close()

	lc := newTestLangCache(t, srv.URL)
	if _, err := lc.Store(context.Background(), "p", "r"); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("calls = %d, want 2 (one retry)", calls)
	}

	// 401 is not retried and surfaces as LangCacheError
	var authCalls int32
	authSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&authCalls, 1)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"title":"invalid api key"}`))
	}))
	defer authSrv.Close()

	lc = newTestLangCache(t, authSrv.URL)
	_, err := lc.Store(context.Background(), "p", "r")
	lcErr, ok := err.(*LangCacheError)
	if !ok || lcErr.StatusCode != 401 {
		t.Errorf("err = %v", err)
	}
	if atomic.LoadInt32(&authCalls) != 1 {
		t.Errorf("authCalls = %d, want 1 (no retries on 4xx)", authCalls)
	}
}

func TestLangCacheConfigValidation(t *testing.T) {
	t.Setenv("LANGCACHE_API_KEY", "")
	cases := []LangCacheConfig{
		{CacheID: "c", APIKey: "k"},                                                      // missing server url
		{ServerURL: "https://x", APIKey: "k"},                                            // missing cache id
		{ServerURL: "https://x", CacheID: "c"},                                           // missing api key
		{ServerURL: "https://x", CacheID: "c", APIKey: "k", DistanceScale: "bogus"},      // bad scale
		{ServerURL: "https://x", CacheID: "c", APIKey: "k", DisableSemanticSearch: true}, // no strategies
	}
	for i, cfg := range cases {
		if _, err := NewLangCache(cfg); err == nil {
			t.Errorf("case %d: expected config error", i)
		}
	}
	// exact-only is valid
	if _, err := NewLangCache(LangCacheConfig{
		ServerURL: "https://x", CacheID: "c", APIKey: "k",
		UseExactSearch: true, DisableSemanticSearch: true,
	}); err != nil {
		t.Errorf("exact-only config rejected: %v", err)
	}
}
