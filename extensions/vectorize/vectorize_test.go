package vectorize

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// openAIStyleServer fakes an OpenAI/Mistral/VoyageAI-shaped embeddings
// endpoint returning 3-dim vectors, and records the last request body.
func openAIStyleServer(t *testing.T, lastBody *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("bad request body: %v", err)
		}
		if lastBody != nil {
			*lastBody = body
		}
		inputs, _ := body["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			data[i] = map[string]any{"embedding": []float64{0.1, 0.2, float64(i)}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
}

func TestOpenAIVectorizer(t *testing.T) {
	var lastBody map[string]any
	srv := openAIStyleServer(t, &lastBody)
	defer srv.Close()

	ctx := context.Background()
	v, err := NewOpenAIVectorizer(ctx, OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	// probe should have set dims
	if v.Dims() != 3 {
		t.Errorf("dims = %d", v.Dims())
	}
	if v.ModelName() != "text-embedding-ada-002" {
		t.Errorf("model = %q", v.ModelName())
	}
	emb, err := v.Embed(ctx, "hello")
	if err != nil || len(emb) != 3 {
		t.Fatalf("embed: %v %v", emb, err)
	}
	if lastBody["model"] != "text-embedding-ada-002" {
		t.Errorf("request model = %v", lastBody["model"])
	}

	// batching: 25 texts with default batch size 10 -> per-call inputs <= 10
	texts := make([]string, 25)
	for i := range texts {
		texts[i] = fmt.Sprintf("t%d", i)
	}
	embs, err := v.EmbedMany(ctx, texts)
	if err != nil || len(embs) != 25 {
		t.Fatalf("embed many: %d %v", len(embs), err)
	}
	if n := len(lastBody["input"].([]any)); n != 5 { // 25 = 10+10+5
		t.Errorf("last batch size = %d, want 5", n)
	}
}

func TestAzureOpenAIVectorizer(t *testing.T) {
	var gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotKey = r.Header.Get("api-key")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1, 2}}},
		})
	}))
	defer srv.Close()

	ctx := context.Background()
	v, err := NewAzureOpenAIVectorizer(ctx, AzureOpenAIConfig{
		APIKey: "azkey", Endpoint: srv.URL, APIVersion: "2024-02-01",
		Deployment: "my-deploy",
	})
	if err != nil {
		t.Fatal(err)
	}
	if v.Dims() != 2 {
		t.Errorf("dims = %d", v.Dims())
	}
	want := "/openai/deployments/my-deploy/embeddings?api-version=2024-02-01"
	if gotPath != want {
		t.Errorf("path = %q, want %q", gotPath, want)
	}
	if gotKey != "azkey" {
		t.Errorf("api-key header = %q", gotKey)
	}
}

func TestCohereVectorizer(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&lastBody)
		texts, _ := lastBody["texts"].([]any)
		embs := make([][]float64, len(texts))
		for i := range texts {
			embs[i] = []float64{0.5, 0.6}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": map[string]any{"float": embs},
		})
	}))
	defer srv.Close()

	ctx := context.Background()
	v, err := NewCohereVectorizer(ctx, CohereConfig{APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if v.Dims() != 2 {
		t.Errorf("dims = %d", v.Dims())
	}
	if lastBody["input_type"] != "search_document" {
		t.Errorf("input_type = %v", lastBody["input_type"])
	}
	// query-side copy switches input_type
	if _, err := v.ForQueries().Embed(ctx, "q"); err != nil {
		t.Fatal(err)
	}
	if lastBody["input_type"] != "search_query" {
		t.Errorf("query input_type = %v", lastBody["input_type"])
	}
	et, _ := lastBody["embedding_types"].([]any)
	if len(et) != 1 || et[0] != "float" {
		t.Errorf("embedding_types = %v", et)
	}
}

func TestOllamaVectorizer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %q", r.URL.Path)
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		inputs, _ := body["input"].([]any)
		embs := make([][]float64, len(inputs))
		for i := range inputs {
			embs[i] = []float64{1, 2, 3, 4}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": embs})
	}))
	defer srv.Close()

	ctx := context.Background()
	v, err := NewOllamaVectorizer(ctx, OllamaConfig{Host: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if v.Dims() != 4 || v.ModelName() != "nomic-embed-text" {
		t.Errorf("dims=%d model=%q", v.Dims(), v.ModelName())
	}
}

func TestRetryOn429ThenSuccess(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1}}},
		})
	}))
	defer srv.Close()

	ctx := context.Background()
	v, err := NewOpenAIVectorizer(ctx, OpenAIConfig{APIKey: "k", BaseURL: srv.URL, MaxRetries: 3})
	if err != nil {
		t.Fatal(err)
	}
	if v.Dims() != 1 {
		t.Errorf("dims = %d", v.Dims())
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("calls = %d, want 2 (one retry)", calls)
	}
}

func TestNoRetryOn400(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad model"}`))
	}))
	defer srv.Close()

	ctx := context.Background()
	_, err := NewOpenAIVectorizer(ctx, OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	if err == nil {
		t.Fatal("expected error")
	}
	if atomic.LoadInt32(&calls) != 1 {
		t.Errorf("calls = %d, want 1 (no retries on 4xx)", calls)
	}
}

func TestMissingAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	if _, err := NewOpenAIVectorizer(context.Background(), OpenAIConfig{}); err == nil {
		t.Error("expected missing key error")
	}
}

func TestBatchTexts(t *testing.T) {
	batches := batchTexts([]string{"a", "b", "c"}, 2)
	if len(batches) != 2 || len(batches[0]) != 2 || len(batches[1]) != 1 {
		t.Errorf("batches = %v", batches)
	}
	if got := batchTexts(nil, 10); got != nil {
		t.Errorf("empty input should produce no batches, got %v", got)
	}
}
