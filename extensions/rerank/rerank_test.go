package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCohereReranker(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/rerank" {
			t.Errorf("path = %q", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&lastBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{
				{"index": 2, "relevance_score": 0.95},
				{"index": 0, "relevance_score": 0.30},
			},
		})
	}))
	defer srv.Close()

	r, err := NewCohereReranker(CohereConfig{APIKey: "k", BaseURL: srv.URL, Limit: 2})
	if err != nil {
		t.Fatal(err)
	}
	docs := []string{"apple pie", "banana bread", "capital of France"}
	results, err := r.Rank(context.Background(), "France", docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].Document != "capital of France" || results[0].Score != 0.95 {
		t.Errorf("results = %+v", results)
	}
	if results[1].Index != 0 || results[1].Document != "apple pie" {
		t.Errorf("results = %+v", results)
	}
	if lastBody["model"] != "rerank-english-v3.0" || lastBody["top_n"] != float64(2) {
		t.Errorf("request = %v", lastBody)
	}
}

func TestVoyageAIReranker(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("path = %q", r.URL.Path)
		}
		_ = json.NewDecoder(r.Body).Decode(&lastBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 1, "relevance_score": 0.8},
			},
		})
	}))
	defer srv.Close()

	r, err := NewVoyageAIReranker(VoyageAIConfig{APIKey: "k", Model: "rerank-2", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	results, err := r.Rank(context.Background(), "q", []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Document != "b" || results[0].Score != 0.8 {
		t.Errorf("results = %+v", results)
	}
	if lastBody["model"] != "rerank-2" || lastBody["top_k"] != float64(5) {
		t.Errorf("request = %v", lastBody)
	}
}

func TestVoyageAIRequiresModel(t *testing.T) {
	if _, err := NewVoyageAIReranker(VoyageAIConfig{APIKey: "k"}); err == nil {
		t.Error("expected model-required error")
	}
}

func TestRankEmptyDocs(t *testing.T) {
	r, err := NewCohereReranker(CohereConfig{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	results, err := r.Rank(context.Background(), "q", nil)
	if err != nil || results != nil {
		t.Errorf("results=%v err=%v", results, err)
	}
}

func TestOutOfRangeIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{{"index": 9, "relevance_score": 0.5}},
		})
	}))
	defer srv.Close()

	r, _ := NewCohereReranker(CohereConfig{APIKey: "k", BaseURL: srv.URL})
	if _, err := r.Rank(context.Background(), "q", []string{"only one"}); err == nil {
		t.Error("expected out-of-range error")
	}
}
