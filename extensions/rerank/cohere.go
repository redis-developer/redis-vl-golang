package rerank

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// CohereConfig configures a Cohere reranker.
type CohereConfig struct {
	// APIKey defaults to the COHERE_API_KEY environment variable.
	APIKey string
	// Model defaults to "rerank-english-v3.0" (Python parity).
	Model string
	// Limit is the maximum number of results (default 5).
	Limit int
	// BaseURL defaults to "https://api.cohere.com".
	BaseURL string
	// MaxRetries for API calls (default 3).
	MaxRetries int
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
}

// CohereReranker reranks documents with the Cohere rerank API (port of
// redisvl.utils.rerank.CohereReranker).
type CohereReranker struct {
	apiKey     string
	model      string
	limit      int
	baseURL    string
	maxRetries int
	client     *http.Client
}

// NewCohereReranker creates a Cohere reranker.
func NewCohereReranker(cfg CohereConfig) (*CohereReranker, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("COHERE_API_KEY")
	}
	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("missing Cohere API key: provide CohereConfig.APIKey or set COHERE_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "rerank-english-v3.0"
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 5
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.cohere.com"
	}
	return &CohereReranker{
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		limit:      cfg.Limit,
		baseURL:    strings.TrimRight(cfg.BaseURL, "/"),
		maxRetries: cfg.MaxRetries,
		client:     cfg.HTTPClient,
	}, nil
}

type cohereRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type cohereRerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rank implements Reranker.
func (r *CohereReranker) Rank(ctx context.Context, query string, docs []string) ([]Result, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	var resp cohereRerankResponse
	err := postJSON(ctx, r.client, r.maxRetries, r.baseURL+"/v2/rerank",
		map[string]string{"Authorization": "Bearer " + r.apiKey},
		cohereRerankRequest{Model: r.model, Query: query, Documents: docs, TopN: r.limit},
		&resp)
	if err != nil {
		return nil, fmt.Errorf("cohere rerank: %w", err)
	}
	out := make([]Result, 0, len(resp.Results))
	for _, res := range resp.Results {
		if res.Index < 0 || res.Index >= len(docs) {
			return nil, fmt.Errorf("cohere rerank: result index %d out of range", res.Index)
		}
		out = append(out, Result{Index: res.Index, Document: docs[res.Index], Score: res.RelevanceScore})
	}
	return out, nil
}
