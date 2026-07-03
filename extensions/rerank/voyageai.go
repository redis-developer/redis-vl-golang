package rerank

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// VoyageAIConfig configures a VoyageAI reranker.
type VoyageAIConfig struct {
	// APIKey defaults to the VOYAGE_API_KEY environment variable.
	APIKey string
	// Model is required (e.g. "rerank-2"); the Python port also requires
	// it explicitly.
	Model string
	// Limit is the maximum number of results (default 5).
	Limit int
	// BaseURL defaults to "https://api.voyageai.com/v1".
	BaseURL string
	// MaxRetries for API calls (default 3).
	MaxRetries int
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
}

// VoyageAIReranker reranks documents with the VoyageAI rerank API (port of
// redisvl.utils.rerank.VoyageAIReranker).
type VoyageAIReranker struct {
	apiKey     string
	model      string
	limit      int
	baseURL    string
	maxRetries int
	client     *http.Client
}

// NewVoyageAIReranker creates a VoyageAI reranker.
func NewVoyageAIReranker(cfg VoyageAIConfig) (*VoyageAIReranker, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("VOYAGE_API_KEY")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("VoyageAI API key is required: provide VoyageAIConfig.APIKey or set VOYAGE_API_KEY")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("VoyageAI reranker model is required (e.g. \"rerank-2\")")
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 5
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.voyageai.com/v1"
	}
	return &VoyageAIReranker{
		apiKey:     cfg.APIKey,
		model:      cfg.Model,
		limit:      cfg.Limit,
		baseURL:    strings.TrimRight(cfg.BaseURL, "/"),
		maxRetries: cfg.MaxRetries,
		client:     cfg.HTTPClient,
	}, nil
}

type voyageRerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	Model     string   `json:"model"`
	TopK      int      `json:"top_k"`
}

type voyageRerankResponse struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"data"`
}

// Rank implements Reranker.
func (r *VoyageAIReranker) Rank(ctx context.Context, query string, docs []string) ([]Result, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	var resp voyageRerankResponse
	err := postJSON(ctx, r.client, r.maxRetries, r.baseURL+"/rerank",
		map[string]string{"Authorization": "Bearer " + r.apiKey},
		voyageRerankRequest{Query: query, Documents: docs, Model: r.model, TopK: r.limit},
		&resp)
	if err != nil {
		return nil, fmt.Errorf("voyageai rerank: %w", err)
	}
	out := make([]Result, 0, len(resp.Data))
	for _, res := range resp.Data {
		if res.Index < 0 || res.Index >= len(docs) {
			return nil, fmt.Errorf("voyageai rerank: result index %d out of range", res.Index)
		}
		out = append(out, Result{Index: res.Index, Document: docs[res.Index], Score: res.RelevanceScore})
	}
	return out, nil
}
