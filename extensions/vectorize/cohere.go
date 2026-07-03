package vectorize

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/redis-developer/redis-vl-golang/vectors"
)

// CohereConfig configures a Cohere vectorizer.
type CohereConfig struct {
	// APIKey defaults to the COHERE_API_KEY environment variable.
	APIKey string
	// Model defaults to "embed-english-v3.0" (Python parity).
	Model string
	// InputType is Cohere's required embedding purpose hint:
	// "search_document" (default; for indexed texts) or "search_query".
	// Use ForQueries() to derive a query-side vectorizer.
	InputType string
	// BaseURL defaults to "https://api.cohere.com".
	BaseURL string
	// Dims skips the dimension probe when set.
	Dims int
	// DataType is the storage datatype (default float32). int8/uint8
	// request the corresponding Cohere embedding types.
	DataType vectors.DataType
	// BatchSize for EmbedMany (default 10).
	BatchSize int
	// MaxRetries for API calls (default 3).
	MaxRetries int
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
}

// CohereVectorizer embeds text with the Cohere embed API (port of
// redisvl.utils.vectorize.CohereTextVectorizer).
type CohereVectorizer struct {
	base
	apiKey    string
	baseURL   string
	inputType string
	embedType string
	batchSize int
	http      *httpDoer
}

// NewCohereVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewCohereVectorizer(ctx context.Context, cfg CohereConfig) (*CohereVectorizer, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("COHERE_API_KEY")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("Cohere API key is required: provide CohereConfig.APIKey or set COHERE_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "embed-english-v3.0"
	}
	if cfg.InputType == "" {
		cfg.InputType = "search_document"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.cohere.com"
	}

	embedType, err := cohereEmbeddingType(cfg.DataType)
	if err != nil {
		return nil, err
	}

	v := &CohereVectorizer{
		base:      base{model: cfg.Model, dims: cfg.Dims, dtype: cfg.DataType},
		apiKey:    cfg.APIKey,
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		inputType: cfg.InputType,
		embedType: embedType,
		batchSize: cfg.BatchSize,
		http:      newHTTPDoer(cfg.HTTPClient, cfg.MaxRetries),
	}
	if v.dims == 0 {
		dims, err := probeDims(ctx, v)
		if err != nil {
			return nil, err
		}
		v.dims = dims
	}
	return v, nil
}

// cohereEmbeddingType maps a vector datatype to Cohere's embedding_types
// value (mirrors Python's dtype mapping).
func cohereEmbeddingType(dt vectors.DataType) (string, error) {
	switch dt {
	case "", vectors.Float32, vectors.Float64, vectors.Float16, vectors.BFloat16:
		return "float", nil
	case vectors.Int8:
		return "int8", nil
	case vectors.Uint8:
		return "uint8", nil
	}
	return "", fmt.Errorf("unsupported datatype for cohere embeddings: %s", dt)
}

// ForQueries returns a copy of the vectorizer that embeds with
// input_type "search_query" (for the query side of asymmetric search).
func (v *CohereVectorizer) ForQueries() *CohereVectorizer {
	cp := *v
	cp.inputType = "search_query"
	return &cp
}

type cohereEmbedRequest struct {
	Model          string   `json:"model"`
	Texts          []string `json:"texts"`
	InputType      string   `json:"input_type"`
	EmbeddingTypes []string `json:"embedding_types"`
}

type cohereEmbedResponse struct {
	Embeddings struct {
		Float [][]float64 `json:"float"`
		Int8  [][]float64 `json:"int8"`
		Uint8 [][]float64 `json:"uint8"`
	} `json:"embeddings"`
}

func (v *CohereVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp cohereEmbedResponse
	err := v.http.postJSON(ctx, v.baseURL+"/v1/embed",
		map[string]string{"Authorization": "Bearer " + v.apiKey},
		cohereEmbedRequest{
			Model: v.model, Texts: texts,
			InputType: v.inputType, EmbeddingTypes: []string{v.embedType},
		}, &resp)
	if err != nil {
		return nil, fmt.Errorf("cohere embeddings: %w", err)
	}
	var embs [][]float64
	switch v.embedType {
	case "int8":
		embs = resp.Embeddings.Int8
	case "uint8":
		embs = resp.Embeddings.Uint8
	default:
		embs = resp.Embeddings.Float
	}
	if len(embs) != len(texts) {
		return nil, fmt.Errorf("cohere embeddings: got %d embeddings for %d inputs", len(embs), len(texts))
	}
	return embs, nil
}

// Embed implements Vectorizer.
func (v *CohereVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *CohereVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
	var out [][]float64
	for _, batch := range batchTexts(texts, v.batchSize) {
		embs, err := v.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, embs...)
	}
	return out, nil
}
