package vectorize

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/redis/redis-vl-golang/vectors"
)

// VoyageAIConfig configures a VoyageAI vectorizer.
type VoyageAIConfig struct {
	// APIKey defaults to the VOYAGE_API_KEY environment variable.
	APIKey string
	// Model defaults to "voyage-3-large" (Python parity).
	Model string
	// InputType is VoyageAI's optional purpose hint: "document" or
	// "query". Empty omits the hint. Use ForQueries() to derive a
	// query-side vectorizer.
	InputType string
	// BaseURL defaults to "https://api.voyageai.com/v1".
	BaseURL string
	// Dims skips the dimension probe when set.
	Dims int
	// DataType is the storage datatype (default float32).
	DataType vectors.DataType
	// BatchSize for EmbedMany (default 10).
	BatchSize int
	// MaxRetries for API calls (default 3).
	MaxRetries int
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
}

// VoyageAIVectorizer embeds text with the VoyageAI embeddings API (port of
// redisvl.utils.vectorize.VoyageAITextVectorizer).
type VoyageAIVectorizer struct {
	base
	apiKey    string
	baseURL   string
	inputType string
	batchSize int
	http      *httpDoer
}

// NewVoyageAIVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewVoyageAIVectorizer(ctx context.Context, cfg VoyageAIConfig) (*VoyageAIVectorizer, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("VOYAGE_API_KEY")
	}
	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("VoyageAI API key is required: provide VoyageAIConfig.APIKey or set VOYAGE_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "voyage-3-large"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.voyageai.com/v1"
	}
	v := &VoyageAIVectorizer{
		base:      base{model: cfg.Model, dims: cfg.Dims, dtype: cfg.DataType},
		apiKey:    cfg.APIKey,
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		inputType: cfg.InputType,
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

// ForQueries returns a copy of the vectorizer that embeds with
// input_type "query".
func (v *VoyageAIVectorizer) ForQueries() *VoyageAIVectorizer {
	cp := *v
	cp.inputType = "query"
	return &cp
}

type voyageEmbedRequest struct {
	Input     []string `json:"input"`
	Model     string   `json:"model"`
	InputType string   `json:"input_type,omitempty"`
}

func (v *VoyageAIVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp openAIEmbeddingResponse // VoyageAI uses the same response shape
	err := v.http.postJSON(ctx, v.baseURL+"/embeddings",
		map[string]string{"Authorization": "Bearer " + v.apiKey},
		voyageEmbedRequest{Input: texts, Model: v.model, InputType: v.inputType}, &resp)
	if err != nil {
		return nil, fmt.Errorf("voyageai embeddings: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("voyageai embeddings: got %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// Embed implements Vectorizer.
func (v *VoyageAIVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *VoyageAIVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
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
