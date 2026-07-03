package vectorize

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/redis-developer/redis-vl-golang/vectors"
)

// MistralConfig configures a Mistral vectorizer.
type MistralConfig struct {
	// APIKey defaults to the MISTRAL_API_KEY environment variable.
	APIKey string
	// Model defaults to "mistral-embed".
	Model string
	// BaseURL defaults to "https://api.mistral.ai/v1".
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

// MistralVectorizer embeds text with the Mistral embeddings API (port of
// redisvl.utils.vectorize.MistralAITextVectorizer).
type MistralVectorizer struct {
	base
	apiKey    string
	baseURL   string
	batchSize int
	http      *httpDoer
}

// NewMistralVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewMistralVectorizer(ctx context.Context, cfg MistralConfig) (*MistralVectorizer, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("MISTRAL_API_KEY")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("missing Mistral API key: provide MistralConfig.APIKey or set MISTRAL_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "mistral-embed"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.mistral.ai/v1"
	}
	v := &MistralVectorizer{
		base:      base{model: cfg.Model, dims: cfg.Dims, dtype: cfg.DataType},
		apiKey:    cfg.APIKey,
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
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

func (v *MistralVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp openAIEmbeddingResponse // Mistral uses the same response shape
	err := v.http.postJSON(ctx, v.baseURL+"/embeddings",
		map[string]string{"Authorization": "Bearer " + v.apiKey},
		openAIEmbeddingRequest{Input: texts, Model: v.model}, &resp)
	if err != nil {
		return nil, fmt.Errorf("mistral embeddings: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("mistral embeddings: got %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// Embed implements Vectorizer.
func (v *MistralVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *MistralVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
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
