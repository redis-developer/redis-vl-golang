package vectorize

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/redis/redis-vl-golang/vectors"
)

// OllamaConfig configures an Ollama vectorizer.
type OllamaConfig struct {
	// Model defaults to "nomic-embed-text"; it must already be pulled on
	// the Ollama server.
	Model string
	// Host defaults to the OLLAMA_HOST environment variable, then
	// "http://localhost:11434".
	Host string
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

// OllamaVectorizer embeds text with a local Ollama server (port of
// redisvl.utils.vectorize.OllamaTextVectorizer).
type OllamaVectorizer struct {
	base
	host      string
	batchSize int
	http      *httpDoer
}

// NewOllamaVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewOllamaVectorizer(ctx context.Context, cfg OllamaConfig) (*OllamaVectorizer, error) {
	if cfg.Model == "" {
		cfg.Model = "nomic-embed-text"
	}
	if cfg.Host == "" {
		cfg.Host = os.Getenv("OLLAMA_HOST")
	}
	if cfg.Host == "" {
		cfg.Host = "http://localhost:11434"
	}
	v := &OllamaVectorizer{
		base:      base{model: cfg.Model, dims: cfg.Dims, dtype: cfg.DataType},
		host:      strings.TrimRight(cfg.Host, "/"),
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

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (v *OllamaVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp ollamaEmbedResponse
	err := v.http.postJSON(ctx, v.host+"/api/embed", nil,
		ollamaEmbedRequest{Model: v.model, Input: texts}, &resp)
	if err != nil {
		return nil, fmt.Errorf("ollama embeddings: %w", err)
	}
	if len(resp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embeddings: got %d embeddings for %d inputs", len(resp.Embeddings), len(texts))
	}
	return resp.Embeddings, nil
}

// Embed implements Vectorizer.
func (v *OllamaVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *OllamaVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
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
