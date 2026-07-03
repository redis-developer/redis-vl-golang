package vectorize

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/redis-developer/redis-vl-golang/vectors"
)

// OpenAIConfig configures an OpenAI (or OpenAI-compatible) vectorizer.
type OpenAIConfig struct {
	// APIKey defaults to the OPENAI_API_KEY environment variable.
	APIKey string
	// Model defaults to "text-embedding-ada-002" (Python parity).
	Model string
	// BaseURL defaults to "https://api.openai.com/v1"; point it at any
	// OpenAI-compatible endpoint.
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

// OpenAIVectorizer embeds text with the OpenAI embeddings API (port of
// redisvl.utils.vectorize.OpenAITextVectorizer).
type OpenAIVectorizer struct {
	base
	apiKey    string
	baseURL   string
	batchSize int
	http      *httpDoer
}

// NewOpenAIVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewOpenAIVectorizer(ctx context.Context, cfg OpenAIConfig) (*OpenAIVectorizer, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required: provide OpenAIConfig.APIKey or set OPENAI_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-ada-002"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	v := &OpenAIVectorizer{
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

type openAIEmbeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type openAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (v *OpenAIVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp openAIEmbeddingResponse
	err := v.http.postJSON(ctx, v.baseURL+"/embeddings",
		map[string]string{"Authorization": "Bearer " + v.apiKey},
		openAIEmbeddingRequest{Input: texts, Model: v.model}, &resp)
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("openai embeddings: got %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// Embed implements Vectorizer.
func (v *OpenAIVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *OpenAIVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
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

// AzureOpenAIConfig configures an Azure OpenAI vectorizer.
type AzureOpenAIConfig struct {
	// APIKey defaults to the AZURE_OPENAI_API_KEY environment variable.
	APIKey string
	// Endpoint (e.g. https://myresource.openai.azure.com) defaults to
	// AZURE_OPENAI_ENDPOINT.
	Endpoint string
	// APIVersion defaults to OPENAI_API_VERSION.
	APIVersion string
	// Deployment is the Azure deployment name; defaults to Model.
	Deployment string
	// Model defaults to "text-embedding-ada-002".
	Model string
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

// AzureOpenAIVectorizer embeds text with the Azure OpenAI embeddings API
// (port of redisvl.utils.vectorize.AzureOpenAITextVectorizer).
type AzureOpenAIVectorizer struct {
	base
	apiKey    string
	url       string
	batchSize int
	http      *httpDoer
}

// NewAzureOpenAIVectorizer creates the vectorizer and probes the embedding
// dimensions unless cfg.Dims is set.
func NewAzureOpenAIVectorizer(ctx context.Context, cfg AzureOpenAIConfig) (*AzureOpenAIVectorizer, error) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("missing Azure OpenAI API key: provide AzureOpenAIConfig.APIKey or set AZURE_OPENAI_API_KEY")
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("missing Azure OpenAI endpoint: provide AzureOpenAIConfig.Endpoint or set AZURE_OPENAI_ENDPOINT")
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = os.Getenv("OPENAI_API_VERSION")
	}
	if cfg.APIVersion == "" {
		return nil, fmt.Errorf("missing Azure OpenAI API version: provide AzureOpenAIConfig.APIVersion or set OPENAI_API_VERSION")
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-ada-002"
	}
	if cfg.Deployment == "" {
		cfg.Deployment = cfg.Model
	}
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		strings.TrimRight(cfg.Endpoint, "/"), cfg.Deployment, cfg.APIVersion)

	v := &AzureOpenAIVectorizer{
		base:      base{model: cfg.Model, dims: cfg.Dims, dtype: cfg.DataType},
		apiKey:    cfg.APIKey,
		url:       url,
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

func (v *AzureOpenAIVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	var resp openAIEmbeddingResponse
	err := v.http.postJSON(ctx, v.url,
		map[string]string{"api-key": v.apiKey},
		openAIEmbeddingRequest{Input: texts, Model: v.model}, &resp)
	if err != nil {
		return nil, fmt.Errorf("azure openai embeddings: %w", err)
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("azure openai embeddings: got %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// Embed implements Vectorizer.
func (v *AzureOpenAIVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements Vectorizer.
func (v *AzureOpenAIVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
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
