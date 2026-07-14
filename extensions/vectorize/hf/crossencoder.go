package hf

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"sort"

	"github.com/sugarme/tokenizer"

	"github.com/redis/redis-vl-golang/extensions/rerank"
)

// DefaultCrossEncoderModel matches Python HFCrossEncoderReranker's default.
const DefaultCrossEncoderModel = "cross-encoder/ms-marco-MiniLM-L-6-v2"

// CrossEncoderConfig configures a local cross-encoder reranker.
type CrossEncoderConfig struct {
	// Model is a Hugging Face cross-encoder model id. Defaults to
	// DefaultCrossEncoderModel. The repository must contain a
	// tokenizer.json and an ONNX export.
	Model string
	// ONNXFile is the ONNX export path inside the repository (default
	// "onnx/model.onnx").
	ONNXFile string
	// ONNXRuntimePath is the path to the onnxruntime shared library;
	// defaults to ONNXRUNTIME_LIB_PATH.
	ONNXRuntimePath string
	// CacheDir is where model files are stored (default
	// <user cache dir>/redisvl-go/models).
	CacheDir string
	// Endpoint is the Hugging Face Hub base URL (default HF_ENDPOINT, then
	// https://huggingface.co).
	Endpoint string
	// Token authenticates Hub downloads (default HF_TOKEN).
	Token string
	// Limit is the maximum number of results returned by Rank (default 3,
	// Python parity).
	Limit int
	// MaxSeqLength overrides the model's pair truncation length.
	MaxSeqLength int
	// BatchSize is the number of pairs scored per model run (default 32,
	// matching sentence-transformers CrossEncoder.predict).
	BatchSize int
	// HTTPClient overrides the client used for Hub downloads.
	HTTPClient *http.Client
}

// CrossEncoder reranks documents by scoring (query, document) pairs with a
// local Hugging Face cross-encoder run through ONNX Runtime (port of
// redisvl.utils.rerank.HFCrossEncoderReranker). Scores follow
// sentence-transformers semantics: raw logits or sigmoid, as dictated by
// the model's configuration.
type CrossEncoder struct {
	model     string
	limit     int
	batchSize int
	settings  *modelSettings
	tk        *tokenizer.Tokenizer
	onnx      *onnxModel
}

var _ rerank.Reranker = (*CrossEncoder)(nil)

// NewCrossEncoder downloads the model on first use (cached afterwards),
// loads it into ONNX Runtime, and verifies it with a probe pair.
func NewCrossEncoder(ctx context.Context, cfg CrossEncoderConfig) (*CrossEncoder, error) {
	if cfg.Model == "" {
		cfg.Model = DefaultCrossEncoderModel
	}
	if cfg.ONNXFile == "" {
		cfg.ONNXFile = "onnx/model.onnx"
	}
	if cfg.Limit <= 0 {
		cfg.Limit = 3
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 32
	}

	hub, err := newHubClient(Config{
		CacheDir:   cfg.CacheDir,
		Endpoint:   cfg.Endpoint,
		Token:      cfg.Token,
		HTTPClient: cfg.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	files, err := hub.fetchModel(ctx, cfg.Model, cfg.ONNXFile)
	if err != nil {
		return nil, err
	}

	settings, err := loadModelSettings(files)
	if err != nil {
		return nil, err
	}
	if cfg.MaxSeqLength > 0 {
		settings.maxSeqLength = cfg.MaxSeqLength
	}
	if settings.numLabels > 1 {
		return nil, fmt.Errorf("hf: model %s has %d labels; only single-label cross-encoders are supported", cfg.Model, settings.numLabels)
	}

	tk, err := loadTokenizer(files.tokenizer)
	if err != nil {
		return nil, err
	}

	onnx, err := loadONNXModel(files.onnx, cfg.ONNXRuntimePath, []string{"logits"})
	if err != nil {
		return nil, err
	}

	ce := &CrossEncoder{
		model:     cfg.Model,
		limit:     cfg.Limit,
		batchSize: cfg.BatchSize,
		settings:  settings,
		tk:        tk,
		onnx:      onnx,
	}

	// Probe with one pair to fail fast on a non-classification export.
	if _, err := ce.scorePairs(ctx, "probe", []string{"probe"}); err != nil {
		_ = ce.Close()
		return nil, fmt.Errorf("hf: probing cross-encoder: %w", err)
	}
	return ce, nil
}

// scorePairs returns one relevance score per document.
func (c *CrossEncoder) scorePairs(ctx context.Context, query string, docs []string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	in, err := tokenizePairs(c.tk, query, docs, c.settings.maxSeqLength, c.settings.padTokenID)
	if err != nil {
		return nil, err
	}
	scores, err := c.onnx.runLogits(in)
	if err != nil {
		return nil, err
	}
	if c.settings.sigmoid {
		for i, s := range scores {
			scores[i] = 1 / (1 + math.Exp(-s))
		}
	}
	return scores, nil
}

// Rank implements rerank.Reranker: it scores every (query, doc) pair and
// returns the top Limit documents by descending relevance.
func (c *CrossEncoder) Rank(ctx context.Context, query string, docs []string) ([]rerank.Result, error) {
	if query == "" {
		return nil, fmt.Errorf("hf: query cannot be empty")
	}
	if len(docs) == 0 {
		return []rerank.Result{}, nil
	}

	results := make([]rerank.Result, 0, len(docs))
	for start := 0; start < len(docs); start += c.batchSize {
		end := start + c.batchSize
		if end > len(docs) {
			end = len(docs)
		}
		scores, err := c.scorePairs(ctx, query, docs[start:end])
		if err != nil {
			return nil, err
		}
		for i, score := range scores {
			results = append(results, rerank.Result{
				Index:    start + i,
				Document: docs[start+i],
				Score:    score,
			})
		}
	}

	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > c.limit {
		results = results[:c.limit]
	}
	return results, nil
}

// ModelName identifies the underlying cross-encoder model.
func (c *CrossEncoder) ModelName() string { return c.model }

// Close releases the ONNX Runtime session. The reranker must not be used
// after Close.
func (c *CrossEncoder) Close() error {
	if c.onnx != nil {
		err := c.onnx.close()
		c.onnx = nil
		return err
	}
	return nil
}
