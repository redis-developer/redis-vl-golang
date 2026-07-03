package hf

import (
	"context"
	"fmt"
	"net/http"

	"github.com/sugarme/tokenizer"

	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// DefaultModel matches Python HFTextVectorizer's default model.
const DefaultModel = "sentence-transformers/all-mpnet-base-v2"

// Config configures a local Hugging Face vectorizer.
type Config struct {
	// Model is a Hugging Face model id (e.g.
	// "sentence-transformers/all-MiniLM-L6-v2"). Defaults to DefaultModel.
	// The repository must contain a tokenizer.json and an ONNX export.
	Model string
	// ONNXFile is the ONNX export path inside the repository. Defaults to
	// "onnx/model.onnx"; point it at a quantized variant (e.g.
	// "onnx/model_quint8_avx2.onnx") for smaller downloads and faster CPU
	// inference at a small accuracy cost.
	ONNXFile string
	// ONNXRuntimePath is the path to the onnxruntime shared library
	// (libonnxruntime.so / .dylib / onnxruntime.dll). Defaults to the
	// ONNXRUNTIME_LIB_PATH environment variable, then to the platform's
	// default library name resolved by the dynamic linker.
	ONNXRuntimePath string
	// CacheDir is where model files are stored. Defaults to
	// <user cache dir>/redisvl-go/models.
	CacheDir string
	// Endpoint is the Hugging Face Hub base URL. Defaults to HF_ENDPOINT,
	// then https://huggingface.co.
	Endpoint string
	// Token authenticates Hub downloads for gated or private models.
	// Defaults to HF_TOKEN.
	Token string
	// MaxSeqLength overrides the model's token truncation length.
	MaxSeqLength int
	// BatchSize for EmbedMany (default 10, Python parity).
	BatchSize int
	// DataType is the storage datatype (default float32).
	DataType vectors.DataType
	// HTTPClient overrides the client used for Hub downloads.
	HTTPClient *http.Client
}

// TextVectorizer embeds text locally by running a Hugging Face
// sentence-transformer model through ONNX Runtime (port of
// redisvl.utils.vectorize.HFTextVectorizer). No API key or per-call network
// access is required; model files are downloaded once and cached.
type TextVectorizer struct {
	model     string
	dims      int
	dtype     vectors.DataType
	batchSize int
	settings  *modelSettings
	tk        *tokenizer.Tokenizer
	onnx      *onnxModel
}

var _ vectorize.Vectorizer = (*TextVectorizer)(nil)

// New downloads the model on first use (cached afterwards), loads it into
// ONNX Runtime, and probes the embedding dimensions.
func New(ctx context.Context, cfg Config) (*TextVectorizer, error) {
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if cfg.ONNXFile == "" {
		cfg.ONNXFile = "onnx/model.onnx"
	}
	if cfg.DataType == "" {
		cfg.DataType = vectors.Float32
	}
	dtype, err := vectors.Parse(string(cfg.DataType))
	if err != nil {
		return nil, err
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10
	}

	hub, err := newHubClient(cfg)
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

	tk, err := loadTokenizer(files.tokenizer)
	if err != nil {
		return nil, err
	}

	onnx, err := loadONNXModel(files.onnx, cfg.ONNXRuntimePath, embeddingOutputs)
	if err != nil {
		return nil, err
	}

	v := &TextVectorizer{
		model:     cfg.Model,
		dtype:     dtype,
		batchSize: cfg.BatchSize,
		settings:  settings,
		tk:        tk,
		onnx:      onnx,
	}

	// Probe dimensions by embedding a fixed string (Python parity:
	// HFTextVectorizer._set_model_dims embeds "dimension check").
	probe, err := v.embedBatch(ctx, []string{"dimension check"})
	if err != nil {
		v.Close()
		return nil, fmt.Errorf("hf: probing embedding dimensions: %w", err)
	}
	v.dims = len(probe[0])

	return v, nil
}

func (v *TextVectorizer) embedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	in, err := tokenizeBatch(v.tk, texts, v.settings.maxSeqLength, v.settings.padTokenID)
	if err != nil {
		return nil, err
	}
	return v.onnx.run(in, v.settings)
}

// Embed implements vectorize.Vectorizer.
func (v *TextVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	embs, err := v.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// EmbedMany implements vectorize.Vectorizer.
func (v *TextVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
	var out [][]float64
	for start := 0; start < len(texts); start += v.batchSize {
		end := start + v.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		embs, err := v.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, embs...)
	}
	return out, nil
}

// Dims implements vectorize.Vectorizer.
func (v *TextVectorizer) Dims() int { return v.dims }

// ModelName implements vectorize.Vectorizer.
func (v *TextVectorizer) ModelName() string { return v.model }

// Dtype implements vectorize.Vectorizer.
func (v *TextVectorizer) Dtype() vectors.DataType { return v.dtype }

// Close releases the ONNX Runtime session. The vectorizer must not be used
// after Close.
func (v *TextVectorizer) Close() error {
	if v.onnx != nil {
		err := v.onnx.close()
		v.onnx = nil
		return err
	}
	return nil
}
