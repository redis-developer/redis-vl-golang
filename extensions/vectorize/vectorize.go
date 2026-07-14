// Package vectorize defines the Vectorizer interface used by the RedisVL
// extensions (SemanticCache, SemanticMessageHistory, SemanticRouter) to turn
// text into embeddings. It is the Go equivalent of
// redisvl.utils.vectorize.BaseVectorizer.
//
// Provider implementations (OpenAI, Cohere, ...) can be added as separate
// packages; any type satisfying Vectorizer works, including the Func adapter
// for custom embedding functions.
package vectorize

import (
	"context"
	"fmt"

	"github.com/redis/redis-vl-golang/vectors"
)

// Vectorizer converts text into vector embeddings.
type Vectorizer interface {
	// Embed returns the embedding for a single text.
	Embed(ctx context.Context, text string) ([]float64, error)
	// EmbedMany returns embeddings for a batch of texts, in order.
	EmbedMany(ctx context.Context, texts []string) ([][]float64, error)
	// Dims is the embedding dimensionality.
	Dims() int
	// ModelName identifies the underlying embedding model.
	ModelName() string
	// Dtype is the vector datatype used when storing embeddings.
	Dtype() vectors.DataType
}

// Func adapts a plain embedding function into a Vectorizer (equivalent of
// Python's CustomTextVectorizer).
type Func struct {
	// EmbedFunc produces an embedding for one text (required).
	EmbedFunc func(ctx context.Context, text string) ([]float64, error)
	// Model is the reported model name (default "custom").
	Model string
	// Dimensions is the embedding dimensionality (required).
	Dimensions int
	// DataType is the storage datatype (default float32).
	DataType vectors.DataType
}

// Embed implements Vectorizer.
func (f *Func) Embed(ctx context.Context, text string) ([]float64, error) {
	if f.EmbedFunc == nil {
		return nil, fmt.Errorf("vectorize.Func: EmbedFunc is nil")
	}
	return f.EmbedFunc(ctx, text)
}

// EmbedMany implements Vectorizer.
func (f *Func) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, t := range texts {
		v, err := f.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// Dims implements Vectorizer.
func (f *Func) Dims() int { return f.Dimensions }

// ModelName implements Vectorizer.
func (f *Func) ModelName() string {
	if f.Model == "" {
		return "custom"
	}
	return f.Model
}

// Dtype implements Vectorizer.
func (f *Func) Dtype() vectors.DataType {
	if f.DataType == "" {
		return vectors.Float32
	}
	return f.DataType
}
