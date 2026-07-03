package cache

import (
	"context"

	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// CachedVectorizer wraps a Vectorizer with an EmbeddingsCache so repeated
// texts are embedded only once (Go equivalent of passing cache= to a
// Python vectorizer). Cache reads and writes are best-effort: a cache
// failure falls back to the inner vectorizer rather than failing the call.
type CachedVectorizer struct {
	inner vectorize.Vectorizer
	cache *EmbeddingsCache
}

// NewCachedVectorizer wraps inner with an embeddings cache.
func NewCachedVectorizer(inner vectorize.Vectorizer, cache *EmbeddingsCache) *CachedVectorizer {
	return &CachedVectorizer{inner: inner, cache: cache}
}

// Embed implements vectorize.Vectorizer with cache-aside semantics.
func (c *CachedVectorizer) Embed(ctx context.Context, text string) ([]float64, error) {
	if entry, err := c.cache.Get(ctx, text, c.inner.ModelName()); err == nil && entry != nil {
		return entry.Embedding, nil
	}
	emb, err := c.inner.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	_, _ = c.cache.Set(ctx, text, c.inner.ModelName(), emb, nil) // best-effort
	return emb, nil
}

// EmbedMany implements vectorize.Vectorizer: cached texts are served from
// Redis, misses are embedded in one batch and written back.
func (c *CachedVectorizer) EmbedMany(ctx context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))

	// cache lookup (best-effort: treat errors as a full miss)
	var missIdx []int
	entries, err := c.cache.MGet(ctx, c.inner.ModelName(), texts...)
	if err != nil || len(entries) != len(texts) {
		entries = make([]*EmbeddingEntry, len(texts))
	}
	for n, entry := range entries {
		if entry != nil && len(entry.Embedding) > 0 {
			out[n] = entry.Embedding
			continue
		}
		missIdx = append(missIdx, n)
	}
	if len(missIdx) == 0 {
		return out, nil
	}

	missTexts := make([]string, len(missIdx))
	for n, idx := range missIdx {
		missTexts[n] = texts[idx]
	}
	embeddings, err := c.inner.EmbedMany(ctx, missTexts)
	if err != nil {
		return nil, err
	}

	toCache := make(map[string][]float64, len(missIdx))
	for n, idx := range missIdx {
		out[idx] = embeddings[n]
		toCache[missTexts[n]] = embeddings[n]
	}
	_, _ = c.cache.MSet(ctx, c.inner.ModelName(), toCache) // best-effort
	return out, nil
}

// Dims implements vectorize.Vectorizer.
func (c *CachedVectorizer) Dims() int { return c.inner.Dims() }

// ModelName implements vectorize.Vectorizer.
func (c *CachedVectorizer) ModelName() string { return c.inner.ModelName() }

// Dtype implements vectorize.Vectorizer.
func (c *CachedVectorizer) Dtype() vectors.DataType { return c.inner.Dtype() }
