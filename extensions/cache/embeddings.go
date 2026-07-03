// Package cache provides Redis-backed caches for AI workloads: an
// EmbeddingsCache with exact key matching and a SemanticCache for LLM
// responses with vector-similarity lookup. Port of
// redisvl.extensions.cache.
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis-developer/redis-vl-golang"
)

// Field names shared with the Python implementation
// (redisvl.extensions.constants).
const (
	entryIDField    = "entry_id"
	textContent     = "content"
	modelNameField  = "model_name"
	embeddingField  = "embedding"
	insertedAtField = "inserted_at"
	updatedAtField  = "updated_at"
	metadataField   = "metadata"
)

func now() float64 { return float64(time.Now().UnixNano()) / 1e9 }

func formatTS(ts float64) string { return strconv.FormatFloat(ts, 'f', -1, 64) }

// EmbeddingEntry is a cached embedding record.
type EmbeddingEntry struct {
	EntryID    string
	Content    string
	ModelName  string
	Embedding  []float64
	InsertedAt float64
	Metadata   map[string]any
}

// EmbeddingsCache stores embedding vectors keyed exactly by content and
// model name (port of redisvl.extensions.cache.EmbeddingsCache).
type EmbeddingsCache struct {
	name   string
	ttl    time.Duration
	client redis.UniversalClient
}

// EmbeddingsCacheOptions configure an EmbeddingsCache.
type EmbeddingsCacheOptions struct {
	// Name of the cache and key prefix (default "embedcache").
	Name string
	// TTL for cached entries; zero means no expiration.
	TTL time.Duration
}

// NewEmbeddingsCache creates an embeddings cache on the given client.
func NewEmbeddingsCache(client redis.UniversalClient, opts ...EmbeddingsCacheOptions) *EmbeddingsCache {
	var o EmbeddingsCacheOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.Name == "" {
		o.Name = "embedcache"
	}
	return &EmbeddingsCache{name: o.Name, ttl: o.TTL, client: client}
}

// TTL returns the default entry TTL.
func (c *EmbeddingsCache) TTL() time.Duration { return c.ttl }

// SetTTL updates the default entry TTL.
func (c *EmbeddingsCache) SetTTL(ttl time.Duration) { c.ttl = ttl }

func (c *EmbeddingsCache) entryID(content, modelName string) string {
	return redisvl.Hashify(content+":"+modelName, nil)
}

// Key returns the full Redis key for the given content and model name.
func (c *EmbeddingsCache) Key(content, modelName string) string {
	return c.name + ":" + c.entryID(content, modelName)
}

func (c *EmbeddingsCache) expire(ctx context.Context, key string, ttl time.Duration) {
	if ttl == 0 {
		ttl = c.ttl
	}
	if ttl > 0 {
		c.client.Expire(ctx, key, ttl)
	}
}

// Set stores an embedding and returns its Redis key.
func (c *EmbeddingsCache) Set(ctx context.Context, content, modelName string, embedding []float64, metadata map[string]any) (string, error) {
	return c.SetWithTTL(ctx, content, modelName, embedding, metadata, 0)
}

// SetWithTTL stores an embedding with a per-entry TTL override.
func (c *EmbeddingsCache) SetWithTTL(ctx context.Context, content, modelName string, embedding []float64, metadata map[string]any, ttl time.Duration) (string, error) {
	key := c.Key(content, modelName)
	embJSON, err := json.Marshal(embedding)
	if err != nil {
		return "", err
	}
	fields := map[string]any{
		entryIDField:    c.entryID(content, modelName),
		textContent:     content,
		modelNameField:  modelName,
		embeddingField:  string(embJSON),
		insertedAtField: formatTS(now()),
	}
	if metadata != nil {
		metaJSON, err := json.Marshal(metadata)
		if err != nil {
			return "", err
		}
		fields[metadataField] = string(metaJSON)
	}
	if err := c.client.HSet(ctx, key, fields).Err(); err != nil {
		return "", fmt.Errorf("embeddings cache set: %w", err)
	}
	c.expire(ctx, key, ttl)
	return key, nil
}

// MSet stores multiple embeddings in one pipeline; items are (content,
// embedding) pairs for a single model. Returns the keys in order.
func (c *EmbeddingsCache) MSet(ctx context.Context, modelName string, items map[string][]float64) ([]string, error) {
	keys := make([]string, 0, len(items))
	pipe := c.client.Pipeline()
	for content, emb := range items {
		key := c.Key(content, modelName)
		keys = append(keys, key)
		embJSON, err := json.Marshal(emb)
		if err != nil {
			return nil, err
		}
		pipe.HSet(ctx, key, map[string]any{
			entryIDField:    c.entryID(content, modelName),
			textContent:     content,
			modelNameField:  modelName,
			embeddingField:  string(embJSON),
			insertedAtField: formatTS(now()),
		})
		if c.ttl > 0 {
			pipe.Expire(ctx, key, c.ttl)
		}
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("embeddings cache mset: %w", err)
	}
	return keys, nil
}

// Get retrieves a cached embedding by content and model name, refreshing
// its TTL. Returns nil if not found.
func (c *EmbeddingsCache) Get(ctx context.Context, content, modelName string) (*EmbeddingEntry, error) {
	return c.GetByKey(ctx, c.Key(content, modelName))
}

// GetByKey retrieves a cached embedding by full Redis key.
func (c *EmbeddingsCache) GetByKey(ctx context.Context, key string) (*EmbeddingEntry, error) {
	data, err := c.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	c.expire(ctx, key, 0)
	return parseEmbeddingEntry(data)
}

// MGet retrieves multiple embeddings by content for one model. The result
// slice matches the input order; misses are nil.
func (c *EmbeddingsCache) MGet(ctx context.Context, modelName string, contents ...string) ([]*EmbeddingEntry, error) {
	keys := make([]string, len(contents))
	for i, content := range contents {
		keys[i] = c.Key(content, modelName)
	}
	return c.MGetByKeys(ctx, keys...)
}

// MGetByKeys retrieves multiple embeddings by full Redis keys.
func (c *EmbeddingsCache) MGetByKeys(ctx context.Context, keys ...string) ([]*EmbeddingEntry, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	pipe := c.client.Pipeline()
	cmds := make([]*redis.MapStringStringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.HGetAll(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("embeddings cache mget: %w", err)
	}
	out := make([]*EmbeddingEntry, len(keys))
	for i, cmd := range cmds {
		data, err := cmd.Result()
		if err != nil || len(data) == 0 {
			continue
		}
		c.expire(ctx, keys[i], 0)
		entry, err := parseEmbeddingEntry(data)
		if err != nil {
			return nil, err
		}
		out[i] = entry
	}
	return out, nil
}

// Exists reports whether an embedding is cached for the content/model pair.
func (c *EmbeddingsCache) Exists(ctx context.Context, content, modelName string) (bool, error) {
	n, err := c.client.Exists(ctx, c.Key(content, modelName)).Result()
	return n > 0, err
}

// Drop removes an entry by content and model name.
func (c *EmbeddingsCache) Drop(ctx context.Context, content, modelName string) error {
	return c.client.Del(ctx, c.Key(content, modelName)).Err()
}

// DropByKeys removes entries by full Redis keys.
func (c *EmbeddingsCache) DropByKeys(ctx context.Context, keys ...string) error {
	if len(keys) == 0 {
		return nil
	}
	return c.client.Del(ctx, keys...).Err()
}

// Clear removes all entries under this cache's prefix.
func (c *EmbeddingsCache) Clear(ctx context.Context) error {
	var cursor uint64
	for {
		keys, next, err := c.client.Scan(ctx, cursor, c.name+":*", 500).Result()
		if err != nil {
			return err
		}
		if len(keys) > 0 {
			if err := c.client.Del(ctx, keys...).Err(); err != nil {
				return err
			}
		}
		if next == 0 {
			return nil
		}
		cursor = next
	}
}

func parseEmbeddingEntry(data map[string]string) (*EmbeddingEntry, error) {
	entry := &EmbeddingEntry{
		EntryID:   data[entryIDField],
		Content:   data[textContent],
		ModelName: data[modelNameField],
	}
	if raw, ok := data[embeddingField]; ok {
		if err := json.Unmarshal([]byte(raw), &entry.Embedding); err != nil {
			return nil, fmt.Errorf("invalid cached embedding: %w", err)
		}
	}
	if raw, ok := data[insertedAtField]; ok {
		entry.InsertedAt, _ = strconv.ParseFloat(raw, 64)
	}
	if raw, ok := data[metadataField]; ok {
		if err := json.Unmarshal([]byte(raw), &entry.Metadata); err != nil {
			return nil, fmt.Errorf("invalid cached metadata: %w", err)
		}
	}
	return entry, nil
}
