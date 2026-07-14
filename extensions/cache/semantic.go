package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis/redis-vl-golang"
	"github.com/redis/redis-vl-golang/extensions/vectorize"
	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/query"
	"github.com/redis/redis-vl-golang/schema"
	"github.com/redis/redis-vl-golang/vectors"
)

// SemanticCache field names (redisvl.extensions.constants).
const (
	promptField      = "prompt"
	responseField    = "response"
	cacheVectorField = "prompt_vector"
	redisKeyField    = "key"
)

// CacheHit is a semantic cache lookup result.
type CacheHit struct {
	Key            string // full Redis key
	EntryID        string
	Prompt         string
	Response       string
	VectorDistance float64
	InsertedAt     float64
	UpdatedAt      float64
	Metadata       map[string]any
	// Filters holds any filterable-field values stored on the entry.
	Filters map[string]string
}

// SemanticCacheOptions configure a SemanticCache.
type SemanticCacheOptions struct {
	// Name of the cache index and key prefix (default "llmcache").
	Name string
	// DistanceThreshold in Redis COSINE units [0-2]; lower is stricter
	// (default 0.1).
	DistanceThreshold float64
	// TTL for cached entries; zero means no expiration.
	TTL time.Duration
	// FilterableFields adds extra schema fields usable in Check filters.
	FilterableFields []schema.Field
	// Overwrite recreates the index schema if it already exists.
	Overwrite bool
}

// SemanticCache caches LLM responses and retrieves them by vector
// similarity of the prompt (port of redisvl.extensions.cache.SemanticCache).
type SemanticCache struct {
	name              string
	distanceThreshold float64
	ttl               time.Duration
	vectorizer        vectorize.Vectorizer
	index             *redisvl.SearchIndex
	client            redis.UniversalClient
	returnFields      []string
}

// NewSemanticCache creates (and if necessary FT.CREATEs) a semantic cache.
func NewSemanticCache(ctx context.Context, client redis.UniversalClient, vectorizer vectorize.Vectorizer, opts ...SemanticCacheOptions) (*SemanticCache, error) {
	var o SemanticCacheOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.Name == "" {
		o.Name = "llmcache"
	}
	if o.DistanceThreshold == 0 {
		o.DistanceThreshold = 0.1
	}
	if o.DistanceThreshold < 0 || o.DistanceThreshold > 2 {
		return nil, fmt.Errorf("distance must be between 0 and 2, got %v", o.DistanceThreshold)
	}
	if vectorizer == nil {
		return nil, fmt.Errorf("a vectorizer is required (no default local model in the Go port)")
	}

	vf, err := schema.NewVectorField(cacheVectorField, schema.VectorAttrs{
		Dims:           vectorizer.Dims(),
		Algorithm:      schema.Flat,
		Datatype:       string(vectorizer.Dtype()),
		DistanceMetric: schema.Cosine,
	})
	if err != nil {
		return nil, err
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: o.Name, Prefixes: []string{o.Name}},
		schema.NewTextField(promptField),
		schema.NewTextField(responseField),
		schema.NewNumericField(insertedAtField),
		schema.NewNumericField(updatedAtField),
		vf,
	)
	if err != nil {
		return nil, err
	}

	returnFields := []string{
		entryIDField, promptField, responseField,
		insertedAtField, updatedAtField, metadataField,
	}
	protected := map[string]bool{redisKeyField: true}
	for _, f := range returnFields {
		protected[f] = true
	}
	for _, f := range o.FilterableFields {
		if protected[f.Name] {
			return nil, fmt.Errorf("%s is a reserved field name for the semantic cache schema", f.Name)
		}
		if err := s.AddField(f); err != nil {
			return nil, err
		}
		returnFields = append(returnFields, f.Name)
	}

	c := &SemanticCache{
		name:              o.Name,
		distanceThreshold: o.DistanceThreshold,
		ttl:               o.TTL,
		vectorizer:        vectorizer,
		index:             redisvl.NewSearchIndex(s, client),
		client:            client,
		returnFields:      returnFields,
	}
	if err := c.index.Create(ctx, redisvl.CreateOptions{Overwrite: o.Overwrite, Drop: false}); err != nil {
		return nil, err
	}
	return c, nil
}

// Index exposes the underlying SearchIndex.
func (c *SemanticCache) Index() *redisvl.SearchIndex { return c.index }

// DistanceThreshold returns the configured semantic distance threshold.
func (c *SemanticCache) DistanceThreshold() float64 { return c.distanceThreshold }

// SetDistanceThreshold updates the semantic distance threshold.
func (c *SemanticCache) SetDistanceThreshold(t float64) error {
	if t < 0 || t > 2 {
		return fmt.Errorf("distance must be between 0 and 2, got %v", t)
	}
	c.distanceThreshold = t
	return nil
}

// TTL returns the default entry TTL.
func (c *SemanticCache) TTL() time.Duration { return c.ttl }

// SetTTL updates the default entry TTL.
func (c *SemanticCache) SetTTL(ttl time.Duration) { c.ttl = ttl }

// StoreOptions customize a Store call.
type StoreOptions struct {
	// Vector is a precomputed prompt embedding (skips the vectorizer).
	Vector []float64
	// Metadata is arbitrary JSON-serializable data stored with the entry.
	Metadata map[string]any
	// Filters are values for the configured filterable fields.
	Filters map[string]any
	// TTL overrides the cache default for this entry.
	TTL time.Duration
}

// Store caches a prompt/response pair and returns the Redis key.
func (c *SemanticCache) Store(ctx context.Context, prompt, response string, opts ...StoreOptions) (string, error) {
	var o StoreOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	vector := o.Vector
	if vector == nil {
		var err error
		vector, err = c.vectorizer.Embed(ctx, prompt)
		if err != nil {
			return "", fmt.Errorf("vectorizing prompt: %w", err)
		}
	}
	if len(vector) != c.vectorizer.Dims() {
		return "", fmt.Errorf("invalid vector dimensions %d, expected %d", len(vector), c.vectorizer.Dims())
	}

	entryID := c.makeEntryID(prompt, o.Filters)
	blob, err := vectors.ToBuffer(vector, c.vectorizer.Dtype())
	if err != nil {
		return "", err
	}
	ts := formatTS(now())
	fields := map[string]any{
		entryIDField:     entryID,
		promptField:      prompt,
		responseField:    response,
		cacheVectorField: blob,
		insertedAtField:  ts,
		updatedAtField:   ts,
	}
	if o.Metadata != nil {
		metaJSON, err := json.Marshal(o.Metadata)
		if err != nil {
			return "", err
		}
		fields[metadataField] = string(metaJSON)
	}
	for k, v := range o.Filters {
		fields[k] = v
	}

	ttl := o.TTL
	if ttl == 0 {
		ttl = c.ttl
	}
	keys, err := c.index.Load(ctx, []map[string]any{fields},
		redisvl.LoadOptions{IDField: entryIDField, TTL: ttl})
	if err != nil {
		return "", err
	}
	return keys[0], nil
}

// CheckOptions customize a Check call.
type CheckOptions struct {
	// Vector is a precomputed query embedding (skips the vectorizer).
	Vector []float64
	// NumResults is the number of hits to return (default 1).
	NumResults int
	// Filter narrows the search to matching entries.
	Filter *filter.Expression
	// DistanceThreshold overrides the cache default. Note: zero means
	// "use the cache default", mirroring the Python behavior.
	DistanceThreshold float64
}

// Check searches the cache for entries semantically similar to the prompt
// and refreshes the TTL of any hits.
func (c *SemanticCache) Check(ctx context.Context, prompt string, opts ...CheckOptions) ([]CacheHit, error) {
	var o CheckOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if prompt == "" && o.Vector == nil {
		return nil, fmt.Errorf("either prompt or vector must be specified")
	}
	vector := o.Vector
	if vector == nil {
		var err error
		vector, err = c.vectorizer.Embed(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("vectorizing prompt: %w", err)
		}
	}
	if len(vector) != c.vectorizer.Dims() {
		return nil, fmt.Errorf("invalid vector dimensions %d, expected %d", len(vector), c.vectorizer.Dims())
	}
	threshold := o.DistanceThreshold
	if threshold == 0 {
		threshold = c.distanceThreshold
	}
	numResults := o.NumResults
	if numResults == 0 {
		numResults = 1
	}

	q := query.NewVectorRangeQuery(cacheVectorField, vector).
		DistanceThreshold(threshold).
		NumResults(numResults).
		Dtype(c.vectorizer.Dtype()).
		ReturnFields(c.returnFields...).
		Filter(o.Filter)

	docs, err := c.index.Query(ctx, q)
	if err != nil {
		return nil, err
	}

	hits := make([]CacheHit, 0, len(docs))
	for _, doc := range docs {
		hit := parseCacheHit(doc)
		hits = append(hits, hit)
		c.expireKey(ctx, hit.Key, 0)
	}
	return hits, nil
}

// Update modifies fields on an existing entry (metadata must be a
// map[string]any) and refreshes its TTL. With no fields, only the
// TTL is refreshed.
func (c *SemanticCache) Update(ctx context.Context, key string, fields map[string]any) error {
	if len(fields) > 0 {
		valid := map[string]bool{metadataField: true}
		for _, n := range c.index.Schema.FieldNames() {
			valid[n] = true
		}
		update := map[string]any{}
		for k, v := range fields {
			if !valid[k] {
				return fmt.Errorf("%s is not a valid field within the cache entry", k)
			}
			if k == metadataField {
				m, ok := v.(map[string]any)
				if !ok {
					return fmt.Errorf("if specified, cached metadata must be a map")
				}
				metaJSON, err := json.Marshal(m)
				if err != nil {
					return err
				}
				update[k] = string(metaJSON)
				continue
			}
			update[k] = v
		}
		update[updatedAtField] = formatTS(now())
		if err := c.client.HSet(ctx, key, update).Err(); err != nil {
			return err
		}
	}
	c.expireKey(ctx, key, 0)
	return nil
}

// Expire sets or refreshes a TTL on an entry key.
func (c *SemanticCache) Expire(ctx context.Context, key string, ttl time.Duration) {
	c.expireKey(ctx, key, ttl)
}

func (c *SemanticCache) expireKey(ctx context.Context, key string, ttl time.Duration) {
	if ttl == 0 {
		ttl = c.ttl
	}
	if ttl > 0 {
		c.client.Expire(ctx, key, ttl)
	}
}

// Drop removes entries by entry ID.
func (c *SemanticCache) Drop(ctx context.Context, ids ...string) error {
	keys := make([]string, len(ids))
	for i, id := range ids {
		keys[i] = c.index.Key(id)
	}
	return c.DropKeys(ctx, keys...)
}

// DropKeys removes entries by full Redis key.
func (c *SemanticCache) DropKeys(ctx context.Context, keys ...string) error {
	_, err := c.index.DropKeys(ctx, keys...)
	return err
}

// Clear removes all cache entries but keeps the index.
func (c *SemanticCache) Clear(ctx context.Context) error {
	_, err := c.index.Clear(ctx)
	return err
}

// Delete removes the cache index and all its entries.
func (c *SemanticCache) Delete(ctx context.Context) error {
	return c.index.Delete(ctx, true)
}

func (c *SemanticCache) makeEntryID(prompt string, filters map[string]any) string {
	return redisvl.Hashify(prompt, filters)
}

func parseCacheHit(doc map[string]any) CacheHit {
	hit := CacheHit{Filters: map[string]string{}}
	known := map[string]bool{
		"id": true, entryIDField: true, promptField: true, responseField: true,
		"vector_distance": true, insertedAtField: true, updatedAtField: true,
		metadataField: true, "score": true,
	}
	for k, v := range doc {
		s := fmt.Sprint(v)
		switch k {
		case "id":
			hit.Key = s
		case entryIDField:
			hit.EntryID = s
		case promptField:
			hit.Prompt = s
		case responseField:
			hit.Response = s
		case "vector_distance":
			hit.VectorDistance, _ = strconv.ParseFloat(s, 64)
		case insertedAtField:
			hit.InsertedAt, _ = strconv.ParseFloat(s, 64)
		case updatedAtField:
			hit.UpdatedAt, _ = strconv.ParseFloat(s, 64)
		case metadataField:
			_ = json.Unmarshal([]byte(s), &hit.Metadata)
		default:
			if !known[k] {
				hit.Filters[k] = s
			}
		}
	}
	return hit
}
