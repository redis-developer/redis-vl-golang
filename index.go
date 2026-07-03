// Package redisvl is a Go client library for using Redis as a vector
// database — a port of the core of the Python redisvl library. It provides
// schema-driven index management (SearchIndex), query builders
// (vector KNN, vector range, filter, count, full text), and vector buffer
// utilities.
package redisvl

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/redis-developer/redis-vl-golang/query"
	"github.com/redis-developer/redis-vl-golang/schema"
)

// SearchIndex manages a Redis search index defined by an IndexSchema.
// It is the Go equivalent of both SearchIndex and AsyncSearchIndex in
// Python: every method takes a context and is safe to use from multiple
// goroutines (per go-redis client guarantees).
type SearchIndex struct {
	Schema *schema.IndexSchema

	client     redis.UniversalClient
	ownsClient bool
}

// NewSearchIndex creates a SearchIndex bound to an existing go-redis
// client.
func NewSearchIndex(s *schema.IndexSchema, client redis.UniversalClient) *SearchIndex {
	return &SearchIndex{Schema: s, client: client}
}

// NewSearchIndexFromURL creates a SearchIndex with its own client
// connection from a Redis URL (e.g. "redis://localhost:6379"). Call Close
// to release the connection.
func NewSearchIndexFromURL(s *schema.IndexSchema, redisURL string) (*SearchIndex, error) {
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redis url: %w", err)
	}
	return &SearchIndex{
		Schema:     s,
		client:     redis.NewClient(opts),
		ownsClient: true,
	}, nil
}

// FromExisting builds a SearchIndex for an index that already exists in
// Redis by reading FT.INFO and reconstructing minimal index settings.
// Field definitions are reconstructed on a best-effort basis; prefer
// loading the original schema YAML when available.
func FromExisting(ctx context.Context, name string, client redis.UniversalClient) (*SearchIndex, error) {
	reply, err := client.Do(ctx, "FT.INFO", name).Result()
	if err != nil {
		return nil, fmt.Errorf("index %q does not exist or FT.INFO failed: %w", name, err)
	}
	info, err := parseInfoReply(reply)
	if err != nil {
		return nil, err
	}

	storage := schema.Hash
	prefixes := []string{"rvl"}
	if idef, ok := info["index_definition"]; ok {
		defMap := map[string]any{}
		if m, ok := asMap(idef); ok {
			defMap = m
		} else if arr, ok := idef.([]any); ok {
			for i := 0; i+1 < len(arr); i += 2 {
				defMap[toString(arr[i])] = arr[i+1]
			}
		}
		if kt, ok := defMap["key_type"]; ok && strings.EqualFold(toString(kt), "JSON") {
			storage = schema.JSON
		}
		if p, ok := defMap["prefixes"].([]any); ok && len(p) > 0 {
			prefixes = nil
			for _, v := range p {
				prefixes = append(prefixes, toString(v))
			}
		}
	}

	s, err := schema.NewIndexSchema(schema.IndexInfo{
		Name:        name,
		Prefixes:    prefixes,
		StorageType: storage,
	})
	if err != nil {
		return nil, err
	}
	return NewSearchIndex(s, client), nil
}

// Client returns the underlying go-redis client.
func (i *SearchIndex) Client() redis.UniversalClient { return i.client }

// Close releases the client connection if this index owns it.
func (i *SearchIndex) Close() error {
	if i.ownsClient && i.client != nil {
		return i.client.Close()
	}
	return nil
}

// Name returns the index name.
func (i *SearchIndex) Name() string { return i.Schema.Index.Name }

// Prefix returns the first key prefix.
func (i *SearchIndex) Prefix() string { return i.Schema.Index.Prefixes[0] }

// Key builds the full Redis key for a document id.
func (i *SearchIndex) Key(id string) string {
	return i.Prefix() + i.Schema.Index.KeySeparator + id
}

// CreateOptions control index creation.
type CreateOptions struct {
	// Overwrite an existing index with the same name.
	Overwrite bool
	// Drop all keys associated with the index when overwriting.
	Drop bool
}

// Create creates the search index in Redis from the schema.
func (i *SearchIndex) Create(ctx context.Context, opts ...CreateOptions) error {
	var o CreateOptions
	if len(opts) > 0 {
		o = opts[0]
	}

	schemaArgs, err := i.Schema.SchemaArgs()
	if err != nil {
		return err
	}

	exists, err := i.Exists(ctx)
	if err != nil {
		return err
	}
	if exists {
		if !o.Overwrite {
			return nil // index already exists, not overwriting
		}
		if err := i.Delete(ctx, o.Drop); err != nil {
			return err
		}
	}

	onType := "HASH"
	if i.Schema.Index.StorageType == schema.JSON {
		onType = "JSON"
	}
	args := []any{"FT.CREATE", i.Name(), "ON", onType, "PREFIX", len(i.Schema.Index.Prefixes)}
	for _, p := range i.Schema.Index.Prefixes {
		args = append(args, p)
	}
	if sw := i.Schema.Index.Stopwords; sw != nil {
		args = append(args, "STOPWORDS", len(*sw))
		for _, w := range *sw {
			args = append(args, w)
		}
	}
	args = append(args, "SCHEMA")
	args = append(args, schemaArgs...)

	if err := i.client.Do(ctx, args...).Err(); err != nil {
		return fmt.Errorf("failed to create index %q: %w", i.Name(), err)
	}
	return nil
}

// Delete removes the search index. If drop is true, all associated
// documents are deleted too (FT.DROPINDEX DD).
func (i *SearchIndex) Delete(ctx context.Context, drop bool) error {
	args := []any{"FT.DROPINDEX", i.Name()}
	if drop {
		args = append(args, "DD")
	}
	if err := i.client.Do(ctx, args...).Err(); err != nil {
		if isUnknownIndexError(err) {
			return fmt.Errorf("index %q: %w", i.Name(), ErrIndexNotFound)
		}
		return fmt.Errorf("error while deleting index %q: %w", i.Name(), err)
	}
	return nil
}

// Exists reports whether the index exists in Redis.
func (i *SearchIndex) Exists(ctx context.Context) (bool, error) {
	names, err := i.ListAll(ctx)
	if err != nil {
		return false, err
	}
	for _, n := range names {
		if n == i.Name() {
			return true, nil
		}
	}
	return false, nil
}

// ListAll lists all search index names in the database.
func (i *SearchIndex) ListAll(ctx context.Context) ([]string, error) {
	reply, err := i.client.Do(ctx, "FT._LIST").Result()
	if err != nil {
		return nil, err
	}
	arr, ok := reply.([]any)
	if !ok {
		return nil, fmt.Errorf("unexpected FT._LIST reply type: %T", reply)
	}
	names := make([]string, 0, len(arr))
	for _, v := range arr {
		names = append(names, toString(v))
	}
	return names, nil
}

// Info returns the FT.INFO attributes of the index as a map.
func (i *SearchIndex) Info(ctx context.Context) (map[string]any, error) {
	reply, err := i.client.Do(ctx, "FT.INFO", i.Name()).Result()
	if err != nil {
		if isUnknownIndexError(err) {
			return nil, fmt.Errorf("index %q: %w", i.Name(), ErrIndexNotFound)
		}
		return nil, fmt.Errorf("error while fetching index info: %w", err)
	}
	return parseInfoReply(reply)
}

// LoadOptions control document loading.
type LoadOptions struct {
	// IDField names the document field whose value becomes the key id;
	// when empty a ULID is generated.
	IDField string
	// Keys explicitly sets the full Redis key per document (overrides
	// IDField); must match len(data) when provided.
	Keys []string
	// TTL sets an expiration on each written key.
	TTL time.Duration
	// BatchSize is the pipeline batch size (default 200).
	BatchSize int
	// Preprocess transforms each document before writing.
	Preprocess func(map[string]any) map[string]any
	// Validate checks records against the schema before writing: vector
	// fields must match the declared dimensions and encoding, and numeric
	// fields must hold numeric values. Off by default, mirroring Python's
	// validate_on_load.
	Validate bool
}

// Load writes documents to Redis (HSET for hash storage, JSON.SET for JSON
// storage) and returns the written keys. Vector values for hash storage
// must already be encoded to []byte (see vectors.ToBuffer); for JSON
// storage plain float slices are stored as JSON arrays.
func (i *SearchIndex) Load(ctx context.Context, data []map[string]any, opts ...LoadOptions) ([]string, error) {
	var o LoadOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.Keys != nil && len(o.Keys) != len(data) {
		return nil, fmt.Errorf("length of provided keys (%d) does not match the length of data (%d)", len(o.Keys), len(data))
	}
	batchSize := o.BatchSize
	if batchSize <= 0 {
		batchSize = 200
	}

	keys := make([]string, len(data))
	for idx, obj := range data {
		if o.Keys != nil {
			keys[idx] = o.Keys[idx]
			continue
		}
		if o.IDField != "" {
			v, ok := obj[o.IDField]
			if !ok {
				return nil, fmt.Errorf("id field %q not found in record %d", o.IDField, idx)
			}
			keys[idx] = i.Key(toString(v))
		} else {
			keys[idx] = i.Key(NewULID())
		}
	}

	isJSON := i.Schema.Index.StorageType == schema.JSON

	for start := 0; start < len(data); start += batchSize {
		end := start + batchSize
		if end > len(data) {
			end = len(data)
		}
		pipe := i.client.Pipeline()
		for idx := start; idx < end; idx++ {
			obj := data[idx]
			if o.Preprocess != nil {
				obj = o.Preprocess(obj)
			}
			if o.Validate {
				if err := i.validateRecord(obj); err != nil {
					return nil, fmt.Errorf("record %d: %w", idx, err)
				}
			}
			if isJSON {
				blob, err := json.Marshal(obj)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal record %d: %w", idx, err)
				}
				pipe.Do(ctx, "JSON.SET", keys[idx], "$", string(blob))
			} else {
				flat := make(map[string]any, len(obj))
				for k, v := range obj {
					flat[k] = v
				}
				pipe.HSet(ctx, keys[idx], flat)
			}
			if o.TTL > 0 {
				pipe.Expire(ctx, keys[idx], o.TTL)
			}
		}
		if _, err := pipe.Exec(ctx); err != nil {
			return nil, fmt.Errorf("error loading data: %w", err)
		}
	}
	return keys, nil
}

// Fetch retrieves a document by id (not full key). Returns nil if the
// document does not exist.
func (i *SearchIndex) Fetch(ctx context.Context, id string) (map[string]any, error) {
	key := i.Key(id)
	if i.Schema.Index.StorageType == schema.JSON {
		reply, err := i.client.Do(ctx, "JSON.GET", key, "$").Result()
		if err == redis.Nil {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		var arr []map[string]any
		if err := json.Unmarshal([]byte(toString(reply)), &arr); err != nil || len(arr) == 0 {
			return nil, fmt.Errorf("unable to parse json data from Redis for key %q", key)
		}
		return arr[0], nil
	}

	m, err := i.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	if len(m) == 0 {
		return nil, nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out, nil
}

// DropKeys deletes the given full Redis keys, returning the number removed.
func (i *SearchIndex) DropKeys(ctx context.Context, keys ...string) (int64, error) {
	if len(keys) == 0 {
		return 0, nil
	}
	return i.client.Del(ctx, keys...).Result()
}

// DropDocuments deletes documents by id (prefix applied automatically).
func (i *SearchIndex) DropDocuments(ctx context.Context, ids ...string) (int64, error) {
	keys := make([]string, len(ids))
	for n, id := range ids {
		keys[n] = i.Key(id)
	}
	return i.DropKeys(ctx, keys...)
}

// ExpireKeys sets a TTL on the given full Redis keys.
func (i *SearchIndex) ExpireKeys(ctx context.Context, ttl time.Duration, keys ...string) error {
	pipe := i.client.Pipeline()
	for _, k := range keys {
		pipe.Expire(ctx, k, ttl)
	}
	_, err := pipe.Exec(ctx)
	return err
}

// Clear removes all keys tracked by the index while preserving the index
// itself. Returns the number of deleted records.
func (i *SearchIndex) Clear(ctx context.Context) (int64, error) {
	const pageSize = 500
	var total int64
	opts := &query.Options{NoContent: true}
	offset := 0
	for {
		reply, err := i.client.Do(ctx,
			"FT.SEARCH", i.Name(), "*", "NOCONTENT",
			"LIMIT", offset, pageSize, "DIALECT", 2,
		).Result()
		if err != nil {
			return total, err
		}
		res, err := parseSearchReply(reply, opts)
		if err != nil {
			return total, err
		}
		if len(res.Docs) == 0 {
			break
		}
		keys := make([]string, len(res.Docs))
		for n, d := range res.Docs {
			keys[n] = d.ID
		}
		n, err := i.client.Del(ctx, keys...).Result()
		if err != nil {
			return total, err
		}
		total += n
		if len(res.Docs) < pageSize {
			break
		}
		// keys were deleted, so do not advance the offset
	}
	return total, nil
}

// Search executes a query and returns the raw parsed result.
func (i *SearchIndex) Search(ctx context.Context, q query.Query) (*SearchResult, error) {
	args := query.BuildSearchArgs(i.Name(), q)
	// check after building: Params() may record deferred encode errors
	if err := queryErr(q); err != nil {
		return nil, err
	}
	reply, err := i.client.Do(ctx, args...).Result()
	if err != nil {
		if isUnknownIndexError(err) {
			return nil, fmt.Errorf("index %q: %w", i.Name(), ErrIndexNotFound)
		}
		return nil, fmt.Errorf("error while searching: %w", err)
	}
	return parseSearchReply(reply, q.Options())
}

// Count executes a CountQuery and returns the number of matching records.
func (i *SearchIndex) Count(ctx context.Context, q *query.CountQuery) (int64, error) {
	res, err := i.Search(ctx, q)
	if err != nil {
		return 0, err
	}
	return res.Total, nil
}

// Query executes a query and returns processed documents as maps, applying
// JSON unpacking and optional vector distance normalization (mirrors the
// Python process_results behavior).
func (i *SearchIndex) Query(ctx context.Context, q query.Query) ([]map[string]any, error) {
	res, err := i.Search(ctx, q)
	if err != nil {
		return nil, err
	}
	return i.processResults(res, q), nil
}

func (i *SearchIndex) processResults(res *SearchResult, q query.Query) []map[string]any {
	o := q.Options()

	// distance normalization for vector queries
	var normFn func(float64) float64
	var vectorField string
	switch vq := q.(type) {
	case *query.VectorQuery:
		if vq.NormalizeDistanceEnabled() {
			vectorField = vq.VectorFieldName()
		}
	case *query.VectorRangeQuery:
		if vq.NormalizeDistanceEnabled() {
			vectorField = vq.VectorFieldName()
		}
	}
	if vectorField != "" {
		if f := i.Schema.Field(vectorField); f != nil && f.Vector != nil {
			switch f.Vector.DistanceMetric {
			case schema.Cosine:
				normFn = query.NormCosineDistance
			case schema.L2:
				normFn = query.NormL2Distance
			}
		}
	}

	unpackJSON := i.Schema.Index.StorageType == schema.JSON &&
		len(o.ReturnFields) == 0 && !o.NoContent

	out := make([]map[string]any, 0, len(res.Docs))
	for _, doc := range res.Docs {
		var m map[string]any
		if unpackJSON {
			m = unpackJSONDoc(doc)
		} else {
			m = map[string]any{"id": doc.ID}
			for k, v := range doc.Fields {
				m[k] = v
			}
		}
		if o.WithScores {
			m["score"] = doc.Score
		}
		if normFn != nil {
			if raw, ok := m[query.DistanceID]; ok {
				if d, ok := toFloat64(raw); ok {
					m[query.DistanceID] = toString(normFn(d))
				}
			}
		}
		out = append(out, m)
	}
	return out
}

// Aggregate executes an FT.AGGREGATE query (AggregateHybridQuery,
// MultiVectorQuery, or any AggregationQuery) and returns the result rows.
func (i *SearchIndex) Aggregate(ctx context.Context, q query.AggregationQuery) ([]map[string]any, error) {
	args := q.AggregateArgs(i.Name())
	if err := queryErr(q); err != nil {
		return nil, err
	}
	reply, err := i.client.Do(ctx, args...).Result()
	if err != nil {
		if isUnknownIndexError(err) {
			return nil, fmt.Errorf("index %q: %w", i.Name(), ErrIndexNotFound)
		}
		return nil, fmt.Errorf("error while aggregating: %w", err)
	}
	return parseAggregateRows(reply)
}

// Hybrid executes a native FT.HYBRID query (requires Redis >= 8.4.0) and
// returns the result rows. For older servers use Aggregate with an
// AggregateHybridQuery.
func (i *SearchIndex) Hybrid(ctx context.Context, q *query.HybridQuery) ([]map[string]any, error) {
	args, err := q.HybridArgs(i.Name())
	if err != nil {
		return nil, err
	}
	reply, err := i.client.Do(ctx, args...).Result()
	if err != nil {
		return nil, fmt.Errorf("error while running hybrid search: %w", err)
	}
	return parseAggregateRows(reply)
}

// Paginate executes a query page by page, invoking fn for each page of
// processed documents. Return an error from fn to stop early.
func (i *SearchIndex) Paginate(ctx context.Context, q query.Query, pageSize int, fn func(page []map[string]any) error) error {
	if pageSize <= 0 {
		return fmt.Errorf("page_size must be a positive integer")
	}
	o := q.Options()
	origOffset, origNum := o.Offset, o.Num
	defer func() { o.Offset, o.Num = origOffset, origNum }()

	offset := 0
	for {
		o.Offset, o.Num = offset, pageSize
		docs, err := i.Query(ctx, q)
		if err != nil {
			return err
		}
		if len(docs) == 0 {
			return nil
		}
		if err := fn(docs); err != nil {
			return err
		}
		if len(docs) < pageSize {
			return nil
		}
		offset += pageSize
	}
}
