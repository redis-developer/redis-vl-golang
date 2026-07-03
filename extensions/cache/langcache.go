package cache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

// LangCache is a client for the Redis LangCache managed semantic caching
// service (port of redisvl.extensions.cache.llm.LangCacheSemanticCache).
// Unlike SemanticCache, embeddings are computed server-side, so no
// vectorizer is required.
//
// The LangCache service is in preview; pin behavior expectations
// accordingly. See https://redis.io/docs/latest/develop/ai/langcache/.
type LangCache struct {
	serverURL     string
	cacheID       string
	apiKey        string
	distanceScale string
	strategies    []string
	ttl           time.Duration
	maxRetries    int
	client        *http.Client
}

// LangCacheConfig configures a LangCache client.
type LangCacheConfig struct {
	// ServerURL is the LangCache API base URL (required), e.g.
	// "https://<host>".
	ServerURL string
	// CacheID identifies the cache (required).
	CacheID string
	// APIKey defaults to the LANGCACHE_API_KEY environment variable.
	APIKey string
	// DistanceScale selects how distance thresholds and returned
	// distances are expressed: "normalized" (0-1, default) or "redis"
	// (COSINE 0-2), mirroring the Python wrapper.
	DistanceScale string
	// UseExactSearch adds the exact (case-insensitive) match strategy.
	UseExactSearch bool
	// DisableSemanticSearch removes the default semantic strategy (only
	// valid together with UseExactSearch).
	DisableSemanticSearch bool
	// TTL is the default time-to-live for stored entries (zero uses the
	// service default).
	TTL time.Duration
	// MaxRetries for API calls (default 3).
	MaxRetries int
	// HTTPClient overrides the default client.
	HTTPClient *http.Client
}

// NewLangCache creates a LangCache client.
func NewLangCache(cfg LangCacheConfig) (*LangCache, error) {
	if cfg.ServerURL == "" {
		return nil, fmt.Errorf("langcache: ServerURL is required")
	}
	if cfg.CacheID == "" {
		return nil, fmt.Errorf("langcache: CacheID is required")
	}
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("LANGCACHE_API_KEY")
	}
	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("langcache: API key is required: provide LangCacheConfig.APIKey or set LANGCACHE_API_KEY")
	}
	switch cfg.DistanceScale {
	case "":
		cfg.DistanceScale = "normalized"
	case "normalized", "redis":
	default:
		return nil, fmt.Errorf("langcache: DistanceScale must be \"normalized\" or \"redis\", got %q", cfg.DistanceScale)
	}

	var strategies []string
	if cfg.UseExactSearch {
		strategies = append(strategies, "exact")
	}
	if !cfg.DisableSemanticSearch {
		strategies = append(strategies, "semantic")
	}
	if len(strategies) == 0 {
		return nil, fmt.Errorf("langcache: at least one search strategy is required (semantic search disabled without exact search)")
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	retries := cfg.MaxRetries
	if retries <= 0 {
		retries = 3
	}
	return &LangCache{
		serverURL:     strings.TrimRight(cfg.ServerURL, "/"),
		cacheID:       cfg.CacheID,
		apiKey:        cfg.APIKey,
		distanceScale: cfg.DistanceScale,
		strategies:    strategies,
		ttl:           cfg.TTL,
		maxRetries:    retries,
		client:        client,
	}, nil
}

func (l *LangCache) url(suffix string) string {
	return l.serverURL + "/v1/caches/" + l.cacheID + suffix
}

// similarityThreshold converts a distance threshold in the configured
// scale to LangCache's similarity threshold in [0,1] (higher is stricter).
func (l *LangCache) similarityThreshold(distance float64) float64 {
	if l.distanceScale == "redis" {
		return (2 - distance) / 2 // norm_cosine_distance
	}
	return 1 - distance
}

// distanceFromSimilarity converts a LangCache similarity in [0,1] back to
// the configured distance scale (lower is better).
func (l *LangCache) distanceFromSimilarity(similarity float64) float64 {
	if l.distanceScale == "redis" {
		return 2 - 2*similarity // denorm_cosine_distance
	}
	return 1 - similarity
}

// LangCacheStoreOptions customize a Store call.
type LangCacheStoreOptions struct {
	// Attributes are stored with the entry and usable as search/delete
	// filters. Attribute names must be pre-configured on the LangCache
	// service.
	Attributes map[string]string
	// TTL overrides the client default for this entry.
	TTL time.Duration
}

type langCacheSetRequest struct {
	Prompt     string            `json:"prompt"`
	Response   string            `json:"response"`
	Attributes map[string]string `json:"attributes,omitempty"`
	TTLMillis  int64             `json:"ttlMillis,omitempty"`
}

type langCacheSetResponse struct {
	EntryID string `json:"entryId"`
	ID      string `json:"id"` // fallback field name
}

// Store caches a prompt/response pair server-side and returns the entry ID.
func (l *LangCache) Store(ctx context.Context, prompt, response string, opts ...LangCacheStoreOptions) (string, error) {
	if prompt == "" {
		return "", fmt.Errorf("langcache: prompt is required")
	}
	if response == "" {
		return "", fmt.Errorf("langcache: response is required")
	}
	var o LangCacheStoreOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	ttl := o.TTL
	if ttl == 0 {
		ttl = l.ttl
	}
	req := langCacheSetRequest{
		Prompt:     prompt,
		Response:   response,
		Attributes: o.Attributes,
	}
	if ttl > 0 {
		req.TTLMillis = ttl.Milliseconds()
	}
	var resp langCacheSetResponse
	if err := l.do(ctx, http.MethodPost, l.url("/entries"), req, &resp); err != nil {
		return "", err
	}
	if resp.EntryID != "" {
		return resp.EntryID, nil
	}
	return resp.ID, nil
}

// LangCacheCheckOptions customize a Check call.
type LangCacheCheckOptions struct {
	// NumResults limits the returned hits (default 1).
	NumResults int
	// DistanceThreshold overrides the service default, expressed in the
	// configured DistanceScale. Zero uses the service default.
	DistanceThreshold float64
	// Attributes scope the search to entries matching all given
	// attributes.
	Attributes map[string]string
}

type langCacheSearchRequest struct {
	Prompt              string            `json:"prompt"`
	SimilarityThreshold float64           `json:"similarityThreshold,omitempty"`
	SearchStrategies    []string          `json:"searchStrategies,omitempty"`
	Attributes          map[string]string `json:"attributes,omitempty"`
}

type langCacheEntry struct {
	ID         string            `json:"id"`
	Prompt     string            `json:"prompt"`
	Response   string            `json:"response"`
	Similarity float64           `json:"similarity"`
	Attributes map[string]string `json:"attributes"`
}

type langCacheSearchResponse struct {
	Data []langCacheEntry `json:"data"`
}

// Check searches the cache for entries semantically similar to the prompt.
// Returned hits carry the entry ID, response, and the distance in the
// configured scale.
func (l *LangCache) Check(ctx context.Context, prompt string, opts ...LangCacheCheckOptions) ([]CacheHit, error) {
	if prompt == "" {
		return nil, fmt.Errorf("langcache: prompt is required")
	}
	var o LangCacheCheckOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	numResults := o.NumResults
	if numResults <= 0 {
		numResults = 1
	}

	req := langCacheSearchRequest{
		Prompt:           prompt,
		SearchStrategies: l.strategies,
		Attributes:       o.Attributes,
	}
	if o.DistanceThreshold > 0 {
		req.SimilarityThreshold = l.similarityThreshold(o.DistanceThreshold)
	}

	var resp langCacheSearchResponse
	if err := l.do(ctx, http.MethodPost, l.url("/entries/search"), req, &resp); err != nil {
		return nil, err
	}

	hits := make([]CacheHit, 0, numResults)
	for _, entry := range resp.Data {
		if len(hits) == numResults {
			break
		}
		var metadata map[string]any
		if len(entry.Attributes) > 0 {
			metadata = make(map[string]any, len(entry.Attributes))
			for k, v := range entry.Attributes {
				metadata[k] = v
			}
		}
		hits = append(hits, CacheHit{
			EntryID:        entry.ID,
			Prompt:         entry.Prompt,
			Response:       entry.Response,
			VectorDistance: l.distanceFromSimilarity(entry.Similarity),
			Metadata:       metadata,
		})
	}
	return hits, nil
}

// DeleteByID removes a single entry by its ID.
func (l *LangCache) DeleteByID(ctx context.Context, entryID string) error {
	if entryID == "" {
		return fmt.Errorf("langcache: entryID is required")
	}
	return l.do(ctx, http.MethodDelete, l.url("/entries/"+entryID), nil, nil)
}

type langCacheDeleteQueryRequest struct {
	Attributes map[string]string `json:"attributes"`
}

type langCacheDeleteQueryResponse struct {
	DeletedEntriesCount int `json:"deletedEntriesCount"`
}

// DeleteQuery removes all entries matching every given attribute and
// returns the number deleted. An empty attributes map deletes ALL entries.
func (l *LangCache) DeleteQuery(ctx context.Context, attributes map[string]string) (int, error) {
	var resp langCacheDeleteQueryResponse
	err := l.do(ctx, http.MethodDelete, l.url("/entries"),
		langCacheDeleteQueryRequest{Attributes: attributes}, &resp)
	if err != nil {
		return 0, err
	}
	return resp.DeletedEntriesCount, nil
}

// Flush removes all entries from the cache. This cannot be undone.
func (l *LangCache) Flush(ctx context.Context) error {
	return l.do(ctx, http.MethodPost, l.url("/flush"), nil, nil)
}

// LangCacheError is a non-retryable LangCache API error (RFC 7807-style
// problem details are preserved in Body).
type LangCacheError struct {
	StatusCode int
	Body       string
}

func (e *LangCacheError) Error() string {
	return fmt.Sprintf("langcache API error (status %d): %s", e.StatusCode, e.Body)
}

// do issues a JSON request with retry on 429/5xx/network errors (matching
// the retry pattern of the vectorize providers).
func (l *LangCache) do(ctx context.Context, method, url string, payload, out any) error {
	var body []byte
	if payload != nil {
		var err error
		body, err = json.Marshal(payload)
		if err != nil {
			return err
		}
	}

	var lastErr error
	for attempt := 0; attempt < l.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			backoff += time.Duration(rand.Int63n(int64(backoff) / 2))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		var reader io.Reader
		if body != nil {
			reader = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, method, url, reader)
		if err != nil {
			return err
		}
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", "Bearer "+l.apiKey)
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}

		resp, err := l.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		respBody, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		switch {
		case resp.StatusCode >= 200 && resp.StatusCode < 300:
			if out == nil || len(respBody) == 0 {
				return nil
			}
			if err := json.Unmarshal(respBody, out); err != nil {
				return fmt.Errorf("langcache: decoding response: %w", err)
			}
			return nil
		case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
			lastErr = &LangCacheError{StatusCode: resp.StatusCode, Body: truncateBody(string(respBody))}
			continue
		default:
			return &LangCacheError{StatusCode: resp.StatusCode, Body: truncateBody(string(respBody))}
		}
	}
	return fmt.Errorf("langcache: request failed after %d attempts: %w", l.maxRetries, lastErr)
}

func truncateBody(s string) string {
	if len(s) <= 300 {
		return s
	}
	return s[:300] + "..."
}
