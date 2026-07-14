package vectorize

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"github.com/redis/redis-vl-golang/vectors"
)

// httpDoer posts JSON with retries — the Go equivalent of the tenacity
// retry decorators on the Python vectorizers. Retries on network errors,
// 429, and 5xx with exponential backoff and jitter.
type httpDoer struct {
	client     *http.Client
	maxRetries int
}

func newHTTPDoer(client *http.Client, maxRetries int) *httpDoer {
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	if maxRetries <= 0 {
		maxRetries = 3
	}
	return &httpDoer{client: client, maxRetries: maxRetries}
}

// APIError is a non-retryable provider error response.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("provider API error (status %d): %s", e.StatusCode, e.Body)
}

func (h *httpDoer) postJSON(ctx context.Context, url string, headers map[string]string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	var lastErr error
	for attempt := 0; attempt < h.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			backoff += time.Duration(rand.Int63n(int64(backoff) / 2))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			req.Header.Set(k, v)
		}

		resp, err := h.client.Do(req)
		if err != nil {
			lastErr = err
			continue // network error: retry
		}
		respBody, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		switch {
		case resp.StatusCode >= 200 && resp.StatusCode < 300:
			if out == nil {
				return nil
			}
			if err := json.Unmarshal(respBody, out); err != nil {
				return fmt.Errorf("decoding provider response: %w", err)
			}
			return nil
		case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
			lastErr = &APIError{StatusCode: resp.StatusCode, Body: truncate(string(respBody), 300)}
			continue // retryable
		default:
			return &APIError{StatusCode: resp.StatusCode, Body: truncate(string(respBody), 300)}
		}
	}
	return fmt.Errorf("request failed after %d attempts: %w", h.maxRetries, lastErr)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// batchTexts splits texts into batches of at most size (default 10,
// matching the Python vectorizers).
func batchTexts(texts []string, size int) [][]string {
	if size <= 0 {
		size = 10
	}
	var out [][]string
	for start := 0; start < len(texts); start += size {
		end := start + size
		if end > len(texts) {
			end = len(texts)
		}
		out = append(out, texts[start:end])
	}
	return out
}

// base carries the common Vectorizer identity fields.
type base struct {
	model string
	dims  int
	dtype vectors.DataType
}

// Dims implements Vectorizer.
func (b *base) Dims() int { return b.dims }

// ModelName implements Vectorizer.
func (b *base) ModelName() string { return b.model }

// Dtype implements Vectorizer.
func (b *base) Dtype() vectors.DataType {
	if b.dtype == "" {
		return vectors.Float32
	}
	return b.dtype
}

// probeDims determines embedding dimensionality by embedding a test string,
// mirroring the Python vectorizers' "dimension check" probe.
func probeDims(ctx context.Context, v Vectorizer) (int, error) {
	emb, err := v.Embed(ctx, "dimension check")
	if err != nil {
		return 0, fmt.Errorf("failed to determine embedding dimensions: %w", err)
	}
	return len(emb), nil
}
