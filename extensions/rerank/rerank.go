// Package rerank provides result rerankers backed by hosted APIs (Cohere,
// VoyageAI). Port of redisvl.utils.rerank. A local HuggingFace
// cross-encoder reranker implementing the same Reranker interface is
// available in the separate extensions/vectorize/hf module
// (hf.NewCrossEncoder), which runs models in-process via ONNX Runtime.
package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

// Result is a reranked document.
type Result struct {
	// Index is the document's position in the input slice.
	Index int
	// Document is the original document text.
	Document string
	// Score is the relevance score assigned by the reranker.
	Score float64
}

// Reranker reorders documents by relevance to a query.
type Reranker interface {
	// Rank returns documents sorted by descending relevance.
	Rank(ctx context.Context, query string, docs []string) ([]Result, error)
}

// postJSON mirrors the retrying HTTP helper in the vectorize package
// (kept private per package to avoid a shared internal dependency).
func postJSON(ctx context.Context, client *http.Client, maxRetries int, url string, headers map[string]string, payload, out any) error {
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	if maxRetries <= 0 {
		maxRetries = 3
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
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
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		switch {
		case resp.StatusCode >= 200 && resp.StatusCode < 300:
			return json.Unmarshal(respBody, out)
		case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
			lastErr = fmt.Errorf("rerank API error (status %d): %s", resp.StatusCode, respBody)
			continue
		default:
			return fmt.Errorf("rerank API error (status %d): %s", resp.StatusCode, respBody)
		}
	}
	return fmt.Errorf("request failed after %d attempts: %w", maxRetries, lastErr)
}
