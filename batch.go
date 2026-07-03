package redisvl

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/redis-developer/redis-vl-golang/query"
	"github.com/redis-developer/redis-vl-golang/schema"
)

// defaultBatchSize matches the Python batch_search default.
const defaultBatchSize = 10

// BatchSearch executes multiple queries with pipelining, returning raw
// parsed results in query order (port of Python's SearchIndex.batch_search).
// The optional batchSize controls how many queries share one network
// round-trip (default 10).
func (i *SearchIndex) BatchSearch(ctx context.Context, queries []query.Query, batchSize ...int) ([]*SearchResult, error) {
	bs := defaultBatchSize
	if len(batchSize) > 0 && batchSize[0] > 0 {
		bs = batchSize[0]
	}

	results := make([]*SearchResult, 0, len(queries))
	for start := 0; start < len(queries); start += bs {
		end := start + bs
		if end > len(queries) {
			end = len(queries)
		}
		batch := queries[start:end]

		pipe := i.client.Pipeline()
		cmds := make([]*redis.Cmd, 0, len(batch))
		for _, q := range batch {
			args := query.BuildSearchArgs(i.Name(), q)
			// check after building: Params() may record deferred encode errors
			if err := queryErr(q); err != nil {
				return nil, err
			}
			cmds = append(cmds, pipe.Do(ctx, args...))
		}
		// Exec returns the first command error; individual results are
		// still inspected below for precise reporting.
		_, execErr := pipe.Exec(ctx)

		for n, cmd := range cmds {
			reply, err := cmd.Result()
			if err != nil {
				if isUnknownIndexError(err) {
					return nil, fmt.Errorf("index %q: %w", i.Name(), ErrIndexNotFound)
				}
				return nil, fmt.Errorf("batch query %d: %w", start+n, err)
			}
			res, err := parseSearchReply(reply, batch[n].Options())
			if err != nil {
				return nil, fmt.Errorf("batch query %d: %w", start+n, err)
			}
			results = append(results, res)
		}
		if execErr != nil && len(results) != end {
			return nil, execErr
		}
	}
	return results, nil
}

// BatchQuery executes multiple queries with pipelining and returns
// processed documents per query, in query order (port of Python's
// SearchIndex.batch_query).
func (i *SearchIndex) BatchQuery(ctx context.Context, queries []query.Query, batchSize ...int) ([][]map[string]any, error) {
	raw, err := i.BatchSearch(ctx, queries, batchSize...)
	if err != nil {
		return nil, err
	}
	out := make([][]map[string]any, len(raw))
	for n, res := range raw {
		out[n] = i.processResults(res, queries[n])
	}
	return out, nil
}

// FetchMany retrieves multiple documents by id in one pipelined
// round-trip. The result slice matches the input order; missing documents
// are nil.
func (i *SearchIndex) FetchMany(ctx context.Context, ids ...string) ([]map[string]any, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	isJSON := i.Schema.Index.StorageType == schema.JSON

	pipe := i.client.Pipeline()
	hashCmds := make([]*redis.MapStringStringCmd, len(ids))
	jsonCmds := make([]*redis.Cmd, len(ids))
	for n, id := range ids {
		if isJSON {
			jsonCmds[n] = pipe.Do(ctx, "JSON.GET", i.Key(id), "$")
		} else {
			hashCmds[n] = pipe.HGetAll(ctx, i.Key(id))
		}
	}
	// Exec error is surfaced per command below (JSON.GET on a missing key
	// returns redis.Nil, which Exec reports but is not a failure here).
	_, _ = pipe.Exec(ctx)

	out := make([]map[string]any, len(ids))
	for n := range ids {
		if isJSON {
			reply, err := jsonCmds[n].Result()
			if err == redis.Nil {
				continue
			}
			if err != nil {
				return nil, fmt.Errorf("fetching %q: %w", ids[n], err)
			}
			var arr []map[string]any
			if jsonErr := json.Unmarshal([]byte(toString(reply)), &arr); jsonErr != nil || len(arr) == 0 {
				continue
			}
			out[n] = arr[0]
			continue
		}
		data, err := hashCmds[n].Result()
		if err != nil && err != redis.Nil {
			return nil, fmt.Errorf("fetching %q: %w", ids[n], err)
		}
		if len(data) == 0 {
			continue
		}
		doc := make(map[string]any, len(data))
		for k, v := range data {
			doc[k] = v
		}
		out[n] = doc
	}
	return out, nil
}
