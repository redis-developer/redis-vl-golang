package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	redisvl "github.com/redis/redis-vl-golang"
	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/query"
	"github.com/redis/redis-vl-golang/schema"
	"github.com/redis/redis-vl-golang/vectors"
)

// SearchArgs are the arguments of the search-records tool (mirrors the
// Python tool contract).
type SearchArgs struct {
	Query  string `json:"query" jsonschema:"the search query text"`
	Limit  int    `json:"limit,omitempty" jsonschema:"maximum number of results to return"`
	Offset int    `json:"offset,omitempty" jsonschema:"number of results to skip for pagination"`
	Filter string `json:"filter,omitempty" jsonschema:"optional Redis query filter expression, e.g. @genre:{scifi}"`
}

// UpsertArgs are the arguments of the upsert-records tool.
type UpsertArgs struct {
	Records []map[string]any `json:"records" jsonschema:"records to insert or update"`
	IDField string           `json:"id_field,omitempty" jsonschema:"record field whose value becomes the document id"`
}

func (s *Server) searchTool(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, any, error) {
	if err := s.requireScope(ctx, s.cfg.Server.Auth.readScope()); err != nil {
		return nil, nil, err
	}
	records, err := s.doSearch(ctx, args)
	if err != nil {
		return nil, nil, err
	}
	return jsonResult(map[string]any{
		"count":   len(records),
		"records": records,
	})
}

func (s *Server) upsertTool(ctx context.Context, req *mcp.CallToolRequest, args UpsertArgs) (*mcp.CallToolResult, any, error) {
	if err := s.requireScope(ctx, s.cfg.Server.Auth.writeScope()); err != nil {
		return nil, nil, err
	}
	keys, err := s.doUpsert(ctx, args)
	if err != nil {
		return nil, nil, err
	}
	return jsonResult(map[string]any{
		"upserted": len(keys),
		"keys":     keys,
	})
}

func jsonResult(v any) (*mcp.CallToolResult, any, error) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, nil, err
	}
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: string(data)}},
	}, nil, nil
}

// doSearch runs a semantic search when a vectorizer is configured,
// otherwise a full-text search over the configured text field.
func (s *Server) doSearch(ctx context.Context, args SearchArgs) ([]map[string]any, error) {
	if args.Query == "" {
		return nil, fmt.Errorf("query must not be empty")
	}
	limit := s.cfg.Runtime.clampLimit(args.Limit)
	offset := args.Offset
	if offset < 0 {
		offset = 0
	}

	var fe *filter.Expression
	if args.Filter != "" {
		fe = filter.Raw(args.Filter)
	}

	var docs []map[string]any
	if s.vectorizer != nil && s.cfg.Runtime.VectorFieldName != "" {
		emb, err := s.vectorizer.Embed(ctx, args.Query)
		if err != nil {
			return nil, fmt.Errorf("embedding query: %w", err)
		}
		q := query.NewVectorQuery(s.cfg.Runtime.VectorFieldName, emb).
			Dtype(s.dtype).
			Paging(offset, limit).
			Filter(fe)
		if len(s.cfg.Runtime.ReturnFields) > 0 {
			q.ReturnFields(s.cfg.Runtime.ReturnFields...)
		}
		docs, err = s.index.Query(ctx, q)
		if err != nil {
			return nil, err
		}
	} else {
		q := query.NewTextQuery(args.Query, s.cfg.Runtime.TextFieldName).
			NumResults(limit).
			Filter(fe)
		if len(s.cfg.Runtime.ReturnFields) > 0 {
			q.ReturnFields(s.cfg.Runtime.ReturnFields...)
		}
		q.Options().Offset = offset
		var err error
		docs, err = s.index.Query(ctx, q)
		if err != nil {
			return nil, err
		}
	}
	return s.sanitizeRecords(docs), nil
}

// sanitizeRecords removes raw vector payloads (binary noise in JSON
// output) from result rows.
func (s *Server) sanitizeRecords(docs []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(docs))
	for _, doc := range docs {
		clean := make(map[string]any, len(doc))
		for k, v := range doc {
			if k == s.cfg.Runtime.VectorFieldName {
				continue
			}
			clean[k] = v
		}
		out = append(out, clean)
	}
	return out
}

// doUpsert embeds records as needed and loads them into the index.
func (s *Server) doUpsert(ctx context.Context, args UpsertArgs) ([]string, error) {
	if s.readOnly {
		return nil, fmt.Errorf("server is read-only; upsert is disabled")
	}
	if len(args.Records) == 0 {
		return nil, fmt.Errorf("records must not be empty")
	}
	if len(args.Records) > s.cfg.Runtime.MaxUpsertRecords {
		return nil, fmt.Errorf("too many records: %d (max %d)", len(args.Records), s.cfg.Runtime.MaxUpsertRecords)
	}

	records, err := s.embedRecords(ctx, args.Records)
	if err != nil {
		return nil, err
	}
	return s.index.Load(ctx, records, redisvl.LoadOptions{IDField: args.IDField})
}

// embedRecords fills the vector field of records from the configured embed
// text field, honoring skip_embedding_if_present.
func (s *Server) embedRecords(ctx context.Context, records []map[string]any) ([]map[string]any, error) {
	rt := s.cfg.Runtime
	if s.vectorizer == nil || rt.VectorFieldName == "" || rt.DefaultEmbedTextField == "" {
		return records, nil
	}

	// collect texts that need embedding
	var texts []string
	var targets []int
	for i, rec := range records {
		if *rt.SkipEmbeddingIfPresent {
			if _, ok := rec[rt.VectorFieldName]; ok {
				continue
			}
		}
		raw, ok := rec[rt.DefaultEmbedTextField]
		if !ok {
			continue
		}
		text, ok := raw.(string)
		if !ok || text == "" {
			continue
		}
		texts = append(texts, text)
		targets = append(targets, i)
	}
	if len(texts) == 0 {
		return records, nil
	}

	embeddings, err := s.vectorizer.EmbedMany(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embedding records: %w", err)
	}
	if len(embeddings) != len(texts) {
		return nil, fmt.Errorf("vectorizer returned %d embeddings for %d texts", len(embeddings), len(texts))
	}

	isJSON := s.index.Schema.Index.StorageType == schema.JSON
	for n, i := range targets {
		if isJSON {
			records[i][rt.VectorFieldName] = embeddings[n]
			continue
		}
		blob, err := vectors.ToBuffer(embeddings[n], s.dtype)
		if err != nil {
			return nil, err
		}
		records[i][rt.VectorFieldName] = blob
	}
	return records, nil
}
