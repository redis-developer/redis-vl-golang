package redisvl

import (
	"encoding/json"
	"fmt"

	"github.com/redis/redis-vl-golang/query"
)

// Document is a single search hit.
type Document struct {
	ID     string
	Score  float64
	Fields map[string]string
}

// SearchResult is the parsed reply of FT.SEARCH.
type SearchResult struct {
	Total int64
	Docs  []Document
}

// parseSearchReply parses an FT.SEARCH reply in either RESP2 (flat array) or
// RESP3 (map) form.
func parseSearchReply(reply any, o *query.Options) (*SearchResult, error) {
	if m, ok := asMap(reply); ok {
		return parseSearchReplyRESP3(m)
	}
	arr, ok := reply.([]any)
	if !ok {
		return nil, fmt.Errorf("unexpected FT.SEARCH reply type: %T", reply)
	}
	return parseSearchReplyRESP2(arr, o)
}

func parseSearchReplyRESP2(arr []any, o *query.Options) (*SearchResult, error) {
	if len(arr) == 0 {
		return nil, fmt.Errorf("empty FT.SEARCH reply")
	}
	total, ok := toInt64(arr[0])
	if !ok {
		return nil, fmt.Errorf("unexpected FT.SEARCH total type: %T", arr[0])
	}
	res := &SearchResult{Total: total}

	i := 1
	for i < len(arr) {
		doc := Document{Fields: map[string]string{}}
		doc.ID = toString(arr[i])
		i++
		if o.WithScores && i < len(arr) {
			if f, ok := toFloat64(arr[i]); ok {
				doc.Score = f
			}
			i++
		}
		if !o.NoContent && i < len(arr) {
			fields, ok := arr[i].([]any)
			if !ok {
				return nil, fmt.Errorf("unexpected FT.SEARCH fields type: %T", arr[i])
			}
			for j := 0; j+1 < len(fields); j += 2 {
				doc.Fields[toString(fields[j])] = toString(fields[j+1])
			}
			i++
		}
		res.Docs = append(res.Docs, doc)
	}
	return res, nil
}

func parseSearchReplyRESP3(m map[string]any) (*SearchResult, error) {
	res := &SearchResult{}
	if t, ok := toInt64(m["total_results"]); ok {
		res.Total = t
	}
	results, _ := m["results"].([]any)
	for _, r := range results {
		rm, ok := asMap(r)
		if !ok {
			continue
		}
		doc := Document{Fields: map[string]string{}}
		doc.ID = toString(rm["id"])
		if f, ok := toFloat64(rm["score"]); ok {
			doc.Score = f
		}
		if attrs, ok := asMap(rm["extra_attributes"]); ok {
			for k, v := range attrs {
				doc.Fields[k] = toString(v)
			}
		}
		res.Docs = append(res.Docs, doc)
	}
	return res, nil
}

// parseInfoReply parses an FT.INFO reply (flat array in RESP2, map in RESP3)
// into a map.
func parseInfoReply(reply any) (map[string]any, error) {
	if m, ok := asMap(reply); ok {
		return m, nil
	}
	arr, ok := reply.([]any)
	if !ok {
		return nil, fmt.Errorf("unexpected FT.INFO reply type: %T", reply)
	}
	out := make(map[string]any, len(arr)/2)
	for i := 0; i+1 < len(arr); i += 2 {
		out[toString(arr[i])] = arr[i+1]
	}
	return out, nil
}

// parseAggregateRows parses an FT.AGGREGATE (or FT.HYBRID) reply in RESP2
// or RESP3 form into row maps. The internal "__score" attribute is dropped,
// mirroring Python's process_aggregate_results.
func parseAggregateRows(reply any) ([]map[string]any, error) {
	var rows []map[string]any

	appendRow := func(fields map[string]any) {
		delete(fields, "__score")
		rows = append(rows, fields)
	}

	switch rep := reply.(type) {
	case []any: // RESP2: [total, [k, v, ...], ...]
		for i := 1; i < len(rep); i++ {
			row, ok := rep[i].([]any)
			if !ok {
				continue
			}
			fields := map[string]any{}
			for j := 0; j+1 < len(row); j += 2 {
				fields[toString(row[j])] = toString(row[j+1])
			}
			appendRow(fields)
		}
	case map[any]any, map[string]any:
		m, _ := asMap(rep)
		results, _ := m["results"].([]any)
		for _, res := range results {
			rm, ok := asMap(res)
			if !ok {
				continue
			}
			if extra, ok := asMap(rm["extra_attributes"]); ok {
				rm = extra
			}
			fields := map[string]any{}
			for k, v := range rm {
				fields[k] = toString(v)
			}
			appendRow(fields)
		}
	default:
		return nil, fmt.Errorf("unexpected aggregate reply type: %T", reply)
	}
	return rows, nil
}

// unpackJSONDoc unpacks the "$" attribute of a JSON-storage document into
// individual fields (mirrors the Python behavior for FilterQuery on JSON
// indices with no explicit return fields).
func unpackJSONDoc(doc Document) map[string]any {
	out := map[string]any{"id": doc.ID}
	raw, ok := doc.Fields["$"]
	if !ok {
		for k, v := range doc.Fields {
			out[k] = v
		}
		return out
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(raw), &data); err == nil {
		for k, v := range data {
			out[k] = v
		}
	} else {
		out["$"] = raw
	}
	return out
}
