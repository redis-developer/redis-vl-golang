// Package query provides builders for Redis Search queries: vector KNN,
// vector range, filter, count, and full-text queries. Port of
// redisvl.query.
package query

import (
	"sort"

	"github.com/redis/redis-vl-golang/filter"
)

// SortBy is a sort specification (single field; Redis Search supports one
// SORTBY field).
type SortBy struct {
	Field string
	Asc   bool
}

// Options are the FT.SEARCH modifiers shared by all query types.
type Options struct {
	ReturnFields []string
	NoContent    bool
	SortBy       *SortBy
	Offset       int
	Num          int
	Dialect      int
	InOrder      bool
	Scorer       string
	WithScores   bool
}

// Query is any search query that can be executed by a SearchIndex.
type Query interface {
	// QueryString is the Redis query-language string (first FT.SEARCH arg
	// after the index name).
	QueryString() string
	// Params are the PARAMS key/value pairs (values may be []byte, string,
	// int, or float64).
	Params() map[string]any
	// Options are the search modifiers.
	Options() *Options
}

// BuildSearchArgs assembles the full FT.SEARCH command arguments for a query
// against the named index.
func BuildSearchArgs(indexName string, q Query) []any {
	o := q.Options()
	args := []any{"FT.SEARCH", indexName, q.QueryString()}
	if o.NoContent {
		args = append(args, "NOCONTENT")
	}
	if o.Scorer != "" {
		args = append(args, "SCORER", o.Scorer)
	}
	if o.WithScores {
		args = append(args, "WITHSCORES")
	}
	if o.InOrder {
		args = append(args, "INORDER")
	}
	if len(o.ReturnFields) > 0 {
		args = append(args, "RETURN", len(o.ReturnFields))
		for _, f := range o.ReturnFields {
			args = append(args, f)
		}
	}
	if o.SortBy != nil {
		dir := "ASC"
		if !o.SortBy.Asc {
			dir = "DESC"
		}
		args = append(args, "SORTBY", o.SortBy.Field, dir)
	}
	num := o.Num
	if num == 0 && !isCount(o) {
		num = 10
	}
	args = append(args, "LIMIT", o.Offset, num)

	params := q.Params()
	if len(params) > 0 {
		args = append(args, "PARAMS", len(params)*2)
		// deterministic ordering for testability
		keys := make([]string, 0, len(params))
		for k := range params {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			args = append(args, k, params[k])
		}
	}
	dialect := o.Dialect
	if dialect == 0 {
		dialect = 2
	}
	args = append(args, "DIALECT", dialect)
	return args
}

func isCount(o *Options) bool {
	return o.NoContent && o.Num == 0 && o.Offset == 0
}

// filterString renders a filter expression, defaulting to "*".
func filterString(f *filter.Expression) string {
	if f == nil {
		return "*"
	}
	return f.String()
}

// DistanceID is the alias used for the returned vector distance field.
const DistanceID = "vector_distance"

// Distance normalization helpers (port of redisvl.utils.utils).

// NormCosineDistance converts a cosine distance (0..2) to a similarity score
// (0..1).
func NormCosineDistance(d float64) float64 { return (2 - d) / 2 }

// DenormCosineDistance converts a similarity score (0..1) back to a cosine
// distance (0..2).
func DenormCosineDistance(s float64) float64 { return 2 - 2*s }

// NormL2Distance converts an unbounded L2 distance to a 0..1 score.
func NormL2Distance(d float64) float64 { return 1 / (1 + d) }
