package query

import (
	"fmt"
	"strings"

	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/vectors"
)

// VectorSearchMethod selects the vector arm of an FT.HYBRID query.
type VectorSearchMethod string

const (
	// KNN performs k-nearest-neighbor vector search.
	KNN VectorSearchMethod = "KNN"
	// Range performs radius-based vector search.
	Range VectorSearchMethod = "RANGE"
)

// CombinationMethod selects how FT.HYBRID fuses text and vector scores.
type CombinationMethod string

const (
	// RRF is reciprocal rank fusion.
	RRF CombinationMethod = "RRF"
	// Linear combines scores as alpha*text + beta*vector.
	Linear CombinationMethod = "LINEAR"
)

// HybridQuery is a native FT.HYBRID query combining full-text search and
// vector similarity with server-side score fusion (port of
// redisvl.query.HybridQuery).
//
// Requires Redis >= 8.4.0. For older servers use AggregateHybridQuery,
// which achieves similar scoring via FT.AGGREGATE.
type HybridQuery struct {
	text        string
	textField   string
	vector      []float64
	vectorBytes []byte
	vectorField string
	dtype       vectors.DataType

	textScorer       string
	yieldTextScoreAs string

	searchMethod     VectorSearchMethod
	knnEfRuntime     int
	rangeRadius      float64
	rangeEpsilon     float64
	yieldVsimScoreAs string

	filter *filter.Expression

	combination          CombinationMethod
	rrfWindow            int
	rrfConstant          int
	linearAlpha          float64
	yieldCombinedScoreAs string

	numResults  int
	returnAttrs []string
	stopwords   map[string]bool
	textWeights map[string]float64
}

// NewHybridQuery creates an FT.HYBRID query with Python-parity defaults:
// BM25STD scorer, 10 results, English stopwords, server-default fusion
// (RRF).
func NewHybridQuery(text, textFieldName string, vector []float64, vectorFieldName string) *HybridQuery {
	return &HybridQuery{
		text:         text,
		textField:    textFieldName,
		vector:       vector,
		vectorField:  vectorFieldName,
		dtype:        vectors.Float32,
		textScorer:   "BM25STD",
		knnEfRuntime: 10,
		rangeEpsilon: 0.01,
		rrfWindow:    20,
		rrfConstant:  60,
		linearAlpha:  0.3,
		numResults:   10,
		stopwords:    englishStopwords(),
	}
}

// VectorBytes sets a pre-encoded query vector.
func (q *HybridQuery) VectorBytes(b []byte) *HybridQuery { q.vectorBytes = b; return q }

// Dtype sets the vector datatype (default float32).
func (q *HybridQuery) Dtype(dt vectors.DataType) *HybridQuery { q.dtype = dt; return q }

// TextScorer sets the full-text scoring algorithm (default BM25STD).
func (q *HybridQuery) TextScorer(s string) *HybridQuery { q.textScorer = s; return q }

// YieldTextScoreAs names the yielded text score field.
func (q *HybridQuery) YieldTextScoreAs(name string) *HybridQuery {
	q.yieldTextScoreAs = name
	return q
}

// UseKNN selects KNN vector search with an optional HNSW EF_RUNTIME.
func (q *HybridQuery) UseKNN(efRuntime int) *HybridQuery {
	q.searchMethod = KNN
	if efRuntime > 0 {
		q.knnEfRuntime = efRuntime
	}
	return q
}

// UseRange selects radius vector search.
func (q *HybridQuery) UseRange(radius, epsilon float64) *HybridQuery {
	q.searchMethod = Range
	q.rangeRadius = radius
	if epsilon > 0 {
		q.rangeEpsilon = epsilon
	}
	return q
}

// YieldVsimScoreAs names the yielded vector similarity score field.
func (q *HybridQuery) YieldVsimScoreAs(name string) *HybridQuery {
	q.yieldVsimScoreAs = name
	return q
}

// Filter applies a filter expression to both search arms.
func (q *HybridQuery) Filter(f *filter.Expression) *HybridQuery { q.filter = f; return q }

// CombineRRF selects reciprocal rank fusion (window/constant <= 0 keep the
// defaults 20/60).
func (q *HybridQuery) CombineRRF(window, constant int) *HybridQuery {
	q.combination = RRF
	if window > 0 {
		q.rrfWindow = window
	}
	if constant > 0 {
		q.rrfConstant = constant
	}
	return q
}

// CombineLinear selects linear fusion with the given text weight alpha
// (beta = 1 - alpha).
func (q *HybridQuery) CombineLinear(alpha float64) *HybridQuery {
	q.combination = Linear
	q.linearAlpha = alpha
	return q
}

// YieldCombinedScoreAs names the yielded fused score field.
func (q *HybridQuery) YieldCombinedScoreAs(name string) *HybridQuery {
	q.yieldCombinedScoreAs = name
	return q
}

// NumResults sets the number of results (default 10).
func (q *HybridQuery) NumResults(n int) *HybridQuery { q.numResults = n; return q }

// ReturnFields loads document attributes into the result rows.
func (q *HybridQuery) ReturnFields(fields ...string) *HybridQuery {
	q.returnAttrs = fields
	return q
}

// Stopwords replaces the client-side stopword set (default English); pass
// nothing to disable removal.
func (q *HybridQuery) Stopwords(words ...string) *HybridQuery {
	q.stopwords = map[string]bool{}
	for _, w := range words {
		q.stopwords[w] = true
	}
	return q
}

// TextWeights boosts individual words within the query text.
func (q *HybridQuery) TextWeights(weights map[string]float64) *HybridQuery {
	q.textWeights = map[string]float64{}
	for word, w := range weights {
		q.textWeights[strings.ToLower(strings.TrimSpace(word))] = w
	}
	return q
}

// SearchQueryString renders the text arm of the query.
func (q *HybridQuery) SearchQueryString() string {
	return fullTextQueryString(q.text, q.textField, q.filter, q.stopwords, q.textWeights)
}

// HybridArgs assembles the full FT.HYBRID command arguments for the named
// index.
func (q *HybridQuery) HybridArgs(indexName string) ([]any, error) {
	if strings.TrimSpace(q.text) == "" {
		return nil, fmt.Errorf("text string cannot be empty")
	}
	if err := q.filter.Err(); err != nil {
		return nil, err
	}
	blob := q.vectorBytes
	if blob == nil {
		var err error
		blob, err = vectors.ToBuffer(q.vector, q.dtype)
		if err != nil {
			return nil, err
		}
	}

	args := []any{"FT.HYBRID", indexName}

	// SEARCH arm
	args = append(args, "SEARCH", q.SearchQueryString())
	if q.textScorer != "" {
		args = append(args, "SCORER", q.textScorer)
	}
	if q.yieldTextScoreAs != "" {
		args = append(args, "YIELD_SCORE_AS", q.yieldTextScoreAs)
	}

	// VSIM arm
	args = append(args, "VSIM", "@"+q.vectorField, "$vector")
	switch q.searchMethod {
	case KNN:
		params := []any{"K", q.numResults}
		if q.knnEfRuntime > 0 {
			params = append(params, "EF_RUNTIME", q.knnEfRuntime)
		}
		args = append(args, "KNN", len(params))
		args = append(args, params...)
	case Range:
		if q.rangeRadius <= 0 {
			return nil, fmt.Errorf("must provide a radius when vector search method is RANGE")
		}
		params := []any{"RADIUS", q.rangeRadius}
		if q.rangeEpsilon > 0 {
			params = append(params, "EPSILON", q.rangeEpsilon)
		}
		args = append(args, "RANGE", len(params))
		args = append(args, params...)
	}
	if fs := filterString(q.filter); fs != "" && fs != "*" {
		args = append(args, "FILTER", fs)
	}
	if q.yieldVsimScoreAs != "" {
		args = append(args, "YIELD_SCORE_AS", q.yieldVsimScoreAs)
	}

	// COMBINE
	if q.combination != "" {
		var params []any
		switch q.combination {
		case RRF:
			params = append(params, "WINDOW", q.rrfWindow, "CONSTANT", q.rrfConstant)
		case Linear:
			params = append(params,
				"ALPHA", formatFloat(q.linearAlpha),
				"BETA", formatFloat(1-q.linearAlpha))
		}
		if q.yieldCombinedScoreAs != "" {
			params = append(params, "YIELD_SCORE_AS", q.yieldCombinedScoreAs)
		}
		args = append(args, "COMBINE", string(q.combination), len(params))
		args = append(args, params...)
	}

	// post-processing
	if len(q.returnAttrs) > 0 {
		args = append(args, "LOAD", len(q.returnAttrs))
		for _, f := range q.returnAttrs {
			args = append(args, "@"+f)
		}
	}
	if q.numResults > 0 {
		args = append(args, "LIMIT", 0, q.numResults)
	}
	args = append(args, "PARAMS", 2, "vector", blob)
	return args, nil
}
