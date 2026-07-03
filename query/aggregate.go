package query

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/redis-developer/redis-vl-golang/filter"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// AggregationQuery is any query executed via FT.AGGREGATE
// (port of redisvl.query.AggregationQuery).
type AggregationQuery interface {
	// AggregateArgs assembles the full FT.AGGREGATE command arguments for
	// the named index, including PARAMS and DIALECT.
	AggregateArgs(indexName string) []any
}

// Vector is one vector clause of a MultiVectorQuery (port of
// redisvl.query.aggregate.Vector).
type Vector struct {
	// Values is the query vector.
	Values []float64
	// Bytes is a pre-encoded query vector (used instead of Values when set).
	Bytes []byte
	// FieldName is the vector field to search (must be indexed with the
	// cosine distance metric).
	FieldName string
	// Dtype is the vector datatype (default float32).
	Dtype vectors.DataType
	// Weight of this vector in the combined score (default 1.0).
	Weight float64
	// MaxDistance is the range-search radius in [0, 2] (default 2.0).
	MaxDistance float64
}

func (v *Vector) applyDefaults() error {
	if v.FieldName == "" {
		return fmt.Errorf("vector clause requires a FieldName")
	}
	if v.Dtype == "" {
		v.Dtype = vectors.Float32
	}
	if _, err := vectors.Parse(string(v.Dtype)); err != nil {
		return err
	}
	if v.Weight == 0 {
		v.Weight = 1.0
	}
	if v.MaxDistance == 0 {
		v.MaxDistance = 2.0
	}
	if v.MaxDistance < 0 || v.MaxDistance > 2 {
		return fmt.Errorf("max_distance must be a value between 0.0 and 2.0")
	}
	return nil
}

func (v *Vector) blob() ([]byte, error) {
	if v.Bytes != nil {
		return v.Bytes, nil
	}
	return vectors.ToBuffer(v.Values, v.Dtype)
}

// ---------------------------------------------------------- hybrid (aggregate)

// AggregateHybridQuery combines full-text and vector similarity scoring in
// one FT.AGGREGATE pipeline: hybrid_score = (1-alpha)*text_score +
// alpha*vector_similarity (port of redisvl.query.AggregateHybridQuery).
//
// Note: FT.AGGREGATE does not support runtime parameters like EF_RUNTIME;
// use VectorQuery / VectorRangeQuery for those.
type AggregateHybridQuery struct {
	err         error
	text        string
	textField   string
	vector      []float64
	vectorBytes []byte
	vectorField string
	textScorer  string
	filter      *filter.Expression
	alpha       float64
	dtype       vectors.DataType
	numResults  int
	returnAttrs []string
	stopwords   map[string]bool
	textWeights map[string]float64
	dialect     int
}

// NewAggregateHybridQuery creates a hybrid text+vector aggregation with
// defaults matching Python: BM25STD scorer, alpha 0.7, 10 results, English
// stopwords, dialect 2.
func NewAggregateHybridQuery(text, textFieldName string, vector []float64, vectorFieldName string) *AggregateHybridQuery {
	return &AggregateHybridQuery{
		text:        text,
		textField:   textFieldName,
		vector:      vector,
		vectorField: vectorFieldName,
		textScorer:  "BM25STD",
		alpha:       0.7,
		dtype:       vectors.Float32,
		numResults:  10,
		stopwords:   englishStopwords(),
		dialect:     2,
	}
}

// VectorBytes sets a pre-encoded query vector.
func (q *AggregateHybridQuery) VectorBytes(b []byte) *AggregateHybridQuery {
	q.vectorBytes = b
	return q
}

// Alpha sets the vector weight: hybrid_score = (1-alpha)*text_score +
// alpha*vector_similarity (default 0.7).
func (q *AggregateHybridQuery) Alpha(a float64) *AggregateHybridQuery { q.alpha = a; return q }

// TextScorer sets the full-text scoring algorithm (default BM25STD).
func (q *AggregateHybridQuery) TextScorer(s string) *AggregateHybridQuery {
	q.textScorer = s
	return q
}

// Filter applies a filter expression to the text portion of the query.
func (q *AggregateHybridQuery) Filter(f *filter.Expression) *AggregateHybridQuery {
	q.filter = f
	return q
}

// Dtype sets the vector datatype (default float32). An invalid datatype is
// surfaced via Err when the query is executed.
func (q *AggregateHybridQuery) Dtype(dt vectors.DataType) *AggregateHybridQuery {
	if _, err := vectors.Parse(string(dt)); err != nil {
		q.err = err
		return q
	}
	q.dtype = dt
	return q
}

// Err returns any deferred construction error (invalid dtype or filter).
func (q *AggregateHybridQuery) Err() error {
	if q.err != nil {
		return q.err
	}
	return q.filter.Err()
}

// NumResults sets the number of results (default 10).
func (q *AggregateHybridQuery) NumResults(n int) *AggregateHybridQuery {
	q.numResults = n
	return q
}

// ReturnFields loads additional document attributes into the result rows.
func (q *AggregateHybridQuery) ReturnFields(fields ...string) *AggregateHybridQuery {
	q.returnAttrs = fields
	return q
}

// Stopwords replaces the client-side stopword set (default English); pass
// nothing to disable removal.
func (q *AggregateHybridQuery) Stopwords(words ...string) *AggregateHybridQuery {
	q.stopwords = map[string]bool{}
	for _, w := range words {
		q.stopwords[w] = true
	}
	return q
}

// TextWeights boosts individual words within the query text.
func (q *AggregateHybridQuery) TextWeights(weights map[string]float64) *AggregateHybridQuery {
	q.textWeights = map[string]float64{}
	for word, w := range weights {
		q.textWeights[strings.ToLower(strings.TrimSpace(word))] = w
	}
	return q
}

// Dialect sets the query dialect (default 2).
func (q *AggregateHybridQuery) Dialect(d int) *AggregateHybridQuery { q.dialect = d; return q }

// QueryString renders the underlying query expression.
func (q *AggregateHybridQuery) QueryString() string {
	text := fullTextQueryString(q.text, q.textField, q.filter, q.stopwords, q.textWeights)
	knn := fmt.Sprintf("KNN %d @%s $vector AS %s", q.numResults, q.vectorField, DistanceID)
	return text + "=>[" + knn + "]"
}

// AggregateArgs implements AggregationQuery.
func (q *AggregateHybridQuery) AggregateArgs(indexName string) []any {
	blob := q.vectorBytes
	if blob == nil {
		var err error
		blob, err = vectors.ToBuffer(q.vector, q.dtype)
		if err != nil {
			q.err = err // surfaced via Err() at execution time
		}
	}

	args := []any{"FT.AGGREGATE", indexName, q.QueryString()}
	if len(q.returnAttrs) > 0 {
		args = append(args, "LOAD", len(q.returnAttrs))
		for _, f := range q.returnAttrs {
			args = append(args, f)
		}
	}
	args = append(args, "SCORER", q.textScorer, "ADDSCORES")
	args = append(args,
		"APPLY", fmt.Sprintf("(2 - @%s)/2", DistanceID), "AS", "vector_similarity",
		"APPLY", "@__score", "AS", "text_score",
	)
	args = append(args,
		"APPLY", fmt.Sprintf("%s*@text_score + %s*@vector_similarity",
			formatFloat(1-q.alpha), formatFloat(q.alpha)),
		"AS", "hybrid_score",
	)
	args = append(args, "SORTBY", 2, "@hybrid_score", "DESC", "MAX", q.numResults)
	args = append(args, "PARAMS", 2, "vector", blob)
	args = append(args, "DIALECT", q.dialect)
	return args
}

// ---------------------------------------------------------- multi-vector

// MultiVectorQuery searches multiple vector fields simultaneously and
// scores documents by the weighted sum of per-field cosine similarities
// (port of redisvl.query.MultiVectorQuery). All vector fields must use the
// cosine distance metric.
type MultiVectorQuery struct {
	err         error
	vecs        []Vector
	filter      *filter.Expression
	numResults  int
	returnAttrs []string
	dialect     int
}

// NewMultiVectorQuery creates a multi-vector aggregation query.
func NewMultiVectorQuery(vecs ...Vector) (*MultiVectorQuery, error) {
	if len(vecs) == 0 {
		return nil, fmt.Errorf("at least one Vector is required")
	}
	for i := range vecs {
		if err := vecs[i].applyDefaults(); err != nil {
			return nil, err
		}
	}
	return &MultiVectorQuery{vecs: vecs, numResults: 10, dialect: 2}, nil
}

// Filter applies an additional filter expression.
func (q *MultiVectorQuery) Filter(f *filter.Expression) *MultiVectorQuery { q.filter = f; return q }

// NumResults sets the number of results (default 10).
func (q *MultiVectorQuery) NumResults(n int) *MultiVectorQuery { q.numResults = n; return q }

// ReturnFields loads additional document attributes into the result rows.
func (q *MultiVectorQuery) ReturnFields(fields ...string) *MultiVectorQuery {
	q.returnAttrs = fields
	return q
}

// Dialect sets the query dialect (default 2).
func (q *MultiVectorQuery) Dialect(d int) *MultiVectorQuery { q.dialect = d; return q }

// Err returns any deferred construction error (invalid dtype or filter).
func (q *MultiVectorQuery) Err() error {
	if q.err != nil {
		return q.err
	}
	return q.filter.Err()
}

// QueryString renders the underlying query expression.
func (q *MultiVectorQuery) QueryString() string {
	ranges := make([]string, len(q.vecs))
	for i, v := range q.vecs {
		ranges[i] = fmt.Sprintf("@%s:[VECTOR_RANGE %s $vector_%d]=>{$YIELD_DISTANCE_AS: distance_%d}",
			v.FieldName, formatFloat(v.MaxDistance), i, i)
	}
	rangeQuery := strings.Join(ranges, " AND ")

	fs := filterString(q.filter)
	if fs != "" && fs != "*" {
		return "(" + rangeQuery + ") AND (" + fs + ")"
	}
	return rangeQuery
}

// AggregateArgs implements AggregationQuery.
func (q *MultiVectorQuery) AggregateArgs(indexName string) []any {
	args := []any{"FT.AGGREGATE", indexName, q.QueryString()}
	if len(q.returnAttrs) > 0 {
		args = append(args, "LOAD", len(q.returnAttrs))
		for _, f := range q.returnAttrs {
			args = append(args, f)
		}
	}

	// per-vector similarity scores
	for i := range q.vecs {
		args = append(args,
			"APPLY", fmt.Sprintf("(2 - @distance_%d)/2", i),
			"AS", fmt.Sprintf("score_%d", i),
		)
	}

	// weighted combination
	terms := make([]string, len(q.vecs))
	for i, v := range q.vecs {
		terms[i] = fmt.Sprintf("@score_%d * %s", i, formatFloat(v.Weight))
	}
	args = append(args, "APPLY", strings.Join(terms, " + "), "AS", "combined_score")

	args = append(args, "SORTBY", 2, "@combined_score", "DESC", "MAX", q.numResults)

	args = append(args, "PARAMS", len(q.vecs)*2)
	for i := range q.vecs {
		blob, err := q.vecs[i].blob()
		if err != nil {
			q.err = err // surfaced via Err() at execution time
		}
		args = append(args, fmt.Sprintf("vector_%d", i), blob)
	}
	args = append(args, "DIALECT", q.dialect)
	return args
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}
