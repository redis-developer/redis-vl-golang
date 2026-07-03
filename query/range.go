package query

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/redis-developer/redis-vl-golang/filter"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// VectorRangeQuery finds all vectors within a distance threshold, with an
// optional filter (port of redisvl.query.VectorRangeQuery).
type VectorRangeQuery struct {
	opts Options
	err  error

	fieldName   string
	vector      []float64
	vectorBytes []byte
	dtype       vectors.DataType
	filter      *filter.Expression

	distanceThreshold float64
	epsilon           *float64
	hybridPolicy      HybridPolicy
	batchSize         int

	// SVS-VAMANA runtime parameters
	searchWindowSize     int
	useSearchHistory     string
	searchBufferCapacity int

	normalizeDistance bool
}

// NewVectorRangeQuery creates a vector range query on the given field with
// the default distance threshold of 0.2.
func NewVectorRangeQuery(vectorFieldName string, vector []float64) *VectorRangeQuery {
	q := &VectorRangeQuery{
		fieldName:         vectorFieldName,
		vector:            vector,
		dtype:             vectors.Float32,
		distanceThreshold: 0.2,
		opts: Options{
			Num:     10,
			Dialect: 2,
			SortBy:  &SortBy{Field: DistanceID, Asc: true},
		},
	}
	q.opts.ReturnFields = append(q.opts.ReturnFields, DistanceID)
	return q
}

// NewVectorRangeQueryFromBytes creates a range query from a pre-encoded
// vector buffer.
func NewVectorRangeQueryFromBytes(vectorFieldName string, vector []byte) *VectorRangeQuery {
	q := NewVectorRangeQuery(vectorFieldName, nil)
	q.vectorBytes = vector
	return q
}

// DistanceThreshold sets the vector distance threshold (default 0.2). When
// NormalizeVectorDistance is enabled the threshold is interpreted as a 0..1
// similarity score and denormalized before use.
func (q *VectorRangeQuery) DistanceThreshold(t float64) *VectorRangeQuery {
	q.distanceThreshold = t
	return q
}

// Epsilon sets the range search boundary factor.
func (q *VectorRangeQuery) Epsilon(e float64) *VectorRangeQuery { q.epsilon = &e; return q }

// Dtype sets the vector datatype (default float32). An invalid datatype is
// surfaced via Err when the query is executed.
func (q *VectorRangeQuery) Dtype(dt vectors.DataType) *VectorRangeQuery {
	if _, err := vectors.Parse(string(dt)); err != nil {
		q.err = err
		return q
	}
	q.dtype = dt
	return q
}

// Err returns any deferred construction error (invalid dtype or filter).
func (q *VectorRangeQuery) Err() error {
	if q.err != nil {
		return q.err
	}
	return q.filter.Err()
}

// Filter applies a filter expression alongside the range query.
func (q *VectorRangeQuery) Filter(f *filter.Expression) *VectorRangeQuery { q.filter = f; return q }

// ReturnFields sets the fields to return with results. The vector distance
// is always included.
func (q *VectorRangeQuery) ReturnFields(fields ...string) *VectorRangeQuery {
	for _, f := range fields {
		if f == DistanceID {
			q.opts.ReturnFields = fields
			return q
		}
	}
	q.opts.ReturnFields = append(fields, DistanceID)
	return q
}

// NumResults sets the maximum number of results (default 10).
func (q *VectorRangeQuery) NumResults(k int) *VectorRangeQuery { q.opts.Num = k; return q }

// SortByField overrides the default sort (vector distance).
func (q *VectorRangeQuery) SortByField(field string, asc bool) *VectorRangeQuery {
	q.opts.SortBy = &SortBy{Field: field, Asc: asc}
	return q
}

// Dialect sets the query dialect (default 2).
func (q *VectorRangeQuery) Dialect(d int) *VectorRangeQuery { q.opts.Dialect = d; return q }

// InOrder requires query terms to appear in order.
func (q *VectorRangeQuery) InOrder() *VectorRangeQuery { q.opts.InOrder = true; return q }

// SetHybridPolicy sets BATCHES or ADHOC_BF (passed via PARAMS).
func (q *VectorRangeQuery) SetHybridPolicy(p HybridPolicy) *VectorRangeQuery {
	q.hybridPolicy = p
	return q
}

// BatchSize sets the batch size when the hybrid policy is BATCHES.
func (q *VectorRangeQuery) BatchSize(n int) *VectorRangeQuery { q.batchSize = n; return q }

// SearchWindowSize sets the SVS-VAMANA SEARCH_WINDOW_SIZE attribute.
func (q *VectorRangeQuery) SearchWindowSize(n int) *VectorRangeQuery {
	q.searchWindowSize = n
	return q
}

// UseSearchHistory sets the SVS-VAMANA USE_SEARCH_HISTORY attribute.
func (q *VectorRangeQuery) UseSearchHistory(v string) *VectorRangeQuery {
	q.useSearchHistory = strings.ToUpper(v)
	return q
}

// SearchBufferCapacity sets the SVS-VAMANA SEARCH_BUFFER_CAPACITY attribute.
func (q *VectorRangeQuery) SearchBufferCapacity(n int) *VectorRangeQuery {
	q.searchBufferCapacity = n
	return q
}

// NormalizeVectorDistance treats the distance threshold as a 0..1 similarity
// score and converts returned distances to scores.
func (q *VectorRangeQuery) NormalizeVectorDistance() *VectorRangeQuery {
	q.normalizeDistance = true
	return q
}

// NormalizeDistanceEnabled reports whether distance normalization was
// requested.
func (q *VectorRangeQuery) NormalizeDistanceEnabled() bool { return q.normalizeDistance }

// VectorFieldName returns the vector field being searched.
func (q *VectorRangeQuery) VectorFieldName() string { return q.fieldName }

// QueryString implements Query.
func (q *VectorRangeQuery) QueryString() string {
	base := fmt.Sprintf("@%s:[VECTOR_RANGE $distance_threshold $vector]", q.fieldName)

	attrs := []string{"$YIELD_DISTANCE_AS: " + DistanceID}
	if q.epsilon != nil {
		attrs = append(attrs, "$EPSILON: "+strconv.FormatFloat(*q.epsilon, 'g', -1, 64))
	}
	if q.searchWindowSize > 0 {
		attrs = append(attrs, fmt.Sprintf("$SEARCH_WINDOW_SIZE: %d", q.searchWindowSize))
	}
	if q.useSearchHistory != "" {
		attrs = append(attrs, "$USE_SEARCH_HISTORY: "+q.useSearchHistory)
	}
	if q.searchBufferCapacity > 0 {
		attrs = append(attrs, fmt.Sprintf("$SEARCH_BUFFER_CAPACITY: %d", q.searchBufferCapacity))
	}
	attrSection := "=>{" + strings.Join(attrs, "; ") + "}"

	fe := filterString(q.filter)
	if fe == "*" {
		return base + attrSection
	}
	return "(" + base + attrSection + " " + fe + ")"
}

// Params implements Query.
func (q *VectorRangeQuery) Params() map[string]any {
	var blob []byte
	if q.vectorBytes != nil {
		blob = q.vectorBytes
	} else {
		var err error
		blob, err = vectors.ToBuffer(q.vector, q.dtype)
		if err != nil {
			q.err = err // surfaced via Err() at execution time
		}
	}
	threshold := q.distanceThreshold
	if q.normalizeDistance {
		threshold = DenormCosineDistance(threshold)
	}
	params := map[string]any{
		"vector":             blob,
		"distance_threshold": threshold,
	}
	if q.hybridPolicy != "" {
		params["HYBRID_POLICY"] = string(q.hybridPolicy)
		if q.hybridPolicy == Batches && q.batchSize > 0 {
			params["BATCH_SIZE"] = q.batchSize
		}
	}
	return params
}

// Options implements Query.
func (q *VectorRangeQuery) Options() *Options { return &q.opts }
