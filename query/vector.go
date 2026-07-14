package query

import (
	"fmt"
	"strings"

	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/vectors"
)

// HybridPolicy controls how filters are applied during vector search.
type HybridPolicy string

// Supported hybrid filter policies.
const (
	Batches HybridPolicy = "BATCHES"
	AdhocBF HybridPolicy = "ADHOC_BF"
)

// VectorQuery is a KNN vector similarity search with an optional filter
// (port of redisvl.query.VectorQuery). Build with NewVectorQuery and chain
// setters.
type VectorQuery struct {
	opts Options
	err  error

	fieldName    string
	vector       []float64
	vectorBytes  []byte
	dtype        vectors.DataType
	filter       *filter.Expression
	numResults   int
	hybridPolicy HybridPolicy
	batchSize    int
	efRuntime    int

	// SVS-VAMANA runtime parameters
	searchWindowSize     int
	useSearchHistory     string
	searchBufferCapacity int

	normalizeDistance bool
}

// NewVectorQuery creates a KNN query on the given vector field.
func NewVectorQuery(vectorFieldName string, vector []float64) *VectorQuery {
	q := &VectorQuery{
		fieldName:  vectorFieldName,
		vector:     vector,
		dtype:      vectors.Float32,
		numResults: 10,
		opts: Options{
			Num:     10,
			Dialect: 2,
			SortBy:  &SortBy{Field: DistanceID, Asc: true},
		},
	}
	q.opts.ReturnFields = append(q.opts.ReturnFields, DistanceID)
	return q
}

// NewVectorQueryFromBytes creates a KNN query from a pre-encoded vector
// buffer.
func NewVectorQueryFromBytes(vectorFieldName string, vector []byte) *VectorQuery {
	q := NewVectorQuery(vectorFieldName, nil)
	q.vectorBytes = vector
	return q
}

// Dtype sets the vector datatype used to encode the query vector
// (default float32). An invalid datatype is surfaced via Err when the
// query is executed.
func (q *VectorQuery) Dtype(dt vectors.DataType) *VectorQuery {
	if _, err := vectors.Parse(string(dt)); err != nil {
		q.err = err
		return q
	}
	q.dtype = dt
	return q
}

// Err returns any deferred construction error (invalid dtype or filter).
func (q *VectorQuery) Err() error {
	if q.err != nil {
		return q.err
	}
	return q.filter.Err()
}

// Filter applies a filter expression alongside the vector search.
func (q *VectorQuery) Filter(f *filter.Expression) *VectorQuery { q.filter = f; return q }

// ReturnFields sets the fields to return with results. The vector distance
// is always included unless NoReturnScore is called.
func (q *VectorQuery) ReturnFields(fields ...string) *VectorQuery {
	for _, f := range fields {
		if f == DistanceID {
			q.opts.ReturnFields = fields
			return q
		}
	}
	q.opts.ReturnFields = append(fields, DistanceID)
	return q
}

// NoReturnScore removes the vector distance from the returned fields.
func (q *VectorQuery) NoReturnScore() *VectorQuery {
	out := q.opts.ReturnFields[:0]
	for _, f := range q.opts.ReturnFields {
		if f != DistanceID {
			out = append(out, f)
		}
	}
	q.opts.ReturnFields = out
	return q
}

// NumResults sets the top-k results to return (default 10).
func (q *VectorQuery) NumResults(k int) *VectorQuery {
	q.numResults = k
	q.opts.Num = k
	return q
}

// Paging sets the offset and number of results.
func (q *VectorQuery) Paging(offset, num int) *VectorQuery {
	q.opts.Offset = offset
	q.opts.Num = num
	q.numResults = num
	return q
}

// SortByField overrides the default sort (vector distance).
func (q *VectorQuery) SortByField(field string, asc bool) *VectorQuery {
	q.opts.SortBy = &SortBy{Field: field, Asc: asc}
	return q
}

// Dialect sets the query dialect (default 2).
func (q *VectorQuery) Dialect(d int) *VectorQuery { q.opts.Dialect = d; return q }

// InOrder requires query terms to appear in order.
func (q *VectorQuery) InOrder() *VectorQuery { q.opts.InOrder = true; return q }

// SetHybridPolicy sets BATCHES or ADHOC_BF.
func (q *VectorQuery) SetHybridPolicy(p HybridPolicy) *VectorQuery { q.hybridPolicy = p; return q }

// BatchSize sets the batch size when the hybrid policy is BATCHES.
func (q *VectorQuery) BatchSize(n int) *VectorQuery { q.batchSize = n; return q }

// EfRuntime sets the HNSW EF_RUNTIME parameter.
func (q *VectorQuery) EfRuntime(n int) *VectorQuery { q.efRuntime = n; return q }

// SearchWindowSize sets the SVS-VAMANA SEARCH_WINDOW_SIZE parameter.
func (q *VectorQuery) SearchWindowSize(n int) *VectorQuery { q.searchWindowSize = n; return q }

// UseSearchHistory sets the SVS-VAMANA USE_SEARCH_HISTORY parameter
// ("OFF", "ON", or "AUTO").
func (q *VectorQuery) UseSearchHistory(v string) *VectorQuery {
	q.useSearchHistory = strings.ToUpper(v)
	return q
}

// SearchBufferCapacity sets the SVS-VAMANA SEARCH_BUFFER_CAPACITY parameter.
func (q *VectorQuery) SearchBufferCapacity(n int) *VectorQuery {
	q.searchBufferCapacity = n
	return q
}

// NormalizeVectorDistance converts COSINE/L2 distances to a 0..1 similarity
// score in results.
func (q *VectorQuery) NormalizeVectorDistance() *VectorQuery {
	q.normalizeDistance = true
	return q
}

// NormalizeDistanceEnabled reports whether distance normalization was
// requested.
func (q *VectorQuery) NormalizeDistanceEnabled() bool { return q.normalizeDistance }

// VectorFieldName returns the vector field being searched.
func (q *VectorQuery) VectorFieldName() string { return q.fieldName }

// QueryString implements Query.
func (q *VectorQuery) QueryString() string {
	knn := fmt.Sprintf("KNN %d @%s $vector", q.numResults, q.fieldName)
	if q.hybridPolicy != "" {
		knn += " HYBRID_POLICY " + string(q.hybridPolicy)
		if q.hybridPolicy == Batches && q.batchSize > 0 {
			knn += fmt.Sprintf(" BATCH_SIZE %d", q.batchSize)
		}
	}
	if q.efRuntime > 0 {
		knn += " EF_RUNTIME $EF"
	}
	if q.searchWindowSize > 0 {
		knn += " SEARCH_WINDOW_SIZE $SEARCH_WINDOW_SIZE"
	}
	if q.useSearchHistory != "" {
		knn += " USE_SEARCH_HISTORY $USE_SEARCH_HISTORY"
	}
	if q.searchBufferCapacity > 0 {
		knn += " SEARCH_BUFFER_CAPACITY $SEARCH_BUFFER_CAPACITY"
	}
	knn += " AS " + DistanceID
	return fmt.Sprintf("%s=>[%s]", filterString(q.filter), knn)
}

// Params implements Query.
func (q *VectorQuery) Params() map[string]any {
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
	params := map[string]any{"vector": blob}
	if q.efRuntime > 0 {
		params["EF"] = q.efRuntime
	}
	if q.searchWindowSize > 0 {
		params["SEARCH_WINDOW_SIZE"] = q.searchWindowSize
	}
	if q.useSearchHistory != "" {
		params["USE_SEARCH_HISTORY"] = q.useSearchHistory
	}
	if q.searchBufferCapacity > 0 {
		params["SEARCH_BUFFER_CAPACITY"] = q.searchBufferCapacity
	}
	return params
}

// Options implements Query.
func (q *VectorQuery) Options() *Options { return &q.opts }
