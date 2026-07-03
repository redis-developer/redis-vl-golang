package query

import "github.com/redis-developer/redis-vl-golang/filter"

// FilterQuery runs a filtered (non-vector) search (port of
// redisvl.query.FilterQuery).
type FilterQuery struct {
	opts   Options
	filter *filter.Expression
	params map[string]any
}

// NewFilterQuery creates a filter query; a nil filter matches everything.
func NewFilterQuery(f *filter.Expression) *FilterQuery {
	return &FilterQuery{
		filter: f,
		opts:   Options{Num: 10, Dialect: 2},
	}
}

// ReturnFields sets the fields to return with results.
func (q *FilterQuery) ReturnFields(fields ...string) *FilterQuery {
	q.opts.ReturnFields = fields
	return q
}

// NumResults sets the number of results to return (default 10).
func (q *FilterQuery) NumResults(n int) *FilterQuery { q.opts.Num = n; return q }

// Paging sets the offset and number of results.
func (q *FilterQuery) Paging(offset, num int) *FilterQuery {
	q.opts.Offset = offset
	q.opts.Num = num
	return q
}

// SortByField sorts results by the given field.
func (q *FilterQuery) SortByField(field string, asc bool) *FilterQuery {
	q.opts.SortBy = &SortBy{Field: field, Asc: asc}
	return q
}

// Dialect sets the query dialect (default 2).
func (q *FilterQuery) Dialect(d int) *FilterQuery { q.opts.Dialect = d; return q }

// InOrder requires query terms to appear in order.
func (q *FilterQuery) InOrder() *FilterQuery { q.opts.InOrder = true; return q }

// WithParams attaches PARAMS to the query.
func (q *FilterQuery) WithParams(params map[string]any) *FilterQuery {
	q.params = params
	return q
}

// SetFilter replaces the filter expression.
func (q *FilterQuery) SetFilter(f *filter.Expression) *FilterQuery { q.filter = f; return q }

// Err returns any deferred filter construction error.
func (q *FilterQuery) Err() error { return q.filter.Err() }

// QueryString implements Query.
func (q *FilterQuery) QueryString() string { return filterString(q.filter) }

// Params implements Query.
func (q *FilterQuery) Params() map[string]any { return q.params }

// Options implements Query.
func (q *FilterQuery) Options() *Options { return &q.opts }

// CountQuery counts the records matching a filter expression (port of
// redisvl.query.CountQuery).
type CountQuery struct {
	opts   Options
	filter *filter.Expression
	params map[string]any
}

// NewCountQuery creates a count query; a nil filter counts all records.
func NewCountQuery(f *filter.Expression) *CountQuery {
	return &CountQuery{
		filter: f,
		opts:   Options{NoContent: true, Num: 0, Offset: 0, Dialect: 2},
	}
}

// Dialect sets the query dialect (default 2).
func (q *CountQuery) Dialect(d int) *CountQuery { q.opts.Dialect = d; return q }

// WithParams attaches PARAMS to the query.
func (q *CountQuery) WithParams(params map[string]any) *CountQuery {
	q.params = params
	return q
}

// Err returns any deferred filter construction error.
func (q *CountQuery) Err() error { return q.filter.Err() }

// QueryString implements Query.
func (q *CountQuery) QueryString() string { return filterString(q.filter) }

// Params implements Query.
func (q *CountQuery) Params() map[string]any { return q.params }

// Options implements Query.
func (q *CountQuery) Options() *Options { return &q.opts }
