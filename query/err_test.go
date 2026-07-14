package query

import (
	"testing"

	"github.com/redis/redis-vl-golang/filter"
)

func TestDeferredDtypeError(t *testing.T) {
	q := NewVectorQuery("embedding", []float64{0.1}).Dtype("float128")
	if q.Err() == nil {
		t.Error("expected deferred dtype error")
	}
	rq := NewVectorRangeQuery("embedding", []float64{0.1}).Dtype("nope")
	if rq.Err() == nil {
		t.Error("expected deferred dtype error")
	}
	aq := NewAggregateHybridQuery("hi", "description", []float64{0.1}, "embedding").Dtype("bogus")
	if aq.Err() == nil {
		t.Error("expected deferred dtype error")
	}
}

func TestDeferredFilterError(t *testing.T) {
	bad := filter.Geo("location").Within(filter.GeoRadius{
		Longitude: 1, Latitude: 2, Radius: 3, Unit: "parsec",
	})
	if bad.Err() == nil {
		t.Fatal("expected geo unit error on expression")
	}
	if bad.String() != "*" {
		t.Errorf("errored expression should render as *, got %q", bad.String())
	}
	// error propagates through combination
	combined := filter.Tag("brand").Eq("nike").And(bad)
	if combined.Err() == nil {
		t.Error("expected error to propagate through And")
	}
	// and through queries
	if NewFilterQuery(combined).Err() == nil {
		t.Error("expected FilterQuery.Err to surface filter error")
	}
	if NewVectorQuery("embedding", []float64{0.1}).Filter(combined).Err() == nil {
		t.Error("expected VectorQuery.Err to surface filter error")
	}
}

func TestValidDtypeNoError(t *testing.T) {
	q := NewVectorQuery("embedding", []float64{0.1}).Dtype("float16")
	if err := q.Err(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_ = q.Params() // encoding should also succeed
	if err := q.Err(); err != nil {
		t.Errorf("unexpected error after Params: %v", err)
	}
}
