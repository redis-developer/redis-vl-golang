package router

import (
	"testing"
)

func TestThresholdFilter(t *testing.T) {
	r := &SemanticRouter{routes: []Route{
		{Name: "greeting", DistanceThreshold: 0.3},
		{Name: "farewell", DistanceThreshold: 0.5},
	}}
	want := "(@route_name == 'greeting' && @distance < 0.3) || (@route_name == 'farewell' && @distance < 0.5)"
	if got := r.thresholdFilter(); got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestRouteValidate(t *testing.T) {
	r := Route{Name: "x", References: []string{"hello"}}
	if err := r.validate(); err != nil {
		t.Fatal(err)
	}
	if r.DistanceThreshold != 0.5 {
		t.Errorf("default threshold = %v", r.DistanceThreshold)
	}
	bad := []Route{
		{Name: "", References: []string{"a"}},
		{Name: "x", References: nil},
		{Name: "x", References: []string{" "}},
		{Name: "x", References: []string{"a"}, DistanceThreshold: 3},
	}
	for i, b := range bad {
		if err := b.validate(); err == nil {
			t.Errorf("case %d: expected validation error", i)
		}
	}
}

func TestParseAggregateMatchesRESP2(t *testing.T) {
	reply := []any{
		int64(2),
		[]any{"route_name", "greeting", "distance", "0.11"},
		[]any{"route_name", "farewell", "distance", "0.42"},
	}
	matches, err := parseAggregateMatches(reply)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 2 || matches[0].Name != "greeting" || matches[0].Distance != 0.11 {
		t.Errorf("matches = %+v", matches)
	}
}

func TestParseAggregateMatchesRESP3(t *testing.T) {
	reply := map[any]any{
		"total_results": int64(1),
		"results": []any{
			map[any]any{
				"extra_attributes": map[any]any{
					"route_name": "greeting",
					"distance":   "0.2",
				},
			},
		},
	}
	matches, err := parseAggregateMatches(reply)
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 || matches[0].Name != "greeting" || matches[0].Distance != 0.2 {
		t.Errorf("matches = %+v", matches)
	}
}

func TestRouteThresholdUpdates(t *testing.T) {
	r := &SemanticRouter{routes: []Route{
		{Name: "a", DistanceThreshold: 0.5},
		{Name: "b", DistanceThreshold: 0.5},
	}}
	r.UpdateRouteThresholds(map[string]float64{"a": 0.9})
	th := r.RouteThresholds()
	if th["a"] != 0.9 || th["b"] != 0.5 {
		t.Errorf("thresholds = %v", th)
	}
}
