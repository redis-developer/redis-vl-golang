// Package router implements a semantic router that classifies statements
// against routes defined by reference phrases, using vector search and
// FT.AGGREGATE. Port of redisvl.extensions.router.SemanticRouter.
package router

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis-developer/redis-vl-golang"
	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/query"
	"github.com/redis-developer/redis-vl-golang/schema"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

const routeVectorField = "vector"

// Route is a routing path defined by reference phrases.
type Route struct {
	// Name of the route (required, non-empty).
	Name string `json:"name" yaml:"name"`
	// References are example phrases for the route (required, non-empty).
	References []string `json:"references" yaml:"references"`
	// Metadata associated with the route.
	Metadata map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	// DistanceThreshold for matching the route, in (0, 2] (default 0.5).
	DistanceThreshold float64 `json:"distance_threshold" yaml:"distance_threshold"`
}

func (r *Route) validate() error {
	if strings.TrimSpace(r.Name) == "" {
		return fmt.Errorf("route name must not be empty")
	}
	if len(r.References) == 0 {
		return fmt.Errorf("route %q: references must not be empty", r.Name)
	}
	for _, ref := range r.References {
		if strings.TrimSpace(ref) == "" {
			return fmt.Errorf("route %q: all references must be non-empty strings", r.Name)
		}
	}
	if r.DistanceThreshold == 0 {
		r.DistanceThreshold = 0.5
	}
	if r.DistanceThreshold <= 0 || r.DistanceThreshold > 2 {
		return fmt.Errorf("route %q: distance_threshold must be in (0, 2]", r.Name)
	}
	return nil
}

// RouteMatch is a classification result.
type RouteMatch struct {
	// Name of the matched route ("" when no route matched).
	Name string
	// Distance is the aggregated vector distance to the route.
	Distance float64
}

// AggregationMethod combines distances of multiple references per route.
type AggregationMethod string

const (
	Avg AggregationMethod = "avg"
	Min AggregationMethod = "min"
	Sum AggregationMethod = "sum"
)

// RoutingConfig controls routing behavior.
type RoutingConfig struct {
	// MaxK is the maximum number of matches RouteMany returns (default 1).
	MaxK int
	// Aggregation method for reference distances (default Avg).
	Aggregation AggregationMethod
}

// SemanticRouterOptions configure a SemanticRouter.
type SemanticRouterOptions struct {
	// Config is the routing configuration.
	Config RoutingConfig
	// Overwrite recreates the index schema if it already exists.
	Overwrite bool
}

// SemanticRouter routes statements to the best matching Route.
type SemanticRouter struct {
	name       string
	routes     []Route
	config     RoutingConfig
	vectorizer vectorize.Vectorizer
	index      *redisvl.SearchIndex
	client     redis.UniversalClient
}

// NewSemanticRouter creates the router, its index, and indexes all route
// references.
func NewSemanticRouter(ctx context.Context, client redis.UniversalClient, name string, routes []Route, vectorizer vectorize.Vectorizer, opts ...SemanticRouterOptions) (*SemanticRouter, error) {
	var o SemanticRouterOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if vectorizer == nil {
		return nil, fmt.Errorf("a vectorizer is required (no default local model in the Go port)")
	}
	if o.Config.MaxK == 0 {
		o.Config.MaxK = 1
	}
	if o.Config.Aggregation == "" {
		o.Config.Aggregation = Avg
	}
	for i := range routes {
		if err := routes[i].validate(); err != nil {
			return nil, err
		}
	}

	vf, err := schema.NewVectorField(routeVectorField, schema.VectorAttrs{
		Dims:           vectorizer.Dims(),
		Algorithm:      schema.Flat,
		Datatype:       string(vectorizer.Dtype()),
		DistanceMetric: schema.Cosine,
	})
	if err != nil {
		return nil, err
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: name, Prefixes: []string{name}},
		schema.NewTagField("reference_id"),
		schema.NewTagField("route_name"),
		schema.NewTextField("reference"),
		vf,
	)
	if err != nil {
		return nil, err
	}

	r := &SemanticRouter{
		name:       name,
		routes:     routes,
		config:     o.Config,
		vectorizer: vectorizer,
		index:      redisvl.NewSearchIndex(s, client),
		client:     client,
	}
	if err := r.index.Create(ctx, redisvl.CreateOptions{Overwrite: o.Overwrite, Drop: o.Overwrite}); err != nil {
		return nil, err
	}
	if err := r.indexRoutes(ctx, routes); err != nil {
		return nil, err
	}
	return r, nil
}

// Name returns the router name.
func (r *SemanticRouter) Name() string { return r.name }

// Routes returns the configured routes.
func (r *SemanticRouter) Routes() []Route { return r.routes }

// Config returns the routing configuration.
func (r *SemanticRouter) Config() RoutingConfig { return r.config }

// UpdateConfig replaces the routing configuration.
func (r *SemanticRouter) UpdateConfig(cfg RoutingConfig) {
	if cfg.MaxK == 0 {
		cfg.MaxK = 1
	}
	if cfg.Aggregation == "" {
		cfg.Aggregation = Avg
	}
	r.config = cfg
}

// RouteThresholds returns the distance threshold per route name.
func (r *SemanticRouter) RouteThresholds() map[string]float64 {
	out := map[string]float64{}
	for _, route := range r.routes {
		out[route.Name] = route.DistanceThreshold
	}
	return out
}

// UpdateRouteThresholds updates distance thresholds by route name.
func (r *SemanticRouter) UpdateRouteThresholds(thresholds map[string]float64) {
	for i := range r.routes {
		if t, ok := thresholds[r.routes[i].Name]; ok {
			r.routes[i].DistanceThreshold = t
		}
	}
}

func (r *SemanticRouter) refKey(routeName, referenceHash string) string {
	return r.name + ":" + routeName + ":" + referenceHash
}

// indexRoutes embeds and stores all references of the given routes.
func (r *SemanticRouter) indexRoutes(ctx context.Context, routes []Route) error {
	pipe := r.client.Pipeline()
	for _, route := range routes {
		embeddings, err := r.vectorizer.EmbedMany(ctx, route.References)
		if err != nil {
			return fmt.Errorf("vectorizing references for route %q: %w", route.Name, err)
		}
		for i, reference := range route.References {
			refHash := redisvl.Hashify(reference, nil)
			blob, err := vectors.ToBuffer(embeddings[i], r.vectorizer.Dtype())
			if err != nil {
				return err
			}
			pipe.HSet(ctx, r.refKey(route.Name, refHash), map[string]any{
				"reference_id":   refHash,
				"route_name":     route.Name,
				"reference":      reference,
				routeVectorField: blob,
			})
		}
	}
	_, err := pipe.Exec(ctx)
	return err
}

// AddRoute adds a new route (or additional references to an existing one)
// and indexes its references.
func (r *SemanticRouter) AddRoute(ctx context.Context, route Route) error {
	if err := route.validate(); err != nil {
		return err
	}
	if err := r.indexRoutes(ctx, []Route{route}); err != nil {
		return err
	}
	for i := range r.routes {
		if r.routes[i].Name == route.Name {
			r.routes[i].References = append(r.routes[i].References, route.References...)
			return nil
		}
	}
	r.routes = append(r.routes, route)
	return nil
}

// RemoveRoute deletes a route's references from Redis and the router.
func (r *SemanticRouter) RemoveRoute(ctx context.Context, routeName string) error {
	pattern := r.name + ":" + routeName + ":*"
	var cursor uint64
	for {
		keys, next, err := r.client.Scan(ctx, cursor, pattern, 500).Result()
		if err != nil {
			return err
		}
		if len(keys) > 0 {
			if err := r.client.Del(ctx, keys...).Err(); err != nil {
				return err
			}
		}
		if next == 0 {
			break
		}
		cursor = next
	}
	for i := range r.routes {
		if r.routes[i].Name == routeName {
			r.routes = append(r.routes[:i], r.routes[i+1:]...)
			break
		}
	}
	return nil
}

// Route classifies a statement (or precomputed vector) to the single best
// route. A zero-value RouteMatch means no route matched.
func (r *SemanticRouter) Route(ctx context.Context, statement string, vector ...[]float64) (RouteMatch, error) {
	matches, err := r.routeMatches(ctx, statement, 1, vector...)
	if err != nil {
		return RouteMatch{}, err
	}
	if len(matches) == 0 {
		return RouteMatch{}, nil
	}
	return matches[0], nil
}

// RouteMany classifies a statement to up to Config.MaxK routes.
func (r *SemanticRouter) RouteMany(ctx context.Context, statement string, vector ...[]float64) ([]RouteMatch, error) {
	return r.routeMatches(ctx, statement, r.config.MaxK, vector...)
}

func (r *SemanticRouter) routeMatches(ctx context.Context, statement string, maxK int, vec ...[]float64) ([]RouteMatch, error) {
	if len(r.routes) == 0 {
		return nil, fmt.Errorf("no routes configured for router %q", r.name)
	}
	var vector []float64
	if len(vec) > 0 && vec[0] != nil {
		vector = vec[0]
	} else {
		if statement == "" {
			return nil, fmt.Errorf("must provide a vector or statement to the router")
		}
		var err error
		vector, err = r.vectorizer.Embed(ctx, statement)
		if err != nil {
			return nil, fmt.Errorf("vectorizing statement: %w", err)
		}
	}
	if maxK <= 0 {
		maxK = 1
	}

	// Range search up to the maximum route threshold; per-route thresholds
	// are enforced by the aggregation FILTER below.
	maxThreshold := 0.0
	for _, route := range r.routes {
		if route.DistanceThreshold > maxThreshold {
			maxThreshold = route.DistanceThreshold
		}
	}

	rq := query.NewVectorRangeQuery(routeVectorField, vector).
		DistanceThreshold(maxThreshold).
		Dtype(r.vectorizer.Dtype())

	blob, err := vectors.ToBuffer(vector, r.vectorizer.Dtype())
	if err != nil {
		return nil, err
	}

	reduce := strings.ToUpper(string(r.config.Aggregation))
	if reduce == "" {
		reduce = "AVG"
	}

	args := []any{
		"FT.AGGREGATE", r.name, rq.QueryString(),
		"GROUPBY", 1, "@route_name",
		"REDUCE", reduce, 1, "vector_distance", "AS", "distance",
		"SORTBY", 2, "@distance", "ASC", "MAX", maxK,
		"FILTER", r.thresholdFilter(),
		"PARAMS", 4,
		"distance_threshold", strconv.FormatFloat(maxThreshold, 'g', -1, 64),
		"vector", blob,
		"DIALECT", 2,
	}
	reply, err := r.client.Do(ctx, args...).Result()
	if err != nil {
		if strings.Contains(err.Error(), "VSS is not yet supported on FT.AGGREGATE") {
			return nil, fmt.Errorf("semantic routing is only available on Redis version 7.x or greater: %w", err)
		}
		return nil, err
	}
	return parseAggregateMatches(reply)
}

// thresholdFilter applies distance thresholds route by route (mirrors
// Python's _distance_threshold_filter).
func (r *SemanticRouter) thresholdFilter() string {
	parts := make([]string, 0, len(r.routes))
	for _, route := range r.routes {
		parts = append(parts, fmt.Sprintf("(@route_name == '%s' && @distance < %s)",
			route.Name, strconv.FormatFloat(route.DistanceThreshold, 'g', -1, 64)))
	}
	return strings.Join(parts, " || ")
}

// Clear deletes all route references but keeps the index and route
// definitions.
func (r *SemanticRouter) Clear(ctx context.Context) error {
	_, err := r.index.Clear(ctx)
	return err
}

// Delete removes the router index and all references.
func (r *SemanticRouter) Delete(ctx context.Context) error {
	return r.index.Delete(ctx, true)
}

// parseAggregateMatches parses an FT.AGGREGATE reply (RESP2 or RESP3) into
// route matches.
func parseAggregateMatches(reply any) ([]RouteMatch, error) {
	var matches []RouteMatch

	appendRow := func(fields map[string]string) {
		m := RouteMatch{Name: fields["route_name"]}
		if d, err := strconv.ParseFloat(fields["distance"], 64); err == nil {
			m.Distance = d
		}
		if m.Name != "" {
			matches = append(matches, m)
		}
	}

	switch rep := reply.(type) {
	case []any: // RESP2: [total, [k, v, ...], ...]
		for i := 1; i < len(rep); i++ {
			row, ok := rep[i].([]any)
			if !ok {
				continue
			}
			fields := map[string]string{}
			for j := 0; j+1 < len(row); j += 2 {
				fields[toStr(row[j])] = toStr(row[j+1])
			}
			appendRow(fields)
		}
	case map[any]any, map[string]any:
		m := toStrMap(rep)
		results, _ := m["results"].([]any)
		for _, res := range results {
			rm := toStrMap(res)
			if rm == nil {
				continue
			}
			// rows may nest fields under extra_attributes
			if extra := toStrMap(rm["extra_attributes"]); extra != nil {
				rm = extra
			}
			fields := map[string]string{}
			for k, v := range rm {
				fields[k] = toStr(v)
			}
			appendRow(fields)
		}
	default:
		return nil, fmt.Errorf("unexpected FT.AGGREGATE reply type: %T", reply)
	}
	return matches, nil
}

func toStr(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case []byte:
		return string(s)
	case nil:
		return ""
	}
	return fmt.Sprint(v)
}

func toStrMap(v any) map[string]any {
	switch m := v.(type) {
	case map[string]any:
		return m
	case map[any]any:
		out := make(map[string]any, len(m))
		for k, val := range m {
			out[toStr(k)] = val
		}
		return out
	}
	return nil
}
