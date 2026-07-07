package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	redisvl "github.com/redis-developer/redis-vl-golang"
	"github.com/redis-developer/redis-vl-golang/filter"
	"github.com/redis-developer/redis-vl-golang/query"
	"github.com/redis-developer/redis-vl-golang/schema"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// defaultImage is the pinned Redis image, matching the integration tests
// and the Python benchmark so comparisons stay apples-to-apples.
const defaultImage = "redis:8.8.0"

func main() {
	var (
		docs        = flag.Int("docs", 10000, "number of documents to load")
		dims        = flag.Int("dims", 384, "vector dimensions")
		queries     = flag.Int("queries", 500, "sequential queries")
		concurrency = flag.Int("concurrency", 32, "concurrent workers")
		concQueries = flag.Int("conc-queries", 3200, "total queries in the concurrent phase")
		k           = flag.Int("k", 10, "KNN results per query")
		url         = flag.String("url", "", "external redis url (default: start a fresh testcontainer)")
	)
	flag.Parse()

	if err := run(*docs, *dims, *queries, *concurrency, *concQueries, *k, *url); err != nil {
		fmt.Fprintln(os.Stderr, "gobench:", err)
		os.Exit(1)
	}
}

// startContainer launches a fresh Redis testcontainer and returns its
// connection URL and a cleanup function.
func startContainer(ctx context.Context) (string, func(), error) {
	image := os.Getenv("REDIS_IMAGE")
	if image == "" {
		image = defaultImage
	}
	fmt.Fprintf(os.Stderr, "starting %s testcontainer (pass -url to use an external Redis)...\n", image)

	container, err := testcontainers.Run(ctx, image,
		testcontainers.WithExposedPorts("6379/tcp"),
		testcontainers.WithWaitStrategy(wait.ForListeningPort("6379/tcp")),
	)
	if err != nil {
		return "", nil, fmt.Errorf("starting redis testcontainer (is Docker running?): %w", err)
	}
	cleanup := func() { _ = testcontainers.TerminateContainer(container) }

	url, err := container.PortEndpoint(ctx, "6379/tcp", "redis")
	if err != nil {
		cleanup()
		return "", nil, err
	}
	return url, cleanup, nil
}

func run(nDocs, dims, nQueries, concurrency, concQueries, k int, url string) error {
	ctx := context.Background()

	// Each benchmark run gets its own identical, freshly started database
	// unless the user explicitly points at an external one.
	if url == "" {
		containerURL, cleanup, err := startContainer(ctx)
		if err != nil {
			return err
		}
		defer cleanup()
		url = containerURL
	}

	vf, err := schema.NewVectorField("embedding", schema.VectorAttrs{
		Dims: dims, Algorithm: schema.HNSW, Datatype: "float32",
		DistanceMetric: schema.Cosine,
	})
	if err != nil {
		return err
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: "bench-go", Prefixes: []string{"bench-go"}},
		schema.NewTextField("content"),
		schema.NewTagField("category"),
		schema.NewNumericField("price"),
		vf,
	)
	if err != nil {
		return err
	}
	index, err := redisvl.NewSearchIndexFromURL(s, url)
	if err != nil {
		return err
	}
	defer index.Close()
	defer index.Delete(context.Background(), true) //nolint:errcheck // best-effort cleanup

	if err := index.Create(ctx, redisvl.CreateOptions{Overwrite: true, Drop: true}); err != nil {
		return err
	}

	// --- dataset (generation excluded from timings) ---
	rng := rand.New(rand.NewSource(42))
	records := make([]map[string]any, nDocs)
	for i := range records {
		blob, err := vectors.ToBuffer(randomUnitVector(rng, dims), vectors.Float32)
		if err != nil {
			return err
		}
		records[i] = map[string]any{
			"doc_id":    fmt.Sprintf("%d", i),
			"content":   fmt.Sprintf("document %d about topic %d with some benchmark filler text", i, i%10),
			"category":  fmt.Sprintf("cat%d", i%10),
			"price":     float64(i % 1000),
			"embedding": blob,
		}
	}
	queryVectors := make([][]float64, nQueries)
	for i := range queryVectors {
		queryVectors[i] = randomUnitVector(rng, dims)
	}

	// --- 1. load ---
	loadStart := time.Now()
	if _, err := index.Load(ctx, records, redisvl.LoadOptions{IDField: "doc_id", BatchSize: 500}); err != nil {
		return err
	}
	loadSecs := time.Since(loadStart).Seconds()

	if err := waitIndexed(ctx, index, nDocs); err != nil {
		return err
	}

	// --- 2. sequential queries ---
	latencies := make([]float64, 0, nQueries)
	for i := 0; i < nQueries; i++ {
		start := time.Now()
		if _, err := index.Query(ctx, buildQuery(queryVectors[i], i, k)); err != nil {
			return err
		}
		latencies = append(latencies, float64(time.Since(start).Microseconds())/1000)
	}

	// --- 3. concurrent queries ---
	concLatencies := make([]float64, concQueries)
	var wg sync.WaitGroup
	jobs := make(chan int)
	concStart := time.Now()
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				start := time.Now()
				_, err := index.Query(ctx, buildQuery(queryVectors[i%len(queryVectors)], i, k))
				if err != nil {
					fmt.Fprintln(os.Stderr, "query error:", err)
				}
				concLatencies[i] = float64(time.Since(start).Microseconds()) / 1000
			}
		}()
	}
	for i := 0; i < concQueries; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	concSecs := time.Since(concStart).Seconds()

	report("go", nDocs, loadSecs, latencies, concQueries, concurrency, concSecs, concLatencies)
	return nil
}

func buildQuery(vec []float64, i, k int) *query.VectorQuery {
	q := query.NewVectorQuery("embedding", vec).
		NumResults(k).
		ReturnFields("content", "category", "price")
	if i%2 == 0 {
		q.Filter(filter.Tag("category").Eq(fmt.Sprintf("cat%d", i%10)))
	}
	return q
}

func randomUnitVector(rng *rand.Rand, dims int) []float64 {
	v := make([]float64, dims)
	var norm float64
	for i := range v {
		v[i] = rng.Float64()*2 - 1
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= norm
	}
	return v
}

// waitIndexed polls FT.INFO until all documents are indexed, so query
// timings do not race background indexing.
func waitIndexed(ctx context.Context, index *redisvl.SearchIndex, want int) error {
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		info, err := index.Info(ctx)
		if err != nil {
			return err
		}
		if n, ok := toInt(info["num_docs"]); ok && n >= int64(want) {
			if pct, ok := toFloat(info["percent_indexed"]); !ok || pct >= 1 {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("index did not finish indexing %d docs in time", want)
}

func toInt(v any) (int64, bool) {
	switch x := v.(type) {
	case int64:
		return x, true
	case float64:
		return int64(x), true
	case string:
		var n int64
		_, err := fmt.Sscan(x, &n)
		return n, err == nil
	}
	return 0, false
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int64:
		return float64(x), true
	case string:
		var f float64
		_, err := fmt.Sscan(x, &f)
		return f, err == nil
	}
	return 0, false
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func report(impl string, nDocs int, loadSecs float64, seq []float64, concQueries, concurrency int, concSecs float64, conc []float64) {
	sort.Float64s(seq)
	sort.Float64s(conc)
	var mean float64
	for _, l := range seq {
		mean += l
	}
	mean /= float64(len(seq))

	result := map[string]any{
		"impl":              impl,
		"docs":              nDocs,
		"load_secs":         round2(loadSecs),
		"load_docs_per_sec": round2(float64(nDocs) / loadSecs),
		"seq_queries":       len(seq),
		"seq_mean_ms":       round3(mean),
		"seq_p50_ms":        round3(percentile(seq, 0.50)),
		"seq_p95_ms":        round3(percentile(seq, 0.95)),
		"seq_p99_ms":        round3(percentile(seq, 0.99)),
		"conc_queries":      concQueries,
		"concurrency":       concurrency,
		"conc_secs":         round2(concSecs),
		"conc_qps":          round2(float64(concQueries) / concSecs),
		"conc_p50_ms":       round3(percentile(conc, 0.50)),
		"conc_p99_ms":       round3(percentile(conc, 0.99)),
	}

	fmt.Printf("== RedisVL for Go ==\n")
	fmt.Printf("load:       %d docs in %.2fs (%.0f docs/s)\n", nDocs, loadSecs, float64(nDocs)/loadSecs)
	fmt.Printf("sequential: %d queries  mean %.3fms  p50 %.3fms  p95 %.3fms  p99 %.3fms\n",
		len(seq), mean, percentile(seq, 0.50), percentile(seq, 0.95), percentile(seq, 0.99))
	fmt.Printf("concurrent: %d queries x %d workers in %.2fs => %.0f qps  p50 %.3fms  p99 %.3fms\n",
		concQueries, concurrency, concSecs, float64(concQueries)/concSecs,
		percentile(conc, 0.50), percentile(conc, 0.99))

	data, _ := json.Marshal(result)
	fmt.Println(string(data))
}

func round2(f float64) float64 { return math.Round(f*100) / 100 }
func round3(f float64) float64 { return math.Round(f*1000) / 1000 }
