// End-to-end demo: SemanticCache with real OpenAI embeddings.
//
// Requires a running Redis Stack / Redis 8 and OPENAI_API_KEY:
//
//	OPENAI_API_KEY=sk-... REDIS_URL=redis://localhost:6379 \
//	  go run ./examples/semantic_cache_openai
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/redis-developer/redis-vl-golang/extensions/cache"
	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
)

func main() {
	ctx := context.Background()

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://localhost:6379"
	}
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		log.Fatal(err)
	}
	client := redis.NewClient(opts)
	defer client.Close()

	fmt.Println("creating OpenAI vectorizer (probing dimensions)...")
	vectorizer, err := vectorize.NewOpenAIVectorizer(ctx, vectorize.OpenAIConfig{
		Model: "text-embedding-3-small",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("model=%s dims=%d\n\n", vectorizer.ModelName(), vectorizer.Dims())

	llmCache, err := cache.NewSemanticCache(ctx, client, vectorizer, cache.SemanticCacheOptions{
		Name:              "demo-llmcache",
		DistanceThreshold: 0.2,
		TTL:               time.Hour,
		Overwrite:         true,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer llmCache.Delete(ctx)

	// store a response
	key, err := llmCache.Store(ctx,
		"What is the capital of France?", "The capital of France is Paris.",
		cache.StoreOptions{Metadata: map[string]any{"model": "demo"}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("stored entry: %s\n\n", key)

	// semantically similar phrasing -> cache hit
	for _, prompt := range []string{
		"What's France's capital city?",
		"Tell me the capital of France",
		"How do I bake sourdough bread?", // unrelated -> miss
	} {
		hits, err := llmCache.Check(ctx, prompt)
		if err != nil {
			log.Fatal(err)
		}
		if len(hits) > 0 {
			fmt.Printf("HIT  %-40q -> %q (distance %.4f)\n",
				prompt, hits[0].Response, hits[0].VectorDistance)
		} else {
			fmt.Printf("MISS %-40q\n", prompt)
		}
	}
}
