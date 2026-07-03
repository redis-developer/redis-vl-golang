package vectorize

// Live API tests — hit the real OpenAI API and are skipped unless
// OPENAI_API_KEY is set:
//
//	OPENAI_API_KEY=sk-... go test -run TestLive ./extensions/vectorize/ -v

import (
	"context"
	"math"
	"os"
	"testing"
)

func TestLiveOpenAIEmbed(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set; skipping live API test")
	}
	ctx := context.Background()

	v, err := NewOpenAIVectorizer(ctx, OpenAIConfig{Model: "text-embedding-3-small"})
	if err != nil {
		t.Fatal(err)
	}

	// text-embedding-3-small produces 1536-dim vectors; the constructor
	// probe should have detected that
	if v.Dims() != 1536 {
		t.Errorf("dims = %d, want 1536", v.Dims())
	}

	emb, err := v.Embed(ctx, "the quick brown fox")
	if err != nil {
		t.Fatal(err)
	}
	if len(emb) != v.Dims() {
		t.Errorf("embedding length %d != dims %d", len(emb), v.Dims())
	}

	// batch call preserves order and count
	embs, err := v.EmbedMany(ctx, []string{"first text", "second text", "third text"})
	if err != nil {
		t.Fatal(err)
	}
	if len(embs) != 3 {
		t.Fatalf("got %d embeddings, want 3", len(embs))
	}

	// semantic sanity: identical texts embed identically, and a related
	// text is closer than an unrelated one
	same, err := v.Embed(ctx, "the quick brown fox")
	if err != nil {
		t.Fatal(err)
	}
	if cosineSim(emb, same) < 0.999 {
		t.Errorf("identical texts should have ~1.0 similarity, got %f", cosineSim(emb, same))
	}
	related, _ := v.Embed(ctx, "a fast auburn fox")
	unrelated, _ := v.Embed(ctx, "quarterly financial report for the board")
	if cosineSim(emb, related) <= cosineSim(emb, unrelated) {
		t.Errorf("related text (%f) should be closer than unrelated (%f)",
			cosineSim(emb, related), cosineSim(emb, unrelated))
	}
}

func cosineSim(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
