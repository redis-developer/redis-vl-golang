package hf

import (
	"context"
	"math"
	"os"
	"testing"
)

// TestLiveMiniLM downloads sentence-transformers/all-MiniLM-L6-v2 (~90MB on
// first run) and runs real inference. It requires the onnxruntime shared
// library and network access, so it only runs when RUN_HF_LIVE_TESTS is set:
//
//	RUN_HF_LIVE_TESTS=1 ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.dylib go test ./... -run TestLiveMiniLM -v
func TestLiveMiniLM(t *testing.T) {
	if os.Getenv("RUN_HF_LIVE_TESTS") == "" {
		t.Skip("set RUN_HF_LIVE_TESTS=1 (and ONNXRUNTIME_LIB_PATH) to run live HF tests")
	}
	ctx := context.Background()

	vec, err := New(ctx, Config{Model: "sentence-transformers/all-MiniLM-L6-v2"})
	if err != nil {
		t.Fatal(err)
	}
	defer vec.Close()

	if vec.Dims() != 384 {
		t.Fatalf("dims = %d, want 384", vec.Dims())
	}

	t.Run("unit norm", func(t *testing.T) {
		// MiniLM has a Normalize module, so embeddings are unit length.
		emb, err := vec.Embed(ctx, "hello")
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, x := range emb {
			sum += x * x
		}
		if math.Abs(sum-1) > 1e-3 {
			t.Errorf("norm^2 = %f, want 1", sum)
		}
	})

	t.Run("tokenization", func(t *testing.T) {
		// BERT-uncased reference ids: [CLS] hello [SEP] = [101 7592 102].
		// If the special tokens are missing, the post-processor from
		// tokenizer.json was not applied and every embedding downstream
		// diverges from sentence-transformers.
		enc, err := vec.tk.EncodeSingle("hello", true)
		if err != nil {
			t.Fatal(err)
		}
		want := []int{101, 7592, 102}
		if len(enc.Ids) != len(want) {
			t.Fatalf("token ids = %v, want %v", enc.Ids, want)
		}
		for i, id := range want {
			if enc.Ids[i] != id {
				t.Fatalf("token ids = %v, want %v", enc.Ids, want)
			}
		}
	})

	t.Run("golden components", func(t *testing.T) {
		// Reference values for "hello" from Python:
		// SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
		//     .encode('hello')  (verified 2026-07-03)
		// A mismatch here means our tokenize/pool/normalize pipeline
		// diverges from the sentence-transformers reference.
		emb, err := vec.Embed(ctx, "hello")
		if err != nil {
			t.Fatal(err)
		}
		golden := map[int]float64{
			0:   -0.06277172,
			1:   0.054958835,
			2:   0.05216483,
			382: 0.051483534,
			383: 0.007092172,
		}
		for i, want := range golden {
			if got := emb[i]; math.Abs(got-want) > 2e-3 {
				t.Errorf("emb[%d] = %f, want %f (±2e-3)", i, got, want)
			}
		}
	})

	t.Run("semantic ordering", func(t *testing.T) {
		base, err := vec.Embed(ctx, "The dog is running in the park")
		if err != nil {
			t.Fatal(err)
		}
		candidates, err := vec.EmbedMany(ctx, []string{
			"A dog runs through the park",
			"I love eating pizza for dinner",
		})
		if err != nil {
			t.Fatal(err)
		}
		cos := func(a, b []float64) float64 {
			var dot float64
			for i := range a {
				dot += a[i] * b[i]
			}
			return dot // unit vectors
		}
		similar, unrelated := cos(base, candidates[0]), cos(base, candidates[1])
		if similar <= unrelated {
			t.Errorf("cos(similar)=%f <= cos(unrelated)=%f", similar, unrelated)
		}
		if similar < 0.7 {
			t.Errorf("cos(similar)=%f, expected > 0.7", similar)
		}
	})

	t.Run("batch matches single", func(t *testing.T) {
		// Batching pads shorter texts; padding must not change results.
		single, err := vec.Embed(ctx, "short")
		if err != nil {
			t.Fatal(err)
		}
		batch, err := vec.EmbedMany(ctx, []string{
			"short",
			"a much longer sentence that forces the shorter one to be padded in the batch",
		})
		if err != nil {
			t.Fatal(err)
		}
		var maxDiff float64
		for i := range single {
			if d := math.Abs(single[i] - batch[0][i]); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-4 {
			t.Errorf("batched embedding differs from single by %g", maxDiff)
		}
	})

	t.Run("truncation", func(t *testing.T) {
		// A text far beyond max_seq_length (256) must not error.
		long := ""
		for i := 0; i < 2000; i++ {
			long += "word "
		}
		if _, err := vec.Embed(ctx, long); err != nil {
			t.Fatalf("embedding long text: %v", err)
		}
	})
}

// TestLiveCrossEncoder downloads cross-encoder/ms-marco-MiniLM-L-6-v2
// (~91MB on first run) and reranks with real inference. Gated like
// TestLiveMiniLM (RUN_HF_LIVE_TESTS=1 + onnxruntime).
//
// To pin exact golden scores against Python, run:
//
//	python -c "from sentence_transformers import CrossEncoder; \
//	  m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
//	  print(m.predict([('How many people live in Berlin?', 'Berlin had a population of 3,520,031 registered inhabitants.'), \
//	                   ('How many people live in Berlin?', 'Berlin is well known for its museums.')]))"
func TestLiveCrossEncoder(t *testing.T) {
	if os.Getenv("RUN_HF_LIVE_TESTS") == "" {
		t.Skip("set RUN_HF_LIVE_TESTS=1 (and ONNXRUNTIME_LIB_PATH) to run live HF tests")
	}
	ctx := context.Background()

	ce, err := NewCrossEncoder(ctx, CrossEncoderConfig{Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()

	query := "How many people live in Berlin?"
	docs := []string{
		"Berlin is well known for its museums.",
		"The capital of France is Paris.",
		"Berlin had a population of 3,520,031 registered inhabitants.",
	}
	results, err := ce.Rank(ctx, query, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}

	// The population statement must decisively win.
	if results[0].Index != 2 {
		t.Errorf("top result = %q (index %d), want the population doc", results[0].Document, results[0].Index)
	}
	if last := results[len(results)-1]; last.Score >= results[0].Score {
		t.Errorf("scores not descending: %v", results)
	}

	// Golden raw logits from Python (ms-marco declares Identity activation):
	// CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').predict(...)
	// gave [8.505133, -4.320078] for these pairs (verified 2026-07-03).
	for _, r := range results {
		var want float64
		switch r.Index {
		case 2:
			want = 8.505133
		case 0:
			want = -4.320078
		default:
			continue // no golden for the Paris doc
		}
		if math.Abs(r.Score-want) > 0.02 {
			t.Errorf("score[doc %d] = %f, want %f (±0.02)", r.Index, r.Score, want)
		}
	}

	// Limit is applied after sorting.
	limited, err := ce.Rank(ctx, query, append(docs, "Unrelated text about cooking pasta."))
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != 3 {
		t.Errorf("limit not applied: got %d results", len(limited))
	}
	for _, r := range limited {
		if r.Document == "Unrelated text about cooking pasta." {
			t.Errorf("least relevant doc survived the limit: %v", limited)
		}
	}
}
