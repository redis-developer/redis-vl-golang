package schema

import "testing"

// Expected values mirror the Python CompressionAdvisor docstrings.

func TestRecommendCompressionHighDim(t *testing.T) {
	attrs, err := RecommendCompression(1536, Balanced)
	if err != nil {
		t.Fatal(err)
	}
	if attrs.Compression != LeanVec4x8 {
		t.Errorf("compression = %s, want LeanVec4x8", attrs.Compression)
	}
	if attrs.Reduce == nil || *attrs.Reduce != 768 {
		t.Errorf("reduce = %v, want 768", attrs.Reduce)
	}
	if attrs.Datatype != "float16" || attrs.Algorithm != SVSVamana {
		t.Errorf("attrs = %+v", attrs)
	}
	if *attrs.GraphMaxDegree != 64 || *attrs.ConstructionWindowSize != 300 || *attrs.SearchWindowSize != 30 {
		t.Errorf("graph params = %d/%d/%d", *attrs.GraphMaxDegree, *attrs.ConstructionWindowSize, *attrs.SearchWindowSize)
	}

	// speed clamps reduce at max(256, dims/4)
	attrs, _ = RecommendCompression(1536, Speed)
	if *attrs.Reduce != 384 || *attrs.SearchWindowSize != 40 {
		t.Errorf("speed: reduce=%d window=%d", *attrs.Reduce, *attrs.SearchWindowSize)
	}
	attrs, _ = RecommendCompression(1024, Speed)
	if *attrs.Reduce != 256 { // 1024/4 == 256 == floor
		t.Errorf("speed floor: reduce=%d", *attrs.Reduce)
	}
}

func TestRecommendCompressionLowDim(t *testing.T) {
	cases := map[Priority]Compression{
		Memory:   LVQ4,
		Speed:    LVQ4x8,
		Balanced: LVQ4x4,
	}
	for priority, want := range cases {
		attrs, err := RecommendCompression(384, priority)
		if err != nil {
			t.Fatal(err)
		}
		if attrs.Compression != want {
			t.Errorf("%s: compression = %s, want %s", priority, attrs.Compression, want)
		}
		if attrs.Datatype != "float32" || *attrs.GraphMaxDegree != 40 {
			t.Errorf("%s: attrs = %+v", priority, attrs)
		}
		if attrs.Reduce != nil {
			t.Errorf("%s: LVQ should not set reduce", priority)
		}
	}
}

func TestRecommendCompressionValidation(t *testing.T) {
	if _, err := RecommendCompression(0, Balanced); err == nil {
		t.Error("expected error for dims=0")
	}
	// recommendations must pass VectorAttrs validation and build a field
	attrs, err := RecommendCompression(1536, Memory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewVectorField("embedding", attrs); err != nil {
		t.Errorf("recommended attrs rejected by NewVectorField: %v", err)
	}
}

func TestEstimateMemorySavings(t *testing.T) {
	// Values from the Python docstrings.
	if got := EstimateMemorySavings(LeanVec4x8, 1536, 768); got != 81.2 {
		t.Errorf("LeanVec4x8 1536->768 = %v, want 81.2", got)
	}
	if got := EstimateMemorySavings(LVQ4, 384, 0); got != 87.5 {
		t.Errorf("LVQ4 384 = %v, want 87.5", got)
	}
	// unknown compression -> no savings
	if got := EstimateMemorySavings(Compression("bogus"), 100, 0); got != 0 {
		t.Errorf("unknown compression = %v, want 0", got)
	}
}
