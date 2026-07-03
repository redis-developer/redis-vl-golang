package hf

import (
	"math"
	"testing"
)

func almostEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func TestPoolMean(t *testing.T) {
	// 3 tokens x 2 dims; third token is padding (mask 0)
	tokens := []float32{
		1, 2,
		3, 4,
		100, 100,
	}
	mask := []int{1, 1, 0}
	got := poolTokens(tokens, 3, 2, mask, poolMean)
	want := []float64{2, 3}
	if !almostEqual(got, want, 1e-9) {
		t.Errorf("mean pool = %v, want %v", got, want)
	}
}

func TestPoolMeanAllMasked(t *testing.T) {
	tokens := []float32{1, 2, 3, 4}
	got := poolTokens(tokens, 2, 2, []int{0, 0}, poolMean)
	// sum is 0, count clamped to 1e-9 -> zeros
	if !almostEqual(got, []float64{0, 0}, 1e-9) {
		t.Errorf("all-masked mean pool = %v, want zeros", got)
	}
}

func TestPoolCLS(t *testing.T) {
	tokens := []float32{
		7, 8,
		1, 1,
	}
	got := poolTokens(tokens, 2, 2, []int{1, 1}, poolCLS)
	if !almostEqual(got, []float64{7, 8}, 1e-9) {
		t.Errorf("cls pool = %v, want [7 8]", got)
	}
}

func TestPoolMax(t *testing.T) {
	tokens := []float32{
		1, 9,
		5, 2,
		100, 100, // masked out
	}
	got := poolTokens(tokens, 3, 2, []int{1, 1, 0}, poolMax)
	if !almostEqual(got, []float64{5, 9}, 1e-9) {
		t.Errorf("max pool = %v, want [5 9]", got)
	}
}

func TestL2Normalize(t *testing.T) {
	v := []float64{3, 4}
	l2Normalize(v)
	if !almostEqual(v, []float64{0.6, 0.8}, 1e-12) {
		t.Errorf("normalize = %v, want [0.6 0.8]", v)
	}

	zero := []float64{0, 0}
	l2Normalize(zero)
	if !almostEqual(zero, []float64{0, 0}, 0) {
		t.Errorf("zero vector changed: %v", zero)
	}

	// normalizing twice is a no-op (relied upon when a sentence-transformers
	// ONNX export already normalizes inside the graph)
	l2Normalize(v)
	if !almostEqual(v, []float64{0.6, 0.8}, 1e-12) {
		t.Errorf("double normalize = %v, want [0.6 0.8]", v)
	}
}
