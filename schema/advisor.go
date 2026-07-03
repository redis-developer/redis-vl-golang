package schema

import (
	"fmt"
	"math"
)

// Priority is the optimization goal for compression recommendations.
type Priority string

const (
	// Speed optimizes for query latency.
	Speed Priority = "speed"
	// Memory maximizes memory savings.
	Memory Priority = "memory"
	// Balanced balances memory and speed (default).
	Balanced Priority = "balanced"
)

// compressionBits maps compression types to bits per dimension (port of
// CompressionAdvisor.COMPRESSION_BITS).
var compressionBits = map[Compression]int{
	LVQ4:       4,
	LVQ4x4:     8,
	LVQ4x8:     12,
	LVQ8:       8,
	LeanVec4x8: 12,
	LeanVec8x8: 16,
}

// highDimThreshold is the dimensionality at which LeanVec compression is
// preferred over LVQ.
const highDimThreshold = 1024

// RecommendCompression suggests SVS-VAMANA vector attributes based on
// vector dimensionality and an optimization priority (port of
// redisvl.utils.CompressionAdvisor.recommend). The returned attributes can
// be passed directly to NewVectorField. Pass a datatype to override the
// default (float16 for high-dimensional vectors, float32 otherwise).
func RecommendCompression(dims int, priority Priority, datatype ...string) (VectorAttrs, error) {
	if dims <= 0 {
		return VectorAttrs{}, fmt.Errorf("dims must be positive, got %d", dims)
	}
	dt := ""
	if len(datatype) > 0 {
		dt = datatype[0]
	}

	attrs := VectorAttrs{
		Dims:      dims,
		Algorithm: SVSVamana,
	}

	if dims >= highDimThreshold {
		// High-dimensional vectors - use LeanVec
		if dt == "" {
			dt = "float16"
		}
		attrs.Datatype = dt
		attrs.GraphMaxDegree = IntPtr(64)
		attrs.ConstructionWindowSize = IntPtr(300)
		attrs.Compression = LeanVec4x8

		switch priority {
		case Memory:
			attrs.Reduce = IntPtr(dims / 2)
			attrs.SearchWindowSize = IntPtr(20)
		case Speed:
			r := dims / 4
			if r < 256 {
				r = 256
			}
			attrs.Reduce = IntPtr(r)
			attrs.SearchWindowSize = IntPtr(40)
		default: // Balanced
			attrs.Reduce = IntPtr(dims / 2)
			attrs.SearchWindowSize = IntPtr(30)
		}
	} else {
		// Lower-dimensional vectors - use LVQ
		if dt == "" {
			dt = "float32"
		}
		attrs.Datatype = dt
		attrs.GraphMaxDegree = IntPtr(40)
		attrs.ConstructionWindowSize = IntPtr(250)
		attrs.SearchWindowSize = IntPtr(20)

		switch priority {
		case Memory:
			attrs.Compression = LVQ4
		case Speed:
			attrs.Compression = LVQ4x8
		default: // Balanced
			attrs.Compression = LVQ4x4
		}
	}

	if err := attrs.Validate(); err != nil {
		return VectorAttrs{}, err
	}
	return attrs, nil
}

// EstimateMemorySavings estimates the memory savings percentage of a
// compression type versus uncompressed float32 vectors (port of
// CompressionAdvisor.estimate_memory_savings). reduce is the reduced
// dimensionality for LeanVec types; pass 0 when not reducing.
func EstimateMemorySavings(compression Compression, dims, reduce int) float64 {
	const baseBits = 32

	bits, ok := compressionBits[compression]
	if !ok {
		bits = baseBits
	}
	effectiveDims := dims
	if reduce > 0 {
		effectiveDims = reduce
	}

	originalSize := float64(dims * baseBits)
	compressedSize := float64(effectiveDims * bits)
	savings := (1 - compressedSize/originalSize) * 100

	// round-half-to-even to one decimal, matching Python's round()
	return math.RoundToEven(savings*10) / 10
}
