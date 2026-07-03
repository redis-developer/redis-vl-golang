package hf

import "math"

// poolTokens reduces per-token embeddings ([seqLen x hidden], flattened
// row-major) to a single sentence embedding using the attention mask,
// implementing sentence-transformers' Pooling module semantics.
func poolTokens(tokens []float32, seqLen, hidden int, mask []int, mode poolingMode) []float64 {
	out := make([]float64, hidden)
	switch mode {
	case poolCLS:
		for j := 0; j < hidden; j++ {
			out[j] = float64(tokens[j])
		}
	case poolMax:
		for j := range out {
			out[j] = math.Inf(-1)
		}
		any := false
		for i := 0; i < seqLen; i++ {
			if i < len(mask) && mask[i] == 0 {
				continue
			}
			any = true
			row := tokens[i*hidden : (i+1)*hidden]
			for j := 0; j < hidden; j++ {
				if v := float64(row[j]); v > out[j] {
					out[j] = v
				}
			}
		}
		if !any {
			for j := range out {
				out[j] = 0
			}
		}
	default: // poolMean
		var count float64
		for i := 0; i < seqLen; i++ {
			if i < len(mask) && mask[i] == 0 {
				continue
			}
			count++
			row := tokens[i*hidden : (i+1)*hidden]
			for j := 0; j < hidden; j++ {
				out[j] += float64(row[j])
			}
		}
		// clamp like sentence-transformers: sum / max(count, 1e-9)
		if count < 1e-9 {
			count = 1e-9
		}
		for j := range out {
			out[j] /= count
		}
	}
	return out
}

// l2Normalize scales v to unit length in place (no-op for zero vectors),
// matching sentence-transformers' Normalize module.
func l2Normalize(v []float64) {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	norm := math.Sqrt(sum)
	for i := range v {
		v[i] /= norm
	}
}
