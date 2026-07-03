package vectors

import (
	"math"
	"testing"
)

func TestFloat16KnownValues(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{0, 0x0000},
		{1.0, 0x3C00},
		{-2.0, 0xC000},
		{0.5, 0x3800},
		{65504, 0x7BFF},
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
		{1e10, 0x7C00}, // overflow -> inf
	}
	for _, c := range cases {
		if got := Float32ToFloat16(c.f); got != c.want {
			t.Errorf("Float32ToFloat16(%v) = %#04x, want %#04x", c.f, got, c.want)
		}
	}
}

func TestFloat16Roundtrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, 0.25, 2, 1024, 65504, 6.1035156e-05, 5.9604645e-08}
	for _, v := range values {
		got := Float16ToFloat32(Float32ToFloat16(v))
		if got != v {
			t.Errorf("float16 roundtrip: %v -> %v", v, got)
		}
	}
}

func TestFloat16NaN(t *testing.T) {
	h := Float32ToFloat16(float32(math.NaN()))
	back := float64(Float16ToFloat32(h))
	if !math.IsNaN(back) {
		t.Errorf("NaN did not roundtrip: %#04x -> %v", h, back)
	}
}

func TestBFloat16KnownValues(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{0, 0x0000},
		{1.0, 0x3F80},
		{-2.0, 0xC000},
		{float32(math.Inf(1)), 0x7F80},
	}
	for _, c := range cases {
		if got := Float32ToBFloat16(c.f); got != c.want {
			t.Errorf("Float32ToBFloat16(%v) = %#04x, want %#04x", c.f, got, c.want)
		}
	}
}

func TestBFloat16Roundtrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, 2, 128, -3.140625}
	for _, v := range values {
		got := BFloat16ToFloat32(Float32ToBFloat16(v))
		if got != v {
			t.Errorf("bfloat16 roundtrip: %v -> %v", v, got)
		}
	}
}

func TestBufferRoundtrips(t *testing.T) {
	vec := []float64{0.1, 0.2, 0.3, -0.4}
	for _, dt := range []DataType{Float32, Float64, Float16, BFloat16} {
		buf, err := ToBuffer(vec, dt)
		if err != nil {
			t.Fatalf("ToBuffer(%s): %v", dt, err)
		}
		out, err := FromBuffer(buf, dt)
		if err != nil {
			t.Fatalf("FromBuffer(%s): %v", dt, err)
		}
		if len(out) != len(vec) {
			t.Fatalf("%s: length mismatch %d != %d", dt, len(out), len(vec))
		}
		tolerance := map[DataType]float64{
			Float32: 1e-7, Float64: 0, Float16: 1e-3, BFloat16: 1e-2,
		}[dt]
		for i := range vec {
			if math.Abs(out[i]-vec[i]) > tolerance {
				t.Errorf("%s[%d]: got %v want %v (tol %v)", dt, i, out[i], vec[i], tolerance)
			}
		}
	}
}

func TestBufferSizes(t *testing.T) {
	vec := []float64{1, 2, 3}
	sizes := map[DataType]int{
		Float32: 12, Float64: 24, Float16: 6, BFloat16: 6, Int8: 3, Uint8: 3,
	}
	for dt, want := range sizes {
		buf, err := ToBuffer(vec, dt)
		if err != nil {
			t.Fatalf("ToBuffer(%s): %v", dt, err)
		}
		if len(buf) != want {
			t.Errorf("%s: buffer size %d, want %d", dt, len(buf), want)
		}
	}
}

func TestInvalidDtype(t *testing.T) {
	if _, err := ToBuffer([]float64{1}, "float8"); err == nil {
		t.Error("expected error for invalid dtype")
	}
	if _, err := FromBuffer([]byte{0}, "whatever"); err == nil {
		t.Error("expected error for invalid dtype")
	}
}
