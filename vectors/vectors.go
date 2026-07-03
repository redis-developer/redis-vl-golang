// Package vectors provides conversion between float slices and the binary
// buffer format expected by Redis vector fields (port of redisvl.redis.utils
// array_to_buffer / buffer_to_array).
package vectors

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// DataType is the numeric datatype of a vector field.
type DataType string

// Supported vector datatypes, matching the DATATYPE values accepted by
// Redis vector fields.
const (
	BFloat16 DataType = "bfloat16"
	Float16  DataType = "float16"
	Float32  DataType = "float32"
	Float64  DataType = "float64"
	Int8     DataType = "int8"
	Uint8    DataType = "uint8"
)

// Parse normalizes and validates a datatype string.
func Parse(dtype string) (DataType, error) {
	dt := DataType(strings.ToLower(dtype))
	switch dt {
	case BFloat16, Float16, Float32, Float64, Int8, Uint8:
		return dt, nil
	}
	return "", fmt.Errorf(
		"invalid data type: %s; supported types are: bfloat16, float16, float32, float64, int8, uint8",
		dtype,
	)
}

// ToBuffer converts a slice of floats into the little-endian byte buffer
// Redis expects for the given vector datatype.
func ToBuffer(values []float64, dtype DataType) ([]byte, error) {
	dt, err := Parse(string(dtype))
	if err != nil {
		return nil, err
	}
	switch dt {
	case Float32:
		buf := make([]byte, 4*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
		}
		return buf, nil
	case Float64:
		buf := make([]byte, 8*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		return buf, nil
	case Float16:
		buf := make([]byte, 2*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint16(buf[i*2:], Float32ToFloat16(float32(v)))
		}
		return buf, nil
	case BFloat16:
		buf := make([]byte, 2*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint16(buf[i*2:], Float32ToBFloat16(float32(v)))
		}
		return buf, nil
	case Int8:
		buf := make([]byte, len(values))
		for i, v := range values {
			buf[i] = byte(int8(v))
		}
		return buf, nil
	case Uint8:
		buf := make([]byte, len(values))
		for i, v := range values {
			buf[i] = uint8(v)
		}
		return buf, nil
	}
	return nil, fmt.Errorf("unsupported data type: %s", dtype)
}

// ToBuffer32 is a convenience wrapper for []float32 input.
func ToBuffer32(values []float32, dtype DataType) ([]byte, error) {
	f64 := make([]float64, len(values))
	for i, v := range values {
		f64[i] = float64(v)
	}
	return ToBuffer(f64, dtype)
}

// Size returns the byte width of one element of the given datatype.
func Size(dtype DataType) (int, error) {
	dt, err := Parse(string(dtype))
	if err != nil {
		return 0, err
	}
	return map[DataType]int{
		Float32: 4, Float64: 8, Float16: 2, BFloat16: 2, Int8: 1, Uint8: 1,
	}[dt], nil
}

// FromBuffer converts a little-endian byte buffer read from Redis back into a
// slice of floats for the given vector datatype.
func FromBuffer(buf []byte, dtype DataType) ([]float64, error) {
	dt, err := Parse(string(dtype))
	if err != nil {
		return nil, err
	}
	size, _ := Size(dt)
	if len(buf)%size != 0 {
		return nil, fmt.Errorf("buffer length %d is not a multiple of element size %d for %s",
			len(buf), size, dtype)
	}
	n := len(buf) / size
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		chunk := buf[i*size:]
		switch dt {
		case Float32:
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(chunk)))
		case Float64:
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(chunk))
		case Float16:
			out[i] = float64(Float16ToFloat32(binary.LittleEndian.Uint16(chunk)))
		case BFloat16:
			out[i] = float64(BFloat16ToFloat32(binary.LittleEndian.Uint16(chunk)))
		case Int8:
			out[i] = float64(int8(chunk[0]))
		case Uint8:
			out[i] = float64(chunk[0])
		}
	}
	return out, nil
}

// Float32ToBFloat16 converts a float32 to bfloat16 bits with
// round-to-nearest-even. bfloat16 is the upper 16 bits of a float32.
func Float32ToBFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 { // NaN: keep it a NaN after truncation
		return uint16(b>>16) | 0x0040
	}
	b += 0x7fff + (b >> 16 & 1) // round to nearest even
	return uint16(b >> 16)
}

// BFloat16ToFloat32 converts bfloat16 bits to a float32.
func BFloat16ToFloat32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

// Float32ToFloat16 converts a float32 to IEEE 754 half-precision bits with
// round-to-nearest-even.
func Float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp32 := int32(b >> 23 & 0xff)
	mant := b & 0x7fffff

	if exp32 == 0xff { // Inf or NaN
		if mant != 0 {
			return sign | 0x7e00 // NaN
		}
		return sign | 0x7c00 // Inf
	}

	exp := exp32 - 127 + 15
	if exp >= 0x1f { // overflow -> Inf
		return sign | 0x7c00
	}
	if exp <= 0 { // subnormal or zero
		if exp < -10 {
			return sign // underflow to zero
		}
		m := mant | 0x800000
		shift := uint32(14 - exp)
		halfMant := m >> shift
		roundMask := uint32(1) << (shift - 1)
		if m&roundMask != 0 && (m&(roundMask-1) != 0 || halfMant&1 == 1) {
			halfMant++
		}
		return sign | uint16(halfMant)
	}

	halfMant := mant >> 13
	if mant&0x1000 != 0 && (mant&0x0fff != 0 || halfMant&1 == 1) {
		halfMant++
		if halfMant == 0x400 { // mantissa overflow bumps exponent
			halfMant = 0
			exp++
			if exp >= 0x1f {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exp)<<10 | uint16(halfMant)
}

// Float16ToFloat32 converts IEEE 754 half-precision bits to a float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // +/- zero
		}
		// subnormal: normalize
		e := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (113-e)<<23 | mant<<13)
	case 0x1f: // Inf or NaN
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}
