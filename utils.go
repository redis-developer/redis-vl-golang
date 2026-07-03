package redisvl

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"time"
)

const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// NewULID generates a ULID string (48-bit millisecond timestamp + 80 bits of
// randomness, Crockford base32). Used for auto-generated document IDs.
func NewULID() string {
	var out [26]byte

	// 48-bit timestamp -> 10 chars
	t := uint64(time.Now().UnixMilli()) & ((1 << 48) - 1)
	for i := 9; i >= 0; i-- {
		out[i] = crockford[t&31]
		t >>= 5
	}

	// 80 random bits -> 16 chars
	var buf [10]byte
	if _, err := rand.Read(buf[:]); err != nil {
		panic(fmt.Sprintf("redisvl: crypto/rand failed: %v", err))
	}
	var acc uint64
	bits := 0
	pos := 10
	for _, b := range buf {
		acc = acc<<8 | uint64(b)
		bits += 8
		for bits >= 5 {
			bits -= 5
			out[pos] = crockford[(acc>>uint(bits))&31]
			pos++
		}
	}
	return string(out[:])
}

// Hashify creates a SHA-256 hex digest of the content plus optional sorted
// extras (port of redisvl.redis.utils.hashify).
func Hashify(content string, extras map[string]any) string {
	if len(extras) > 0 {
		keys := make([]string, 0, len(extras))
		for k := range extras {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			content += " " + k + fmt.Sprint(extras[k])
		}
	}
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

// toString renders any Redis reply scalar as a string.
func toString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case []byte:
		return string(s)
	case int64:
		return strconv.FormatInt(s, 10)
	case float64:
		return strconv.FormatFloat(s, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(s)
	case nil:
		return ""
	}
	return fmt.Sprint(v)
}

// asMap converts a RESP3 map reply to map[string]any.
func asMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case map[any]any:
		out := make(map[string]any, len(m))
		for k, val := range m {
			out[toString(k)] = val
		}
		return out, true
	}
	return nil, false
}

func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int64:
		return n, true
	case int:
		return int64(n), true
	case float64:
		return int64(n), true
	case string:
		i, err := strconv.ParseInt(n, 10, 64)
		return i, err == nil
	}
	return 0, false
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int64:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	}
	return 0, false
}
