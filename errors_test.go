package redisvl

import (
	"errors"
	"fmt"
	"testing"
)

func TestIsUnknownIndexError(t *testing.T) {
	cases := []struct {
		msg  string
		want bool
	}{
		{"Unknown index name", true},
		{"unknown index name", true},
		{"no such index", true},
		{"Unknown: Index name", true},
		{"connection refused", false},
		{"syntax error", false},
	}
	for _, c := range cases {
		if got := isUnknownIndexError(errors.New(c.msg)); got != c.want {
			t.Errorf("isUnknownIndexError(%q) = %v, want %v", c.msg, got, c.want)
		}
	}
	if isUnknownIndexError(nil) {
		t.Error("nil error should not match")
	}
}

func TestErrIndexNotFoundWrapping(t *testing.T) {
	wrapped := fmt.Errorf("index %q: %w", "myindex", ErrIndexNotFound)
	if !errors.Is(wrapped, ErrIndexNotFound) {
		t.Error("errors.Is should match wrapped ErrIndexNotFound")
	}
}
