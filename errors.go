package redisvl

import (
	"errors"
	"strings"
)

// ErrIndexNotFound is returned (wrapped) when an operation targets a search
// index that does not exist in Redis. Test with errors.Is.
var ErrIndexNotFound = errors.New("index not found")

// isUnknownIndexError reports whether a Redis error indicates a missing
// index (message varies across server versions).
func isUnknownIndexError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unknown index") ||
		strings.Contains(msg, "no such index") ||
		strings.Contains(msg, "unknown: index name")
}

// errWithErr is implemented by query builders that defer construction
// errors (invalid dtype, invalid filter) until execution.
type errWithErr interface{ Err() error }

// queryErr extracts a deferred construction error from a query, if any.
func queryErr(q any) error {
	if e, ok := q.(errWithErr); ok {
		return e.Err()
	}
	return nil
}
