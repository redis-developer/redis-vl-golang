package filter

import "regexp"

// Characters that Redis Search requires us to escape during queries.
// Source: https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/escaping/
var escapedChars = regexp.MustCompile(`[,.<>{}\[\]\\"':;!@#$%^&*()\-+=~/ ?]`)

// Same as above but excludes * and ? to allow wildcard patterns.
var escapedCharsNoWildcard = regexp.MustCompile(`[,.<>{}\[\]\\"':;!@#$%^&()\-+=~/ ]`)

// EscapeToken escapes special characters in a string for use in Redis
// queries. If preserveWildcards is true, * and ? are left intact for
// wildcard matching. Port of redisvl.utils.token_escaper.TokenEscaper.
func EscapeToken(value string, preserveWildcards bool) string {
	re := escapedChars
	if preserveWildcards {
		re = escapedCharsNoWildcard
	}
	return re.ReplaceAllStringFunc(value, func(s string) string {
		return `\` + s
	})
}
