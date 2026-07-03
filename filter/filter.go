// Package filter implements a fluent builder for Redis Search filter
// expressions — the Go equivalent of redisvl.query.filter, which in Python
// uses operator overloading (==, !=, &, |). In Go the same expressions are
// built with methods:
//
//	f := filter.Tag("brand").Eq("nike").And(filter.Num("price").Lt(100))
//	f.String() // (@brand:{nike} @price:[-inf (100])
package filter

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Inclusive controls which sides of a Between range are inclusive.
type Inclusive string

const (
	Both    Inclusive = "both"
	Neither Inclusive = "neither"
	Left    Inclusive = "left"
	Right   Inclusive = "right"
)

// Expression is a Redis Search filter expression, possibly the logical
// combination of other expressions. The zero value is invalid; use the
// field builders (Tag, Num, Text, Geo, Timestamp), All, or Raw.
type Expression struct {
	raw   string // leaf expression
	op    string // " " (AND) or " | " (OR) for combined expressions
	left  *Expression
	right *Expression
	err   error // deferred construction error, surfaced at query time
}

// Err returns any error recorded while building this expression (e.g. an
// invalid geo unit), including errors from combined sub-expressions.
// Queries surface this error when executed.
func (e *Expression) Err() error {
	if e == nil {
		return nil
	}
	if e.err != nil {
		return e.err
	}
	if e.left != nil {
		if err := e.left.Err(); err != nil {
			return err
		}
	}
	if e.right != nil {
		return e.right.Err()
	}
	return nil
}

// All returns the match-everything expression ("*").
func All() *Expression { return &Expression{raw: "*"} }

// Raw wraps a raw Redis Search query string as an Expression.
func Raw(s string) *Expression { return &Expression{raw: s} }

// And combines this expression with others using logical AND (intersection).
func (e *Expression) And(others ...*Expression) *Expression {
	out := e
	for _, o := range others {
		out = &Expression{op: " ", left: out, right: o}
	}
	return out
}

// Or combines this expression with others using logical OR (union).
func (e *Expression) Or(others ...*Expression) *Expression {
	out := e
	for _, o := range others {
		out = &Expression{op: " | ", left: out, right: o}
	}
	return out
}

// String renders the expression in the Redis query language.
func (e *Expression) String() string {
	if e == nil {
		return "*"
	}
	if e.op != "" {
		l, r := e.left.String(), e.right.String()
		switch {
		case l == "*" && r == "*":
			return "*"
		case l == "*":
			return r
		case r == "*":
			return l
		}
		return "(" + l + e.op + r + ")"
	}
	if e.raw == "" {
		return "*"
	}
	return e.raw
}

func missing(field string) *Expression {
	return &Expression{raw: fmt.Sprintf("ismissing(@%s)", field)}
}

// formatNum renders a number the way Python's str() does for ints/floats
// (100 not 100.000000, 0.2 not 0.200000).
func formatNum(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// ---------------------------------------------------------------- Tag

// TagField builds filters on TAG fields.
type TagField struct{ field string }

// Tag starts a filter on a TAG field.
func Tag(field string) *TagField { return &TagField{field} }

func (t *TagField) formatValues(values []string, preserveWildcards bool) string {
	escaped := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" {
			continue
		}
		escaped = append(escaped, EscapeToken(v, preserveWildcards))
	}
	return strings.Join(escaped, "|")
}

// Eq matches documents whose tag field contains any of the given values.
func (t *TagField) Eq(values ...string) *Expression {
	formatted := t.formatValues(values, false)
	if formatted == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf("@%s:{%s}", t.field, formatted)}
}

// Ne matches documents whose tag field contains none of the given values.
func (t *TagField) Ne(values ...string) *Expression {
	formatted := t.formatValues(values, false)
	if formatted == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf("(-@%s:{%s})", t.field, formatted)}
}

// Like matches tag values against wildcard patterns; * and ? are preserved
// (e.g. "tech*", "*tech", "*tech*"). Port of Python's Tag % operator.
func (t *TagField) Like(patterns ...string) *Expression {
	formatted := t.formatValues(patterns, true)
	if formatted == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf("@%s:{%s}", t.field, formatted)}
}

// IsMissing matches documents that do not have this field.
func (t *TagField) IsMissing() *Expression { return missing(t.field) }

// ---------------------------------------------------------------- Num

// NumField builds filters on NUMERIC fields.
type NumField struct{ field string }

// Num starts a filter on a NUMERIC field.
func Num(field string) *NumField { return &NumField{field} }

// Eq matches documents where the field equals v.
func (n *NumField) Eq(v float64) *Expression {
	s := formatNum(v)
	return &Expression{raw: fmt.Sprintf("@%s:[%s %s]", n.field, s, s)}
}

// Ne matches documents where the field does not equal v.
func (n *NumField) Ne(v float64) *Expression {
	s := formatNum(v)
	return &Expression{raw: fmt.Sprintf("(-@%s:[%s %s])", n.field, s, s)}
}

// Gt matches documents where the field is greater than v.
func (n *NumField) Gt(v float64) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[(%s +inf]", n.field, formatNum(v))}
}

// Lt matches documents where the field is less than v.
func (n *NumField) Lt(v float64) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[-inf (%s]", n.field, formatNum(v))}
}

// Ge matches documents where the field is greater than or equal to v.
func (n *NumField) Ge(v float64) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[%s +inf]", n.field, formatNum(v))}
}

// Le matches documents where the field is less than or equal to v.
func (n *NumField) Le(v float64) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[-inf %s]", n.field, formatNum(v))}
}

// Between matches documents where the field is between start and end.
func (n *NumField) Between(start, end float64, inclusive Inclusive) *Expression {
	return &Expression{raw: formatBetween(n.field, formatNum(start), formatNum(end), inclusive)}
}

// IsMissing matches documents that do not have this field.
func (n *NumField) IsMissing() *Expression { return missing(n.field) }

func formatBetween(field, start, end string, inclusive Inclusive) string {
	switch inclusive {
	case Neither:
		return fmt.Sprintf("@%s:[(%s (%s]", field, start, end)
	case Left:
		return fmt.Sprintf("@%s:[%s (%s]", field, start, end)
	case Right:
		return fmt.Sprintf("@%s:[(%s %s]", field, start, end)
	default: // Both
		return fmt.Sprintf("@%s:[%s %s]", field, start, end)
	}
}

// ---------------------------------------------------------------- Text

// TextField builds filters on TEXT fields.
type TextField struct{ field string }

// Text starts a filter on a TEXT field.
func Text(field string) *TextField { return &TextField{field} }

// Eq matches documents with an exact match on the supplied term(s).
func (t *TextField) Eq(value string) *Expression {
	if value == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf(`@%s:("%s")`, t.field, value)}
}

// Ne matches documents without an exact match on the supplied term(s).
func (t *TextField) Ne(value string) *Expression {
	if value == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf(`(-@%s:"%s")`, t.field, value)}
}

// Like is a flexible full-text match supporting wildcards ("engine*"),
// fuzzy matches ("%%engine%%"), and combinations ("engineer|doctor").
// Port of Python's Text % operator.
func (t *TextField) Like(value string) *Expression {
	if value == "" {
		return All()
	}
	return &Expression{raw: fmt.Sprintf("@%s:(%s)", t.field, value)}
}

// IsMissing matches documents that do not have this field.
func (t *TextField) IsMissing() *Expression { return missing(t.field) }

// ---------------------------------------------------------------- Geo

// GeoRadius is a geographic radius (lon, lat, radius, unit).
type GeoRadius struct {
	Longitude float64
	Latitude  float64
	Radius    int
	Unit      string // "m", "km", "mi", "ft"
}

func (g GeoRadius) validate() error {
	switch strings.ToLower(g.Unit) {
	case "m", "km", "mi", "ft":
		return nil
	}
	return fmt.Errorf("geo unit must be one of m, km, mi, ft; got %q", g.Unit)
}

// GeoField builds filters on GEO fields.
type GeoField struct{ field string }

// Geo starts a filter on a GEO field.
func Geo(field string) *GeoField { return &GeoField{field} }

func (g *GeoField) radius(r GeoRadius, negate bool) *Expression {
	if err := r.validate(); err != nil {
		// Defer the error: it surfaces via Expression.Err when the
		// expression is used in a query.
		return &Expression{raw: "*", err: err}
	}
	body := fmt.Sprintf("@%s:[%s %s %d %s]",
		g.field, formatNum(r.Longitude), formatNum(r.Latitude), r.Radius,
		strings.ToLower(r.Unit))
	if negate {
		return &Expression{raw: "(-" + body + ")"}
	}
	return &Expression{raw: body}
}

// Within matches documents whose geo field falls inside the radius.
func (g *GeoField) Within(r GeoRadius) *Expression { return g.radius(r, false) }

// NotWithin matches documents whose geo field falls outside the radius.
func (g *GeoField) NotWithin(r GeoRadius) *Expression { return g.radius(r, true) }

// IsMissing matches documents that do not have this field.
func (g *GeoField) IsMissing() *Expression { return missing(g.field) }

// ---------------------------------------------------------------- Timestamp

// TimestampField builds filters on numeric fields that store Unix
// timestamps. All times are converted to Unix seconds in UTC.
type TimestampField struct{ field string }

// Timestamp starts a filter on a timestamp (numeric) field.
func Timestamp(field string) *TimestampField { return &TimestampField{field} }

func ts(t time.Time) string {
	sec := float64(t.UTC().UnixNano()) / 1e9
	return strconv.FormatFloat(sec, 'f', -1, 64)
}

// Eq matches timestamps equal to t.
func (f *TimestampField) Eq(t time.Time) *Expression {
	s := ts(t)
	return &Expression{raw: fmt.Sprintf("@%s:[%s %s]", f.field, s, s)}
}

// Ne matches timestamps not equal to t.
func (f *TimestampField) Ne(t time.Time) *Expression {
	s := ts(t)
	return &Expression{raw: fmt.Sprintf("(-@%s:[%s %s])", f.field, s, s)}
}

// Gt matches timestamps after t.
func (f *TimestampField) Gt(t time.Time) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[(%s +inf]", f.field, ts(t))}
}

// Lt matches timestamps before t.
func (f *TimestampField) Lt(t time.Time) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[-inf (%s]", f.field, ts(t))}
}

// Ge matches timestamps at or after t.
func (f *TimestampField) Ge(t time.Time) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[%s +inf]", f.field, ts(t))}
}

// Le matches timestamps at or before t.
func (f *TimestampField) Le(t time.Time) *Expression {
	return &Expression{raw: fmt.Sprintf("@%s:[-inf %s]", f.field, ts(t))}
}

// Between matches timestamps between start and end.
func (f *TimestampField) Between(start, end time.Time, inclusive Inclusive) *Expression {
	return &Expression{raw: formatBetween(f.field, ts(start), ts(end), inclusive)}
}

// OnDate matches any timestamp within the calendar day of t (UTC).
// Port of Python's Timestamp == date behavior.
func (f *TimestampField) OnDate(t time.Time) *Expression {
	u := t.UTC()
	start := time.Date(u.Year(), u.Month(), u.Day(), 0, 0, 0, 0, time.UTC)
	// end of day at microsecond precision, matching Python's datetime.time.max
	end := start.Add(24*time.Hour - time.Microsecond)
	return f.Between(start, end, Both)
}

// IsMissing matches documents that do not have this field.
func (f *TimestampField) IsMissing() *Expression { return missing(f.field) }
