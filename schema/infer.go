package schema

import (
	"regexp"
	"strconv"
)

// geoPattern matches "lat,lon" coordinate strings (port of
// redisvl.schema.type_utils.TypeInferrer.GEO_PATTERN).
var geoPattern = regexp.MustCompile(
	`^\s*[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)\s*$`)

// InferFieldType infers a schema field type from a sample value, using the
// same heuristics as the Python TypeInferrer: numeric, geo, tag, then text.
// Returns "" if no type can be inferred.
func InferFieldType(value any) FieldType {
	switch v := value.(type) {
	case int, int32, int64, float32, float64:
		return TypeNumeric
	case string:
		if _, err := strconv.ParseFloat(v, 64); err == nil {
			return TypeNumeric
		}
		if geoPattern.MatchString(v) {
			return TypeGeo
		}
		return TypeText
	case []string:
		return TypeTag
	case []any:
		for _, item := range v {
			if _, ok := item.(string); !ok {
				return ""
			}
		}
		return TypeTag
	}
	return ""
}

// GenerateFields infers field definitions from a sample data point. Vector
// fields are never generated. Fields whose type cannot be inferred are
// skipped (or cause an error in strict mode via the returned slice being
// incomplete — callers wanting strictness should check coverage).
func GenerateFields(data map[string]any, ignore ...string) []Field {
	skip := map[string]bool{}
	for _, n := range ignore {
		skip[n] = true
	}
	var out []Field
	for name, value := range data {
		if skip[name] {
			continue
		}
		switch InferFieldType(value) {
		case TypeNumeric:
			out = append(out, NewNumericField(name))
		case TypeGeo:
			out = append(out, NewGeoField(name))
		case TypeTag:
			out = append(out, NewTagField(name))
		case TypeText:
			out = append(out, NewTextField(name))
		}
	}
	return out
}
